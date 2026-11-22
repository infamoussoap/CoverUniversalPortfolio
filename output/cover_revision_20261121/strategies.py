"""Generate predictable portfolio tapes and time the allocation routines."""

import argparse
from time import perf_counter

import numpy as np
from scipy.linalg import solve
from threadpoolctl import threadpool_limits

from particle_universal.linear_cover import linear_cover
from particle_universal.soft_bayes import soft_bayes
from particle_universal.vb_ftrl import published_steps, vb_ftrl
from study import CACHE, CLASSIC, tape


def simplex(v):
    ranks = np.arange(1, len(v) + 1)
    s = np.sort(v)[::-1]
    shift = np.max((s.cumsum() - 1) / ranks)
    return np.maximum(0, v - shift)


def ons_projection(A, b, active):
    d = len(b)
    for _ in range(4 * d):
        idx = np.flatnonzero(active)
        rhs = np.column_stack((b[idx], np.ones(len(idx))))
        v, h = solve(A[np.ix_(idx, idx)], rhs, assume_a='pos').T
        dual = (v.sum() - 1) / h.sum()
        v -= dual * h
        if v.min() < -1e-13:
            active[idx[v.argmin()]] = False
            continue

        w = np.zeros(d)
        w[idx] = np.maximum(0, v)
        grad = A @ w - b + dual
        grad[active] = np.inf
        if grad.min() >= -1e-10:
            return w / w.sum(), active

        active[grad.argmin()] = True

    raise ArithmeticError('ONS active-set projection failed KKT check')


def toolbox(x, kind):
    T, d = x.shape
    weights = np.empty_like(x)
    w = np.ones(d) / d
    A, b = np.eye(d), np.zeros(d)
    active = np.ones(d, dtype=bool)
    for t, row in enumerate(x):
        weights[t] = w
        if t + 1 == T:
            break
        change = row - row.mean()
        if kind == 'EG':
            w = w * np.exp(0.05 * row / (row @ w))
        elif kind == 'ONS':
            grad = row / (row @ w)
            A += np.outer(grad, grad)
            b += 0.25 * grad
            w, active = ons_projection(A, b, active)
        elif kind == 'PAMR':
            norm = change @ change
            eta = max(0, (row @ w - 0.5) / norm) if norm else 0
            w = simplex(w - eta * change)
        elif kind == 'OLMAR1':
            if t == 0:
                continue
            z = row
            if t + 1 >= 6:
                z = (1 + (1 / x[t-3:t+1][::-1].cumprod(0)).sum(0)) / 5
            centered = z - z.mean()
            norm = centered @ centered
            eta = max(0, (10 - z @ w) / norm) if norm else 0
            w = simplex(w + eta * centered)

        w = w / w.sum()
    return weights


def fixed_sample(x, alpha, seed, M):
    rng = np.random.default_rng(seed)
    pts = rng.dirichlet(np.full(x.shape[1], alpha), size=M)
    mass = np.ones(M) / M
    weights = np.empty_like(x)
    for start in range(0, len(x), 32):
        rows = x[start:start + 32]
        gains = (rows / rows.mean(1, keepdims=True)) @ pts.T
        prefix = np.vstack((mass, gains.cumprod(0) * mass))
        weights[start:start + len(rows)] = (
            prefix[:-1] @ pts / prefix[:-1].sum(1, keepdims=True))
        mass = prefix[-1] / prefix[-1].sum()

    weights[0] = 1 / x.shape[1]
    return weights


def generate(x, kind, alpha, seed):
    if kind == 'first_order':
        return linear_cover(x, alpha=alpha)
    if kind == 'UCRP':
        return np.full_like(x, 1 / x.shape[1])
    if kind == 'Soft-Bayes':
        return soft_bayes(x)
    if kind == 'VB-FTRL':
        return vb_ftrl(x, steps=published_steps(*x.shape), tolerance=-1,
                       skip_cycles=True)['weights']
    if kind.startswith('sample'):
        M = int(kind.split('_')[1])
        return fixed_sample(x, alpha, seed, M)
    return toolbox(x, kind)


def run(name, kinds, repeats):
    x = tape(name)
    for kind in kinds:
        for a in ((1, 0.5) if kind.startswith('sample') or
                  kind in ('first_order', 'UCRP') else (1,)):
            seeds = range(74081, 74086) if kind.startswith('sample') else [0]
            for seed in seeds:
                dest = CACHE / f'w_{name}_{a:g}_{kind}_{seed}.npz'
                times = []
                if dest.exists():
                    with np.load(dest) as data:
                        times = data['times'].tolist()
                while len(times) < repeats:
                    start = perf_counter()
                    w = generate(x, kind, a, seed)
                    times.append(perf_counter() - start)
                    if w.min() < 0 or not np.isfinite(w).all():
                        raise ValueError((name, kind, w.min()))
                    np.testing.assert_allclose(w.sum(1), 1, atol=2e-12, rtol=0)
                    if np.min(np.einsum('ij,ij->i', w, x)) <= 0:
                        raise ValueError('Nonpositive payoff')
                    temp = dest.with_suffix('.part.npz')
                    np.savez(temp, w=w, times=times)
                    temp.replace(dest)
            print(name, a, kind, 'done', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('kinds', nargs='+')
    parser.add_argument('--names', nargs='+', default=CLASSIC)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    with threadpool_limits(1):
        for name in args.names:
            run(name, args.kinds, args.repeats)
