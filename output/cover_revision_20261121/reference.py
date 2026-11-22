"""Eight-scramble quartic control for symmetric Dirichlet priors."""

import sys

import numpy as np
from numba import njit
from scipy import special
from scipy.stats import qmc
from threadpoolctl import threadpool_limits

from particle_universal.sobol import POWER, SCRAMBLES, SEED, polynomial_mass
from particle_universal.market_reference import _prior_particles
from study import CACHE, CLASSIC, tape


def control(a, alpha):
    if alpha == 1:
        return polynomial_mass(a)[:, 1]

    d = a.shape[1]
    r2, r3, r4 = np.cumprod(np.arange(4) + d * alpha)[1:]
    g = np.zeros(d)
    B = np.zeros((d, d))
    cubes = np.zeros(d)
    sizes = np.zeros(d)
    e4 = 0.
    result = np.empty(len(a))
    for t, row in enumerate(a):
        s = row**2
        g += row
        B += np.outer(row, row)
        cubes += row * s
        sizes += row * s.sum()
        e4 += 3 * alpha**2 * s.sum()**2 + 6 * alpha * (s @ s)

        b = B.diagonal()
        tr = b.sum()
        gg = g @ g
        quadratic = alpha * (gg - tr) / (2 * r2)
        cubic = alpha * ((g**3).sum() / 3 - g @ b + 2 * cubes.sum() / 3)
        g4 = 3 * alpha**2 * gg**2 + 6 * alpha * (g**4).sum()
        gj = 3 * alpha**2 * (g @ sizes) + 6 * alpha * (g @ cubes)
        q2 = alpha**2 * (tr**2 + 2 * (B * B).sum()) + 6 * alpha * (b @ b)
        g2q = alpha**2 * (gg * tr + 2 * (g @ B @ g))
        g2q += 6 * alpha * (g**2 @ b)
        fourth = g4 / 24 - g2q / 4 + q2 / 8 + gj / 3 - e4 / 4
        result[t] = 1 + quadratic + cubic / r3 + fourth / r4

    return result


def support(d, alpha, j):
    rng = np.random.default_rng(SEED + j)
    if alpha == 1 and j % 2 == 0:
        return _prior_particles(d, POWER, rng)

    k = d if j % 2 else d - 1
    u = qmc.Sobol(k, scramble=True, rng=rng).random_base2(POWER)
    if j % 2:
        v = -np.log1p(-u) if alpha == 1 else special.gammaincinv(alpha, u)
        return v / v.sum(1, keepdims=True)

    b = special.betaincinv(alpha, alpha * np.arange(d - 1, 0, -1), u)
    rem = np.column_stack((np.ones(len(u)), np.cumprod(1 - b, axis=1)))
    rem[:, :-1] *= b
    return rem


@njit
def mass_residual(y):
    t_max, n = y.shape
    e1, e2, e3, e4, rem = np.zeros((5, n))
    sums = np.zeros(t_max)
    for t in range(t_max):
        for j in range(n):
            v = y[t, j]
            rem[j] += v * (e4[j] + rem[j])
            e4[j] += v * e3[j]
            e3[j] += v * e2[j]
            e2[j] += v * e1[j]
            e1[j] += v
            sums[t] += rem[j]

    return sums


def reference(name, alpha):
    dest = CACHE / f'{name}_{alpha:g}_ref.npz'
    if dest.exists():
        return
    x = tape(name)
    a = x / x.mean(1, keepdims=True) - 1
    a -= a.mean(1, keepdims=True)
    base = control(a, alpha)
    runs = []
    for j in range(SCRAMBLES):
        checkpoint = CACHE / f'{name}_{alpha:g}_scramble_{SEED + j}.npy'
        if checkpoint.is_file():
            runs.append(np.load(checkpoint))
            continue

        pts = support(x.shape[1], alpha, j)
        sums = np.zeros(len(x))
        for start in range(0, len(pts), 512):
            sums += mass_residual(a @ pts[start:start + 512].T)

        result = sums / len(pts)
        temp = checkpoint.with_suffix('.part.npy')
        np.save(temp, result)
        temp.replace(checkpoint)
        runs.append(result)
        print(name, alpha, f'Sobol scramble {j + 1}/{SCRAMBLES}', flush=True)

    pooled = np.log(base + np.mean(runs, axis=0))
    logs = np.log(base + runs)
    temp = dest.with_suffix('.part.npz')
    np.savez(temp, pooled=pooled, logs=logs)
    temp.replace(dest)
    print(name, alpha, np.max(abs(logs - pooled)), flush=True)


if __name__ == '__main__':
    with threadpool_limits(1):
        for alpha in (1, 0.5):
            for dataset in sys.argv[1:] or ('stocks', *CLASSIC):
                reference(dataset, alpha)
