"""Evaluate the frozen Yahoo subsets with the pair-series certificate."""

import argparse
from functools import lru_cache

import numpy as np
import pandas as pd
from flint import arb, arb_mat, arb_poly, ctx
from numba import njit
from scipy.special import roots_legendre

from market import CACHE, LEVEL, ROOT, load_panel
from particle_universal.interval_cover import outward_float
from particle_universal.tilted_interval_cover import TiltedCover
from screen import screen


@njit(cache=True)
def pair_quadrature(raw, nodes, weights):
    mass = np.ones(len(nodes))

    for x, y in raw:
        slope = (x - y) / (x + y)
        for j in range(len(nodes)):
            mass[j] *= 1 + slope * nodes[j]

    return np.log(np.dot(weights, mass) / 2)


@lru_cache(1)
def quadrature_rule():
    return roots_legendre(3300)


def oracle(slopes, score, squares):
    """Enclose the two-asset integral using a truncated polynomial product."""
    degree = 96
    tail_scale = (2 * abs(score) + 2 * squares).exp().upper()

    while tail_scale / arb(2)**degree > arb(2)**-80:
        degree += 32

    level = [arb_poly([1, slope]) for slope in slopes]
    while len(level) > 1:
        products = [(p * q).truncate(degree + 1)
                    for p, q in zip(level[::2], level[1::2])]
        if len(level) % 2:
            products.append(level[-1])
        level = products

    p = level[0]
    mass = sum(p[k] / (k + 1) for k in range(0, len(p), 2))
    return (mass + arb(0, tail_scale / arb(2)**degree)).log()


def pair_checkpoints(raw, days):
    mon = TiltedCover(2, exact_pairs=True)
    score, q, k, c, h, cube = [arb(0) for _ in range(6)]
    pattern = arb_mat([[1, -1], [-1, 1]])
    skew = arb_mat([[1, -1], [1, -1]])
    slopes, boxes = [], {}

    for t, (s, u) in enumerate(raw, 1):
        x, y = arb(float(s)), arb(float(u))
        a = (x - y) / (x + y)
        b, sq = abs(a), a * a
        minimum = 1 - b
        slopes.append(a)

        score += a
        q += sq / 2
        k += sq * sq / (4 * minimum)
        c += sq * b
        cube += a * sq
        h += -b - (-b).log1p()

        if t in days:
            mon.g = [score, -score]
            mon.Q, mon.K = pattern * q, pattern * k
            mon.C, mon.H = pattern * c, pattern * h
            mon.J_matrix = skew * cube
            mon.J_diag = [cube, -cube]
            mon.J_weighted = [2 * cube, -2 * cube]
            mon.R4, mon.H4_root = k / 5, h / arb(3).sqrt()
            boxes[t] = mon.evidence()

    return boxes, oracle(slopes, score, 2 * q)


def general_checkpoints(raw, days):
    d = raw.shape[1]
    mon = TiltedCover(d, exact_pairs=True)
    weights = np.ones(d)
    boxes = {}

    for t, row in enumerate(raw, 1):
        mon.update(row, weights)
        if t in days:
            boxes[t] = mon.evidence()

    return boxes


def evaluate(raw):
    T, d = raw.shape
    a = raw / raw.mean(axis=1, keepdims=True) - 1
    g = a.sum(axis=0)
    ss = np.square(a).sum()
    vr = (g @ g) / ss
    momentum = ((g @ g) - ss) / 2

    out = dict(VR_T=vr, M_T=momentum, pred_logZ=momentum / (d * (d + 1)))
    if d == 50:
        frame = pd.read_csv(ROOT / '5a_summary.csv',
                            float_precision='round_trip')
        row = frame[frame.market == 'Yahoo 50'].iloc[0]
        out.update(z_lo_T=row.z_lo, z_hi_T=row.z_hi, max_e_lo=row.max_e_lo,
                   max_day=row.max_day, significant=pd.notna(row.reject_day),
                   arb_closes=T, screen_diff=np.nan)
        return out

    scores = screen(raw, True)
    if not np.isfinite(scores).all():
        raise ArithmeticError('Nonfinite daily screen')

    days = set(np.flatnonzero((scores >= scores.max() - 1e-9)
                              | (scores >= LEVEL - .05)) + 1)
    days.add(T)
    if d == 2:
        boxes, ref = pair_checkpoints(raw, days)
        val = pair_quadrature(raw, *quadrature_rule())
        lo, hi = boxes[T]
        if not (lo <= ref.lower() and ref.upper() <= hi and lo < val < hi):
            raise ArithmeticError('Terminal pair enclosure failed')

        out.update(exact_logZ_T=val,
                   exact_lo=outward_float(ref.lower(), False),
                   exact_hi=outward_float(ref.upper(), True),
                   gl_oracle_diff=abs(val - float(ref.mid())))
    else:
        boxes = general_checkpoints(raw, days)

    lo, hi = boxes[T]
    best = max(boxes, key=lambda t: float(boxes[t][0]))
    diff = max(abs(float(pair[0]) - scores[t - 1])
               for t, pair in boxes.items())
    if diff >= 1e-9:
        raise ArithmeticError(f'Binary64 screen discrepancy {diff:g}')

    peak = boxes[best][0].exp().lower()
    out.update(z_lo_T=outward_float(lo, False), z_hi_T=outward_float(hi, True),
               max_e_lo=outward_float(peak, False), max_day=best,
               significant=bool(peak >= arb(20)), arb_closes=len(boxes),
               screen_diff=diff)
    return out


def worker(shard, count):
    ctx.prec, ctx.threads = 128, 1
    design = pd.read_csv(ROOT / '5a_subset_design.csv')
    panel = load_panel('full')
    tickers = list(panel['tickers'])
    chunks = [part for _, part in design.groupby(design.index // 8)]
    folder = CACHE / 'subsets'
    folder.mkdir(exist_ok=True)

    for j in range(shard, len(chunks), count):
        dest = folder / f'chunk_{j:03d}.csv'
        if dest.exists():
            continue

        results = []
        for _, item in chunks[j].iterrows():
            cols = [tickers.index(v) for v in item.tickers.split()]
            raw = np.ascontiguousarray(panel['returns'][:, cols])
            result = dict(item)
            result.update(evaluate(raw))
            results.append(result)

        temp = dest.with_suffix('.tmp')
        pd.DataFrame(results).to_csv(temp, index=False)
        temp.replace(dest)
        if j % 10 == 0:
            print('Completed subset chunk', j, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('shard', type=int)
    parser.add_argument('count', type=int)
    args = parser.parse_args()
    worker(args.shard, args.count)
