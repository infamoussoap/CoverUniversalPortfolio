"""Independent exact checks and optional regret and extreme-row diagnostics."""

import sys
from math import comb

import numpy as np
import pandas as pd
from flint import arb, ctx
from threadpoolctl import threadpool_limits

from evalues import positive_pair
from particle_universal.interval_cover import outward_float
from particle_universal.linear_product_tilt import linear_product_evidence
from particle_universal.market_reference import best_fixed
from particle_universal.tilted_interval_cover import TiltedCover
from study import CACHE, CLASSIC, ROOT, cache_path, tape


OUT = ROOT.parent / 'cover_experiments_final'


def ends(lo, hi):
    return outward_float(lo.lower(), False), outward_float(hi.upper(), True)


def exact_pair():
    x = tape('pair')
    ref = positive_pair(x)
    box = np.load(cache_path('pair', 1))['bounds'][:, :2]
    inside = (box[:, 0] <= ref[:, 0]) & (ref[:, 1] <= box[:, 1])
    assert inside.all()

    df = pd.DataFrame(dict(t=np.arange(1, len(x) + 1), exact_lo=ref[:, 0],
                           exact_hi=ref[:, 1], z_lo=box[:, 0], z_hi=box[:, 1],
                           exact_inside=inside))
    df.to_csv(OUT / 'aapl_msft_exact.csv', index=False)


def regret():
    rows = []
    for name in CLASSIC:
        x = tape(name)
        T, d = x.shape
        coef, _ = best_fixed(x)
        w = [arb(float(v)) for v in coef]
        w = [v / sum(w) for v in w]
        grad = [arb(0) for _ in w]
        val, played = arb(0), arb(0)
        first = np.load(CACHE / f'w_{name}_1_first_order_0.npz')['w']

        for row, weights in zip(x, first):
            raw = [arb(float(v)) for v in row]
            gain = sum(i * j for i, j in zip(w, raw))
            val += gain.log()
            grad = [v + s / gain for v, s in zip(grad, raw)]
            wf = [arb(float(v)) for v in weights]
            played += (sum(i * j for i, j in zip(wf, raw)) / sum(wf)).log()

        # For any simplexed w, its dot product with this gradient equals T.
        gap = max(v.upper() for v in grad) - T
        lo, hi = val.lower(), (val + gap).upper()
        a, b, c, e = np.load(cache_path(name, 1))['bounds'][-1]
        cover_lo, cover_hi = arb(a) + arb(c), arb(b) + arb(e)
        row = dict(dataset=name, d=d, T=T, fw_gap=ends(0 * gap, gap)[1])
        row['opt_lo'], row['opt_hi'] = ends(lo, hi)
        row['cover_regret_lo'], row['cover_regret_hi'] = ends(lo - cover_hi,
                                                              hi - cover_lo)
        row['fo_regret_lo'], row['fo_regret_hi'] = ends(lo - played,
                                                        hi - played)
        row['cover_theory_upper'] = ends(lo, arb(comb(T + d - 1, d - 1)).log())[1]
        rows.append(row)

    pd.DataFrame(rows).to_csv(OUT / 'certified_regret.csv', index=False)


def product(name):
    x = tape(name)
    day = np.argmin(x.min(1) / x.mean(1))
    d = x.shape[1]
    weights = np.ones(d)
    mon = TiltedCover(d, exact_pairs=True)

    for t, row in enumerate(x):
        if t != day:
            mon.update(row, weights)

    lo, hi = linear_product_evidence(mon, x[day])
    mon.update(x[day], weights)
    base_lo, base_hi = mon.evidence()
    assert base_lo <= hi and lo <= base_hi
    ref = CACHE / f'{name}_1_ref.npz'
    if ref.exists():
        v = np.load(ref)['pooled'][-1]
        assert lo <= v <= hi

    a, b = ends(lo, hi)
    c, e = ends(base_lo, base_hi)
    return dict(dataset=name, alpha=1, exact_t=day + 1, S_size=1,
                min_m=x[day].min() / x[day].mean(), base_lo=c, base_hi=e,
                exactrow_lo=a, exactrow_hi=b,
                base_width=ends(base_lo, base_hi - base_lo)[1],
                exactrow_width=ends(lo, hi - lo)[1])


if __name__ == '__main__':
    ctx.prec, ctx.threads = 128, 1
    with threadpool_limits(1):
        if sys.argv[1] == 'checks':
            exact_pair()
            regret()
        else:
            names = list(CLASSIC) + ['finol_nyse', 'finol_sp']
            for name in names:
                p = CACHE / f'worst_row_{name}.csv'
                if not p.exists():
                    pd.DataFrame([product(name)]).to_csv(p, index=False)
            pd.concat([pd.read_csv(CACHE / f'worst_row_{n}.csv',
                                   float_precision='round_trip') for n in names]
                      ).to_csv(OUT / 'exact_row_width.csv', index=False)
