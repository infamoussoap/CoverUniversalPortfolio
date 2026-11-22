"""Recompute the width decomposition and the 50-stock companion figure."""

import numpy as np
import pandas as pd
from flint import arb, ctx
from scipy.special import roots_legendre
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from particle_universal.curvature_slack import slack_sample
from particle_universal.interval_cover import moment_bounds, outward_float
from particle_universal.market_reference import _prior_particles
from particle_universal.tilted_interval_cover import TiltedCover
from study import CACHE, ROOT, cache_path, load_panel, tape


def percentages(bounds, ref):
    return np.array([[outward_float((100 * ((arb(float(z)) - float(r)).exp()
                                            - 1)).upper(), True) if j else
                      outward_float((100 * ((arb(float(z)) - float(r)).exp()
                                            - 1)).lower(), False)
                      for j, z in enumerate(row)] for row, r in
                     zip(bounds, ref)])


def figure():
    panel = load_panel('full')
    dates = pd.to_datetime(panel['dates'])
    with np.load(CACHE / 'stocks_1_ref.npz') as data:
        r, logs = data['pooled'], data['logs']
    with np.load(cache_path('stocks', 1)) as data:
        box = data['bounds']

    lo, hi = box[:, :2].T
    mn, mx = logs.min(0), logs.max(0)
    raw = r + np.log(panel['returns'].mean(1)).cumsum()
    band = percentages(box[:, :2], r)
    spread = 100 * np.expm1(np.stack((mn - r, mx - r)))
    frame = pd.DataFrame(dict(date=dates, t=np.arange(1, len(r) + 1),
                              z_lo=lo, z_hi=hi, ref_logZ=r, ref_min=mn,
                              ref_max=mx, ref_dollars=100 * np.exp(raw),
                              lo_pct=band[:, 0], hi_pct=band[:, 1],
                              log_norm_lo=box[:, 2], log_norm_hi=box[:, 3]))
    frame.to_csv(ROOT / 'fig2_50stock.csv', index=False)
    plt.rcParams['text.usetex'] = True
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6.3, 4.25),
                             layout='constrained')
    ax, bx = axes
    ax.plot(dates, frame.ref_dollars, color='black', lw=0.9)
    ax.set_yscale('log')
    ax.set_yticks([100, 200, 500, 1000, 2000])
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_ylabel(r'Cover wealth (\$)')
    ax.tick_params(axis='x', labelbottom=True)
    bx.fill_between(dates, *band.T, color='0.80', label='Certified interval')
    bx.plot(dates, band[:, 0], color='0.35', lw=0.55)
    bx.plot(dates, band[:, 1], color='0.35', lw=0.55)
    bx.fill_between(dates, *spread, color='black', label='Scramble range')
    bx.plot(dates, spread[1], color='black', lw=0.45)
    bx.set_ylabel(r'Relative to estimate (\%)')
    bx.legend(frameon=False, fontsize=8, loc='upper left')
    bx.set_xlim(pd.Timestamp('1999-12-31'), pd.Timestamp('2025-12-31'))
    bx.xaxis.set_major_locator(mdates.YearLocator(5))
    bx.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.savefig(ROOT / 'fig2_50stock.pdf')
    plt.close(fig)
    print('stocks terminal dollars, percent, max width, coverage, spread',
          frame.ref_dollars.iloc[-1], band[-1], np.ptp(band, axis=1).max(),
          ((lo <= r) & (r <= hi)).sum(), np.max(abs(logs - r)), flush=True)


def pair_terms(centered, n):
    nodes, weights = roots_legendre(n)
    p = (nodes + 1) / 2
    terms = []
    for batch in np.array_split(p, max(1, n // 100)):
        y = centered[:, :1] * batch + centered[:, 1:] * (1 - batch)
        low = centered.min(axis=1)
        eta = np.full(len(low), 0.5)
        ix = low != 0
        eta[ix] = (low[ix] - np.log1p(low[ix])) / low[ix]**2
        q = y * y
        D = (y - np.log1p(y)).sum(0)
        D3 = (q * (0.5 - y / 3)).sum(0)
        major = eta @ q
        quart = (0.25 / (1 + low)) @ (q * q)
        terms.append(np.array([y.sum(0), D, D - D3,
                               (major - D) * (major + D) / 2,
                               D**2 / 2 - D - np.expm1(-D), quart]))

    g, D, term1, term2, term3, quart = np.hstack(terms)
    p = weights * np.exp(g)
    p /= p.sum()
    avg = p @ D
    return [np.exp(-avg) * (p @ np.expm1(avg - D)),
            -np.exp(-avg) * np.expm1(p @ (term1 - quart)),
            p @ term1, p @ term2, p @ term3]


def decompose():
    rows = []
    for name in ('pair', 'stocks'):
        x = tape(name)
        a = x / x.mean(1, keepdims=True) - 1
        if name == 'pair':
            trials = [pair_terms(a, n) for n in (256, 512)]
            mean = np.array(trials[-1])
        else:
            trials = [slack_sample(a, _prior_particles(x.shape[1], 13,
                      np.random.default_rng(j))) for j in range(91491, 91499)]
            mean = np.mean(trials, axis=0)

        w = np.ones(x.shape[1])
        monitor = TiltedCover(len(w), exact_pairs=True)
        for row in x:
            monitor.update(row, w)

        low, high = monitor.evidence()
        mgf = moment_bounds(monitor.g)[0]
        mass = (high.exp() - low.exp()) / mgf[0].union(mgf[1])
        spread = np.max(abs(trials - mean) / np.maximum(abs(mean), 1e-300))
        g = [float(v.mid()) for v in monitor.g]
        box = np.load(cache_path(name, 1))
        row = dict(zip(('jensen', 'remainder', 'upper1', 'upper2', 'upper3'),
                       1e6 * mean))
        width = outward_float((1e6 * mass).upper(), True)
        row.update(path=name, implemented_width=width,
                   max_rel_dev=spread, score_range=np.ptp(g),
                   pair_degree=box['degrees'][-1])
        rows.append(row)
    pd.DataFrame(rows).to_csv(ROOT / 'table1_width.csv', index=False)


if __name__ == '__main__':
    ctx.prec, ctx.threads = 128, 1
    with threadpool_limits(1):
        figure()
        decompose()
