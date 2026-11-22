"""Tabulate normalized evidence, sign decisions, and direct wealth audits."""

import argparse

import numpy as np
import pandas as pd
from flint import arb, ctx
from threadpoolctl import threadpool_limits

from particle_universal.interval_cover import outward_float, played_log_gain
from study import CACHE, CLASSIC, ROOT, cache_path, tape


TOLS = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
SOURCE = 'https://www.mdpi.com/2227-7390/10/7/1073'
PUBLISHED = dict(zip(CLASSIC, [0.81, 0.92, 1.65, 26.68, 31.49, 1.60]))


def bounds(value):
    return outward_float(value.lower(), False), outward_float(value.upper(),
                                                             True)


def audit_row(name, source, value, lo, hi, prior, method, **kwargs):
    pos = 'below' if value < lo else 'above' if value > hi else 'inside'
    return dict(dataset=name, source=source, reported_wealth=value,
                cost_setting=0, prior=prior, method=method, cert_lo=lo,
                cert_hi=hi, position=pos, rel_diff=value / ((lo+hi)/2) - 1,
                **kwargs)


def one_gap(name, alpha, f, raw, norm, box, ref, spread, x, tape_alpha=None):
    with np.load(f) as data:
        w = data['w']
        secs = data['times']

    source_alpha = alpha if tape_alpha is None else tape_alpha
    prefix = f'w_{name}_{source_alpha:g}_'
    if not f.stem.startswith(prefix):
        raise ValueError(f'Unexpected strategy tape {f.name}')
    strategy, seed = f.stem[len(prefix):].rsplit('_', 1)
    cache = CACHE / f'g_w_{name}_{alpha:g}_{strategy}_{seed}.npz'
    if cache.exists():
        with np.load(cache) as data:
            gaps, v, wealth = (data[k] for k in ('gaps', 'v', 'wealth'))
    else:
        g = arb(0)
        gaps = np.empty((len(x), 2))
        v = np.empty(len(x))
        for i, (row, weights) in enumerate(zip(raw, w)):
            g += played_log_gain(row, weights, norm[i])
            a, b = box[i, :2]
            gaps[i] = (outward_float((arb(float(a)) - g).lower(), False),
                       outward_float((arb(float(b)) - g).upper(), True))
            v[i] = float(g.mid())
        wealth = float((sum(i.log() for i in norm) + g).exp().mid())
        temp = cache.with_suffix('.part.npz')
        np.savez(temp, gaps=gaps, v=v, wealth=wealth)
        temp.replace(cache)

    low, high = gaps.T
    B = max(-low.min(), high.max())
    E = np.max(abs(ref-v))
    resolved = E >= 3 * spread.max()
    ahead, behind = high < 0, low > 0
    verdict = 'ahead' if ahead[-1] else 'behind' if behind[-1] else 'undecided'
    key = dict(dataset=name, alpha=alpha, strategy=strategy, seed=int(seed))
    lo_pct = bounds(100 * ((-arb(float(high[-1]))).exp() - 1))[0]
    hi_pct = bounds(100 * ((-arb(float(low[-1]))).exp() - 1))[1]
    fail = ''
    if strategy == 'first_order':
        a = x / x.mean(1, keepdims=True) - 1
        fail = (a.cumsum(0)[:-1].min(1) <= -(x.shape[1] * alpha + 1)).sum()
    summary = dict(**key, term_lo=low[-1], term_hi=high[-1], verdict=verdict,
                   pct_lo=lo_pct, pct_hi=hi_pct, B=B, E=E,
                   E_resolved=resolved, B_over_E=B/E if resolved else np.nan,
                   verified_tol=next((e for e in TOLS if B <= e), np.nan),
                   n_ahead=ahead.sum(), n_behind=behind.sum(),
                   n_undecided=(~ahead & ~behind).sum(), fallback=fail,
                   term_wealth=float(wealth))
    timing = dict(**key, seconds=np.median(secs), repeats=len(secs))
    daily = pd.DataFrame(dict(**key, t=np.arange(1, len(x)+1), delta_lo=low,
                              delta_hi=high))
    return daily, summary, timing


def report():
    ctx.prec, ctx.threads = 128, 1
    evidence, summaries, gap_frames, times, audit = [], [], [], [], []
    accuracy, cost = [], []
    meta = pd.read_csv(ROOT / 'bench_datasets.csv').set_index('dataset')
    for name in CLASSIC:
        x = tape(name)
        raw = [[arb(float(v)) for v in row] for row in x]
        norm = [sum(row) / len(row) for row in raw]
        for alpha in (1, 0.5):
            path = cache_path(name, alpha)
            if not path.exists():
                continue
            data = np.load(path)
            box, degree = data['bounds'], data['degrees']
            refdata = np.load(CACHE / f'{name}_{alpha:g}_ref.npz')
            r, logs = refdata['pooled'], refdata['logs']
            spread = np.max(abs(logs-r), axis=0)
            lo, hi = box[:, :2].T
            assert np.isfinite(box).all() and np.all(lo <= hi)
            lower = bounds((arb(box[-1, 0]) + arb(box[-1, 2])).exp())[0]
            upper = bounds((arb(box[-1, 1]) + arb(box[-1, 3])).exp())[1]
            if alpha == 1:
                meta.loc[name, ['N_min', 'N_max']] = degree.min(), degree.max()
                audit.append(audit_row(name, SOURCE, PUBLISHED[name], lower,
                                       upper, 'unspecified', 'published UP',
                                       rounding_resolution=0.01))
            tt = np.arange(1, len(x)+1)
            evidence.append(pd.DataFrame(dict(dataset=name, alpha=alpha,
                                              t=tt, z_lo=lo, z_hi=hi,
                                              ref_logZ=r, ref_spread=spread)))
            sec = np.median(data['times'])
            width = np.nextafter(hi-lo, np.inf)
            accuracy.append(dict(dataset=name, alpha=alpha,
                                 term_width=width[-1], max_width=width.max(),
                                 inside=((lo <= r) & (r <= hi)).sum(),
                                 T=len(x), s_max=spread.max(), seconds=sec))
            cost.append(dict(dataset=name, alpha=alpha, d=x.shape[1],
                             T=len(x), method='pair', seconds=sec,
                             repeats=len(data['times']),
                             peak_mem_mb=float(data['rss']),
                             term_width=width[-1], N_max=degree.max()))
            window = []
            for f in sorted(CACHE.glob(f'w_{name}_{alpha:g}_*.npz')):
                if f.name.endswith('.part.npz'):
                    continue
                daily, row, timing = one_gap(name, alpha, f, raw, norm, box,
                                             r, spread, x)
                summaries.append(row)
                gap_frames.append(daily)
                times.append(timing)
                window.append(row)
            for kind, df in pd.DataFrame(window).groupby('strategy'):
                if not (kind.startswith('sample') or kind == 'OLPS-UP'):
                    continue
                vals = df.term_wealth
                audit.append(audit_row(name, 'local replay', vals.median(),
                                       lower, upper, alpha, kind,
                                       seed_min=vals.min(), seed_max=vals.max(),
                                       n_seeds=len(vals)))
        print(name, flush=True)
    meta.to_csv(ROOT / 'bench_datasets.csv')
    pd.concat(evidence).to_csv(ROOT / 'bench_evidence.csv', index=False)
    pd.concat(gap_frames).to_csv(ROOT / 'bench_gaps.csv', index=False)
    stats = pd.DataFrame(summaries)
    stats.to_csv(ROOT / 'bench_summary.csv', index=False)
    pd.DataFrame(times).to_csv(ROOT / 'bench_alloc_time.csv', index=False)
    pd.DataFrame(audit).to_csv(ROOT / 'up_audit.csv', index=False)
    (pd.DataFrame(accuracy).merge(stats[stats.strategy == 'first_order'],
                                 on=['dataset', 'alpha'])
     .to_csv(ROOT / 'tableB.csv', index=False))
    pd.DataFrame(cost).to_csv(CACHE / 'classic_cost.csv', index=False)


if __name__ == '__main__':
    with threadpool_limits(1):
        report()
