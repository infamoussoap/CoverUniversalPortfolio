"""Verify the MATLAB replay and assemble the market and source-audit tables."""

from decimal import Decimal

import numpy as np
import pandas as pd
from flint import arb, ctx
from scipy.io import loadmat

from bench_report import audit_row, bounds, one_gap
from study import CACHE, CLASSIC, ROOT, cache_path, tape


OUT = ROOT.parent / 'cover_experiments_final'
UP = 'OLPS-UP-MATLAB'
ICML = 'https://icml.cc/2012/papers/168.pdf'
PENG = 'https://doi.org/10.1155/2020/5956146'
JMLR = 'https://jmlr.org/papers/volume17/15-317/15-317.pdf'
SURVEY = 'https://arxiv.org/pdf/1212.2129'
NAMES = dict(zip(CLASSIC, ['DJIA', 'MSCI', 'SP500', 'NYSE(O)', 'NYSE(N)',
                           'TSE']))


def read(name, root=ROOT):
    return pd.read_csv(root / name, float_precision='round_trip')


def matlab_replay(name):
    x = tape(name)
    record = [loadmat(CACHE / f'{name}_matlab_{j}.mat', squeeze_me=True)
              for j in range(1, 4)]
    w = record[0]['w']
    assert np.array_equal(w, record[1]['w'])
    assert np.array_equal(w, record[2]['w'])
    assert w.shape == x.shape and np.allclose(w[0], 1 / x.shape[1])
    native = record[0]['terminal']
    daily = np.einsum('td,td->t', x, w)
    assert np.max(abs(daily - record[0]['gains'])) < 1e-12
    assert abs(np.log(native) - np.log(daily).sum()) < 1e-10

    raw = [[arb(float(v)) for v in row] for row in x]
    norm = [sum(row) / len(row) for row in raw]
    box = np.load(cache_path(name, 1))['bounds']
    ref = np.load(CACHE / f'{name}_1_ref.npz')
    spread = abs(ref['logs'] - ref['pooled']).max(axis=0)
    seconds = np.array([r['seconds'] for r in record])
    path = CACHE / f'w_{name}_1_{UP}_724191.npz'
    np.savez(path, w=w, times=seconds)

    frame, summary, timing = one_gap(name, 1, path, raw, norm, box,
                                     ref['pooled'], spread, x)
    assert abs(native - summary['term_wealth']) <= 1e-10 * native
    extra = dict(dataset=name, wealth=native, t1_s=seconds[0],
                 t2_s=seconds[1], t3_s=seconds[2], median_s=np.median(seconds),
                 binary64_max_daily_diff=abs(daily - record[0]['gains']).max(),
                 arb_reconstructed_wealth=summary['term_wealth'],
                 runtime=record[0]['runtime'])
    return frame, summary, timing, extra


def assemble_benchmarks():
    stats = read('bench_summary.csv')
    timing = read('bench_alloc_time.csv')
    stats = stats[~stats.strategy.isin(('OLPS-UP', UP))]
    timing = timing[~timing.strategy.isin(('OLPS-UP', UP))].copy()
    for i, row in timing[timing.strategy == 'VB-FTRL'].iterrows():
        times = np.load(CACHE / f'w_{row.dataset}_1_VB-FTRL_0.npz')['times']
        timing.loc[i, ['seconds', 'repeats']] = [np.median(times), len(times)]

    gaps = read('bench_gaps.csv')
    gaps = gaps[~gaps.strategy.isin(('OLPS-UP', UP))]
    outputs = [matlab_replay(n) for n in CLASSIC]
    new_gaps, summaries, times, verify = zip(*outputs)
    stats = pd.concat([stats, pd.DataFrame(summaries)], ignore_index=True)
    timing = pd.concat([timing, pd.DataFrame(times)], ignore_index=True)
    gaps = pd.concat([gaps, *new_gaps], ignore_index=True)
    for frame, name in [(stats, 'bench_summary'), (timing, 'bench_alloc_time'),
                        (gaps, 'bench_gaps')]:
        frame.sort_values(['dataset', 'alpha', 'strategy', 'seed'],
                          kind='stable').to_csv(OUT / (name + '.csv'),
                                                index=False)

    pd.DataFrame(verify).to_csv(OUT / 'matlab_replay.csv', index=False)
    assert len(stats) == 306 and len(gaps) == int(
        (stats.n_ahead + stats.n_behind + stats.n_undecided).sum())
    assert stats.loc[stats.E_resolved, 'B_over_E'].min() >= 1
    return stats


def half_prior_comparisons(weight_cache=None):
    """Compare three prior-independent strategy tapes to half-prior Cover."""
    if weight_cache is None:
        weight_cache = CACHE

    alpha = 0.5
    strategies = (('Soft-Bayes', 0), ('VB-FTRL', 0), (UP, 724191))
    daily_rows, summaries = [], []

    for name in CLASSIC:
        x = tape(name)
        raw = [[arb(float(v)) for v in row] for row in x]
        norm = [sum(row) / len(row) for row in raw]
        box = np.load(cache_path(name, alpha))['bounds']
        ref = np.load(CACHE / f'{name}_{alpha:g}_ref.npz')
        r = ref['pooled']
        spread = np.max(abs(ref['logs'] - r), axis=0)

        for strategy, seed in strategies:
            f = weight_cache / f'w_{name}_1_{strategy}_{seed}.npz'
            path, summary, _ = one_gap(name, alpha, f, raw, norm, box, r,
                                       spread, x, tape_alpha=1)
            path['v_lo'] = np.nextafter(box[:, 1] - path.delta_hi, -np.inf)
            path['v_hi'] = np.nextafter(box[:, 0] - path.delta_lo, np.inf)
            key = f'g_w_{name}_{alpha:g}_{strategy}_{seed}.npz'
            played = np.load(CACHE / key)['v']
            assert played.shape == (len(x),)
            assert ((path.v_lo <= played) & (played <= path.v_hi)).all()
            daily_rows.append(path)
            summaries.append(summary)

    path = OUT / 'bench_half_prior_gaps.csv'
    pd.concat(daily_rows).to_csv(path, index=False)
    pd.DataFrame(summaries).to_csv(OUT / 'bench_half_prior_summary.csv',
                                   index=False)


def published_audit(stats):
    original = read('up_audit.csv')
    audits = original[original.method != 'OLPS-UP'].to_dict('records')
    for row in audits:
        row['reference_alpha'] = (1 if row['method'] == 'published UP'
                                  else float(row['prior']))
        row['reported_in'] = ('Table A2, zero cost'
                              if row['method'] == 'published UP'
                              else 'local fixed-sample replay')
    for name in CLASSIC:
        base = original[(original.dataset == name)
                        & (original.method == 'published UP')].iloc[0]
        lo, hi = base.cert_lo, base.cert_hi
        u = stats[(stats.dataset == name) & (stats.strategy == UP)].iloc[0]
        common = dict(reference_alpha=1)
        audits.append(audit_row(name, 'OLPS toolbox, local MATLAB replay',
                                u.term_wealth, lo, hi, 'uniform grid target',
                                UP, reported_in='local MATLAB replay', **common))
        amounts = [(PENG, 1.62 if name == 'sp500' else base.reported_wealth,
                    'Table 3', 'KV random walk, M=100, S=500')]
        if name in ('djia', 'nyse-n', 'nyse-o', 'tse'):
            amounts.append((ICML, base.reported_wealth, 'Table 1(a)',
                            'published UP'))

        for url, val, where, method in amounts:
            audits.append(audit_row(name, url, val, lo, hi, 'unspecified',
                                    method, reported_in=where,
                                    rounding_resolution=.01, **common))
        for url in (JMLR, SURVEY):
            audits.append(dict(dataset=name, source=url, method='source check',
                               reported_in='No six-market UP wealth table',
                               cert_lo=lo, cert_hi=hi, reference_alpha=1))

    f = pd.DataFrame(audits)
    f['rounded_compatible'] = pd.array([pd.NA] * len(f), dtype='boolean')
    for i, row in f[f.rounding_resolution.notna()].iterrows():
        half = Decimal(str(row.rounding_resolution)) / 2
        reported = Decimal(str(row.reported_wealth))
        fit = (reported + half >= Decimal.from_float(row.cert_lo)
               and reported - half <= Decimal.from_float(row.cert_hi))
        f.loc[i, 'rounded_compatible'] = fit
    f.to_csv(OUT / 'up_audit.csv', index=False)
    return f


if __name__ == '__main__':
    ctx.prec, ctx.threads = 128, 1
    audit = published_audit(assemble_benchmarks())
