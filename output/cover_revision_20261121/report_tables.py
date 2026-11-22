"""Build human-readable benchmark matrices and figures from verified exports."""

from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter

from assemble_market import NAMES, OUT, PENG, UP, read
from study import CACHE, CLASSIC, ROOT, cache_path


def directed(value, up, digits):
    number = Decimal.from_float(float(value))
    quantum = Decimal(1).scaleb(-digits)
    rounding = ROUND_CEILING if up else ROUND_FLOOR
    return f'{number.quantize(quantum, rounding=rounding):f}'


def interval(lo, hi, digits):
    return f'[{directed(lo, False, digits)}, {directed(hi, True, digits)}]'


def table_c(stats):
    cells = []
    for (name, strategy), df in stats[stats.alpha == 1].groupby(
            ['dataset', 'strategy']):
        lo, hi = df[['term_lo', 'term_hi']].median().to_numpy()
        small = max(abs(lo), abs(hi)) < .001
        if small:
            display = interval(np.nextafter(lo * 1e5, -np.inf),
                               np.nextafter(hi * 1e5, np.inf), 3)
            unit = 'log gap in 1e-5'
        else:
            display = interval(df.pct_lo.median(), df.pct_hi.median(), 3)
            unit = 'percent strategy relative to Cover'

        verdict = 'ahead' if hi < 0 else 'behind' if lo > 0 else 'undecided'
        row = dict(dataset=name, strategy=strategy, n_seeds=len(df),
                   term_lo=lo, term_hi=hi, display_interval=display,
                   display_unit=unit, verdict=verdict)
        for col in ('term_lo', 'term_hi', 'pct_lo', 'pct_hi', 'n_ahead',
                    'n_behind', 'n_undecided', 'verified_tol'):
            vals = df[col].fillna(np.inf)
            for label, val in zip(('min', 'median', 'max'),
                                  (vals.min(), vals.median(), vals.max())):
                row[col + '_' + label] = val if np.isfinite(val) else np.nan
        row['n_uncertified_tolerance'] = df.verified_tol.isna().sum()
        row['n_terminal_ahead'] = (df.verdict == 'ahead').sum()
        row['n_terminal_behind'] = (df.verdict == 'behind').sum()
        cells.append(row)

    table = pd.DataFrame(cells)
    table.to_csv(OUT / 'tableC_seed_ranges.csv', index=False)
    wide = []
    order = ('nyse-o', 'nyse-n', 'djia', 'sp500', 'msci', 'tse')
    for strategy, block in table.groupby('strategy'):
        row = dict(strategy=strategy)
        for dataset in order:
            r = block[block.dataset == dataset].iloc[0]
            lo = np.nextafter(r.term_lo_min * 1e4, -np.inf)
            hi = np.nextafter(r.term_hi_max * 1e4, np.inf)
            mark = ' *' if r.n_terminal_ahead and r.n_terminal_behind else ''
            row[NAMES[dataset]] = interval(lo, hi, 2) + mark
        wide.append(row)
    pd.DataFrame(wide).to_csv(OUT / 'tableC.csv', index=False)


def table_d(audit):
    rows = []
    for name, df in audit.groupby('dataset'):
        matlab = df[df.method == UP].iloc[0]
        row = dict(dataset=name, cert_wealth_lo=matlab.cert_lo,
                   cert_wealth_hi=matlab.cert_hi, MATLAB_wealth=matlab.reported_wealth,
                   MATLAB_position=matlab.position, MATLAB_rel_diff=matlab.rel_diff)
        for r in df.itertuples():
            if r.source == PENG:
                tag = 'Peng_table3'
            elif 'mdpi.com' in r.source:
                tag = 'Moon_Yoon_tableA2'
            elif 'icml.cc' in r.source:
                tag = 'LiHoi_ICML'
            elif r.method.startswith('sample_') and r.reference_alpha == 1:
                tag = r.method
                row[tag + '_seed_min'] = r.seed_min
                row[tag + '_seed_max'] = r.seed_max
            else:
                continue
            row[tag] = r.reported_wealth
            row[tag + '_position'] = r.position
            row[tag + '_rel_diff'] = r.rel_diff
        rows.append(row)

    pd.DataFrame(rows).to_csv(OUT / 'tableD.csv', index=False)


def wealth_plot():
    fig, axes = plt.subplots(2, 1, figsize=(6.6, 4.6), layout='constrained')
    for ax, name in zip(axes, ('nyse-n', 'djia')):
        ref = np.load(CACHE / f'{name}_1_ref.npz')['pooled']
        box = np.load(cache_path(name, 1))['bounds']
        t = np.arange(1, len(ref) + 1)
        band = 100 * np.exp(box[:, :2] - ref[:, None])
        ax.fill_between(t, *band.T, color='0.8', label='Certified Cover')
        for strategy, color, style in [
                ('first_order', 'k', '-'), ('UCRP', '#166b79', '-'),
                ('sample_1000', '#b1582d', '-'), ('VB-FTRL', '#713e79', ':')]:
            seed = 74081 if strategy == 'sample_1000' else 0
            key = f'g_w_{name}_1_{strategy}_{seed}.npz'
            v = np.load(CACHE / key)['v']
            label = {'first_order': 'First order', 'sample_1000': 'Sampled UP'}
            ax.plot(t, 100 * np.exp(v - ref), style, color=color, lw=.8,
                    label=label.get(strategy, strategy))
        ax.set_title(NAMES[name], fontsize=10)
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    axes[-1].set_xlabel('Trading day')
    fig.supylabel(r'Wealth per \$100 of estimated Cover')
    axes[0].legend(ncol=3, frameon=False, fontsize=8)
    fig.savefig(OUT / 'bench_wealth.pdf')
    plt.close(fig)


if __name__ == '__main__':
    plt.rcParams['text.usetex'] = True
    table_c(read('bench_summary.csv', OUT))
    table_d(read('up_audit.csv', OUT))
    wealth_plot()
