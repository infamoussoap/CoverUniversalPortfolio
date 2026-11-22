"""Summarize three uninterrupted timing trials per dataset and prefix."""

import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, ScalarFormatter

from assemble_market import OUT, read
from study import CACHE, CLASSIC


def trials():
    records = []
    for p in sorted(CACHE.glob('cost_*.json')):
        row = json.loads(p.read_text())
        row.setdefault('alpha', 1.)
        row['raw_seconds'] = row['seconds']
        row['timing'] = 'continuous'
        records.append(row)

    keys = ['dataset', 'd', 'T', 'rep', 'method', 'alpha']
    frame = pd.DataFrame(records).set_index(keys, verify_integrity=True)
    return frame.reset_index()


def scaling_plot(cost):
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.0), layout='constrained')
    subset = cost[(cost.dataset == 'finol_nyse') & (cost.method == 'pair')]
    assert len(subset) == 20
    slopes = []

    for ax, var, group in zip(axes, ('d', 'T'), ('T', 'd')):
        for value, block in subset.groupby(group):
            block = block.sort_values(var)
            x, y = block[var], block.seconds
            power, offset = np.polyfit(np.log(x), np.log(y), 1)
            fitted = np.exp(offset) * x**power
            label = rf'${group}={value}$ ({power:.2f})'
            line, = ax.loglog(x, fitted, lw=.95, label=label)
            ax.loglog(x, y, 'o', ms=3.3, color=line.get_color())
            slopes.append(dict(x=var, fixed=group, value=value,
                               slope=power, log_intercept=offset))

        ax.xaxis.set_major_locator(FixedLocator(sorted(subset[var].unique())))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.tick_params(which='minor', labelbottom=False)
        ax.set_xlabel('Assets' if var == 'd' else 'Trading days')
        ax.set_ylabel('Wall time (seconds)')
        ax.legend(title='Fit (slope)', frameon=False, fontsize=8, ncol=2,
                  loc='lower left', bbox_to_anchor=(0, 1.02), borderaxespad=0)

    fig.savefig(OUT / 'bench_scaling.pdf')
    plt.close(fig)
    pd.DataFrame(slopes).to_csv(OUT / 'bench_cost_slopes.csv', index=False)


def save_costs():
    detail = trials()
    keys = ['dataset', 'alpha', 'd', 'T', 'method']
    groups = detail.groupby(keys)
    assert len(groups) == 35 and (groups.size() == 3).all()
    assert (groups.N_max.nunique() == 1).all()
    assert (groups.term_width.nunique() == 1).all()

    result = groups.agg(seconds=('seconds', 'median'),
                        peak_mem_mb=('peak_mem_mb', 'median'),
                        term_width=('term_width', 'first'),
                        N_max=('N_max', 'first'), repeats=('rep', 'count'),
                        min_s=('seconds', 'min'), max_s=('seconds', 'max'),
                        max_mem_mb=('peak_mem_mb', 'max')).reset_index()
    result.to_csv(OUT / 'bench_cost.csv', index=False)
    detail.to_csv(OUT / 'bench_cost_trials.csv', index=False)
    scaling_plot(result)

    if not (OUT / 'tableB.csv').exists():
        return

    acc = read('tableB.csv', OUT)
    lookup = result.set_index(['dataset', 'alpha'])
    for i, r in acc.iterrows():
        row = lookup.loc[[(r.dataset, r.alpha)]].iloc[0]
        assert abs(row.term_width - r.term_width) < 1e-12
        acc.loc[i, 'seconds'] = row.seconds

    acc.to_csv(OUT / 'tableB.csv', index=False)


if __name__ == '__main__':
    plt.rcParams['text.usetex'] = True
    save_costs()
