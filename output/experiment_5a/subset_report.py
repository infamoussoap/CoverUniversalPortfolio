"""Produce Table G and the two subset figures from complete replay records."""

import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from market import CACHE, ROOT


DIMENSIONS = (2, 3, 5, 10, 20, 50)
TEAL, ORANGE = '#00776e', '#b05221'


def assemble():
    shards = sorted((CACHE / 'subsets').glob('chunk_*.csv'))
    data = pd.concat([pd.read_csv(p, float_precision='round_trip')
                      for p in shards], ignore_index=True)
    design = pd.read_csv(ROOT / '5a_subset_design.csv')
    keys = ['d', 'subset_id']
    data.sort_values(keys, inplace=True, ignore_index=True)
    pd.testing.assert_frame_equal(data[design.columns], design)

    cap = pd.read_csv(CACHE / 'subset_audit.csv', float_precision='round_trip')
    data = data.merge(cap, on=keys, validate='one_to_one')
    assert len(data) == len(design)
    assert (data.z_lo_T <= data.z_hi_T).all()
    assert (data.max_e_lo >= 20).equals(data.significant)
    pair = data[data.d == 2]
    assert len(pair) == 1225
    assert ((pair.z_lo_T <= pair.exact_lo)
            & (pair.exact_hi <= pair.z_hi_T)).all()

    data['midpoint'] = (data.z_lo_T + data.z_hi_T) / 2
    data['verdict'] = np.select([data.z_lo_T > 0, data.z_hi_T < 0],
                                ['ahead', 'behind'], default='undecided')
    data['max_day'] = data.max_day.astype(int)
    data.to_csv(ROOT / '5a_subsets.csv', index=False)
    return data


def summarize(data):
    rows = []
    groups = {'all': data, 'with top five': data[data.has_top5],
              'without top five': data[~data.has_top5]}

    for label, group in groups.items():
        for d in DIMENSIONS:
            frame = group[group.d == d]
            row = dict(group=label, d=d, n=len(frame))
            for verdict in ('ahead', 'behind', 'undecided'):
                yes = frame.verdict == verdict
                row['n_' + verdict] = yes.sum()
                row['pct_' + verdict] = 100 * yes.mean()

            q = frame.midpoint.quantile([.25, .5, .75]).to_numpy()
            row.update(q25=q[0], median=q[1], q75=q[2],
                       n_significant=frame.significant.sum(),
                       pct_significant=100 * frame.significant.mean(),
                       median_VR=frame.VR_T.median(),
                       max_e_lo=frame.max_e_lo.max())
            rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(ROOT / '5a_subset_summary.csv', index=False)
    return summary


def scatter_by_size(data):
    fig, ax = plt.subplots(figsize=(6.3, 3.5), layout='constrained')
    ax.axhline(0, lw=.8, color='0.35')
    rng = np.random.default_rng(583215)

    for index, d in enumerate(DIMENSIONS):
        group = data[data.d == d]
        jitter = rng.uniform(-.24, .24, len(group)) if len(group) > 1 else 0
        x = np.broadcast_to(index + jitter, (len(group),))
        for mask, color in ((group.VR_T <= 1, TEAL), (group.VR_T > 1, ORANGE)):
            ax.scatter(x[mask], group.midpoint[mask], s=8 if d < 50 else 28,
                       color=color, alpha=.45 if d < 50 else 1, linewidths=0)

        ax.plot([index - .28, index + .28], [group.midpoint.median()] * 2,
                color='black', lw=1.2)

    handles = [Line2D([], [], color=c, marker='o', ls='none', ms=5)
               for c in (TEAL, ORANGE)] + [Line2D([], [], color='k', lw=1.2)]
    ax.legend(handles, [r'$VR_T\leq1$', r'$VR_T>1$', 'Median'], frameon=False)
    ax.set_xticks(range(len(DIMENSIONS)), labels=DIMENSIONS)
    ax.set(xlabel='Stocks in subset',
           ylabel='Terminal log wealth ratio (midpoint)')
    fig.savefig(ROOT / 'figure_C.pdf')
    plt.close(fig)


def scatter_by_variance(data):
    fig, axes = plt.subplots(2, 3, figsize=(8.2, 5.2))
    fig.subplots_adjust(left=.10, right=.98, top=.93, bottom=.18,
                        hspace=.53, wspace=.38)

    for ax, d in zip(axes.flat, DIMENSIONS):
        group = data[data.d == d].sort_values('VR_T')
        x = group.VR_T - 1
        ax.axhline(0, lw=.7, color='0.45')
        ax.axvline(0, lw=.6, color='0.6', ls=':')
        ax.scatter(x, group.pred_logZ, color=ORANGE, marker='x', s=10,
                   linewidths=.65, alpha=.6)
        ax.scatter(x, group.midpoint, color=TEAL, s=7, linewidths=0,
                   alpha=.65)
        ax.set_title(f'{d} stocks', fontsize=10)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))

        if d == 50:
            ax.set_xlim(-.4, .08)
            ax.set_ylim(-.006, .001)

    fig.supxlabel(r'$VR_T-1$', fontsize=10, y=.085)
    fig.supylabel('Terminal log wealth ratio', fontsize=10, x=.015)
    handles = [Line2D([], [], color=TEAL, marker='o', ls='none', ms=4),
               Line2D([], [], color=ORANGE, marker='x', ls='none', ms=5)]
    labels = ['Certified interval midpoint', 'Second-order prediction']
    fig.legend(handles, labels, frameon=False, loc='lower center', ncol=2)
    fig.savefig(ROOT / 'figure_D.pdf')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--figures', action='store_true')
    mode.add_argument('--summaries', action='store_true')
    options = parser.parse_args()

    if options.figures:
        frame = pd.read_csv(ROOT / '5a_subsets.csv',
                            float_precision='round_trip')
    else:
        frame = assemble()
        summarize(frame)

    if not options.summaries:
        plt.rcParams.update({'text.usetex': True, 'font.size': 10,
                             'axes.spines.top': False,
                             'axes.spines.right': False})
        scatter_by_size(frame)
        scatter_by_variance(frame)
