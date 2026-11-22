"""Draw the pair-series enclosure and wealth in Figure 1."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flint import arb, ctx
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from particle_universal.interval_cover import outward_float
from particle_universal.yahoo_daily_portfolios import load_panel
from study import ROOT


OUT = ROOT.parent / 'cover_experiments_final'


def draw():
    ctx.prec = 128
    frame = pd.read_csv(OUT / 'aapl_msft_exact.csv',
                        float_precision='round_trip')
    panel = load_panel('full')
    dates = panel['dates'].astype('datetime64[D]')
    cols = [panel['tickers'].tolist().index(i) for i in ('AAPL', 'MSFT')]
    returns = panel['returns'][:, cols]
    assert np.array_equal(frame.t, np.arange(1, len(dates) + 1))

    score = arb(0)
    normal = arb(0)
    wealth, lower, upper = np.empty((3, len(frame)))
    for i, ((x1, x2), r) in enumerate(zip(returns, frame.itertuples())):
        a, b = arb(float(x1)), arb(float(x2))
        score += (a - b) / (a + b)
        normal += ((a + b) / 2).log()
        exact = arb(r.exact_lo).union(arb(r.exact_hi))
        wealth[i] = float((100 * (normal + exact).exp()).mid())
        lo = 100 * ((arb(r.z_lo) - exact.upper()).exp() - 1)
        hi = 100 * ((arb(r.z_hi) - exact.lower()).exp() - 1)
        lower[i] = outward_float(lo.lower(), False)
        upper[i] = outward_float(hi.upper(), True)

    assert (lower <= 0).all() and (upper >= 0).all()
    ratio = exact.exp() / (score.sinh() / score)
    print('Terminal dollars, low, high, exact / MGF, g:', wealth[-1],
          lower[-1], upper[-1], ratio, score)
    print('Peak quadrature order difference:',
          (frame.gl_3270 - frame.gl_3300).abs().max())
    print('Percent ranges', lower.min(), upper.max())
    plt.rcParams['text.usetex'] = True
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'path.simplify': False})
    fig, (cash, band) = plt.subplots(2, 1, sharex=True, figsize=(6.4, 4.1))
    fig.subplots_adjust(left=.16, right=.97, bottom=.11, top=.97, hspace=.18)
    cash.plot(dates, wealth, color='black', lw=.8)
    cash.set_yscale('log')
    cash.yaxis.set_major_locator(FixedLocator([100, 1000, 10000]))
    cash.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
    cash.yaxis.set_minor_locator(NullLocator())
    cash.set_ylabel(r'Cover wealth (\$)')
    cash.tick_params(axis='x', bottom=True, labelbottom=False)

    band.fill_between(dates, lower, upper, color='.82', lw=0)
    band.axhline(0, color='black', ls='--', lw=.75)
    band.set_ylabel('Certified bounds\nrelative to Cover (\\%)')
    band.set_xlim(np.datetime64('1999-12-31'), dates[-1])
    band.xaxis.set_major_locator(mdates.YearLocator(5))
    band.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.savefig(OUT / 'aapl_msft_pair_series.pdf')
    plt.close(fig)
