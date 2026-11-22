"""Tabulate predeclared tests and draw the associated figures."""

import argparse
from math import lgamma

from flint import arb, ctx
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
from scipy.optimize import brentq, minimize
from scipy.stats import binomtest
from threadpoolctl import threadpool_limits

from market import CACHE, NAMES, ROOT, TICKERS, random_market, tape
from particle_universal.interval_cover import outward_float
from particle_universal.tilted_interval_cover import TiltedCover


def gamma_estimate(raw):
    r = raw/raw.mean(1,keepdims=True)
    T, d = r.shape
    def target(w):
        s = r @ w
        return -np.log(s).sum(), -(r.T @ (1/s))

    fit = minimize(target, np.ones(d)/d, jac=True, method='SLSQP',
                   bounds=[(0.,1.)]*d, constraints=dict(type='eq',
                   fun=lambda w: w.sum()-1, jac=lambda w: np.ones(d)),
                   options=dict(ftol=1e-12,maxiter=500))
    w = np.clip(fit.x,0,1)
    w /= w.sum()
    f, grad = target(w)
    gap = -(grad.min()-grad @ w)/T
    if gap > 1e-7:
        raise ArithmeticError(f'CRP optimizer daily gap {gap:g}')
    return -f/T, gap, np.count_nonzero(w>1e-5)


def vertex_check(raw):
    """Check the simplex directional derivatives at the best single asset."""
    ctx.prec = 128
    best = np.log(raw).sum(axis=0).argmax()
    columns = [j for j in range(raw.shape[1]) if j != best]
    totals = [arb(0) for _ in columns]

    for row in raw:
        denom = arb(float(row[best]))
        for i, j in enumerate(columns):
            totals[i] += arb(float(row[j])) / denom - 1

    high = max(v.upper() for v in totals)
    return bool(high < 0), int(best), float(high / len(raw))


def vertex_log_evidence(x, d):
    return x - (d - 1) * np.log(x) + lgamma(d)


def winner_checkpoint(raw, best):
    """Report replay zero's certified terminal mixture and winning asset."""
    d = raw.shape[1]
    monitor = TiltedCover(d, exact_pairs=True)
    log_winner = arb(0)
    weights = np.ones(d)

    for row in raw:
        monitor.update(row, weights)
        payoffs = [arb(float(value)) for value in row]
        log_winner += (payoffs[best] * d / sum(payoffs)).log()

    lo, hi = monitor.evidence()
    result = dict(ticker=TICKERS[best], z_lo=outward_float(lo, False),
                  z_hi=outward_float(hi, True),
                  e_lo=outward_float(lo.exp().lower(), False),
                  e_hi=outward_float(hi.exp().upper(), True),
                  winner_lo=outward_float(log_winner.exp().lower(), False),
                  winner_hi=outward_float(log_winner.exp().upper(), True))
    pd.DataFrame([result]).to_csv(ROOT / '5a_winner.csv', index=False)


def projected_day(rate, d):
    left = d - 1
    right = 2 * left

    def distance(x):
        return vertex_log_evidence(x, d) - np.log(20.)

    while distance(right) < 0:
        right *= 2

    return brentq(distance, left, right) / rate


def null_report():
    files = sorted(CACHE.glob('null_*.csv'))
    tables = [pd.read_csv(f, float_precision='round_trip') for f in files]
    d = pd.concat(tables).sort_values(['market', 'rep'])
    names = {'sectors':NAMES[0], 'djia':'DJIA'}
    d.market = d.market.map(names)
    d.to_csv(ROOT/'5a_null.csv',index=False)
    rows = []
    for name in (NAMES[0],'DJIA'):
        group = d.loc[d.market==name]
        R = 1000 if name == NAMES[0] else 200
        assert len(group)==group.rep.nunique()==R
        hits = int(group.reject.sum())
        ci = binomtest(hits,R).proportion_ci(confidence_level=.95)
        quant = np.quantile(np.exp(group.max_logZ),[.5,.9,.99])
        spread = group.max_reference_deviation.max()
        rows.append(dict(market=name,R=R,rejections=hits,rate=hits/R,
                         cp95_lo=ci.low,cp95_hi=ci.high,peak_median=quant[0],
                         peak_p90=quant[1],peak_p99=quant[2],
                         max_screen_difference=group.max_binary_arb_diff.max(),
                         max_reference_spread=spread))
    pd.DataFrame(rows).to_csv(ROOT/'5a_null_summary.csv',index=False)


def power_report():
    data = pd.concat((pd.read_csv(f, float_precision='round_trip')
                      for f in CACHE.glob('power_*.csv')), ignore_index=True)
    data.sort_values(['g','horizon','rep'],inplace=True)
    x = tape('sectors')
    results = []
    rng = np.random.default_rng(583212)
    boot = rng.integers(0,200,size=(4000,200))
    for (gap,T), df in data.groupby(['g','horizon']):
        assert len(df)==df.rep.nunique()==200
        day = df.detect_day.fillna(np.inf).to_numpy()
        median = np.median(day)
        resampled = np.median(day[boot],axis=1)
        lo, hi = np.quantile(resampled,[.025,.975],method='inverted_cdf')
        raw = random_market(x, 0, T, gap)
        gamma, grad_gap, support = gamma_estimate(raw)
        at_vertex, best, derivative = vertex_check(raw)
        if at_vertex != (support == 1):
            raise ArithmeticError('Active set disagrees with KKT check')

        if gap == .8 and T == len(x):
            winner_checkpoint(raw, best)

        dim = x.shape[1]
        proj = projected_day(gamma, dim) if at_vertex else np.nan
        pred = vertex_log_evidence(T * gamma, dim) if at_vertex else np.nan
        data.loc[df.index, 'Gamma_est'] = gamma
        ci = binomtest(int(df.reject.sum()),len(df)).proportion_ci()
        results.append(dict(g=gap,horizon=T,R=200,rejections=df.reject.sum(),
                            rate=df.reject.mean(),cp95_lo=ci.low,
                            cp95_hi=ci.high,median_day=median,
                            median_boot_lo=lo,median_boot_hi=hi,
                            median_censored=np.isinf(median),Gamma_est=gamma,
                            Gamma_fw_gap=grad_gap,best_crp_assets=support,
                            vertex_kkt=at_vertex,vertex_candidate=best + 1,
                            max_direction_daily=derivative,
                            vertex_pred_logZ=pred,implied_day=proj))
    data.to_csv(ROOT/'5a_power.csv',index=False)
    df = pd.DataFrame(results)
    df['median_day_lower_limit'] = np.minimum(df.median_day, df.horizon)
    for col in ('median_day', 'median_boot_lo', 'median_boot_hi'):
        df[col+'_beyond_followup'] = np.isinf(df[col])
        df[col] = df[col].replace(np.inf, np.nan)
    df.to_csv(ROOT/'5a_power_summary.csv',index=False)
    return df


def figures(power):
    plt.rcParams.update({'text.usetex':True,'font.size':10,
                         'axes.spines.top':False,'axes.spines.right':False})
    df = pd.read_csv(ROOT/'5a_paths.csv')
    fig, (ax,bx) = plt.subplots(2,1,sharex=True,figsize=(6.3,4.4),
                                height_ratios=(1,1.2),layout='constrained')
    for name,color in zip([NAMES[0],'NYSE(O)','NYSE(N)'],
                          ['#161616','#32739b','#ad6334']):
        group = df.loc[df.market==name]
        for axis in (ax,bx):
            axis.plot(group.t,np.exp(group.z_lo),lw=.9,color=color,label=name)
    ax.axhline(20,color='0.45',ls='--',lw=.8)
    ax.text(.99,.97,'Rejection threshold: 20',transform=ax.transAxes,
            va='top',ha='right')
    ax.set_yscale('log')
    ax.set_ylim(.88,33)
    ax.set_yticks([1,2,5,10,20],labels=['1','2','5','10','20'])
    ax.set_ylabel('Lower e-value')
    bx.set(xlabel='Trading day',ylabel='Lower e-value (detail)')
    bx.legend(frameon=False,fontsize=9,ncol=3)
    fig.savefig(ROOT/'figure_A.pdf')
    plt.close(fig)

    fig,ax = plt.subplots(figsize=(6.3,3.55),layout='constrained')
    obs = power[power.horizon==6791]
    ext = power[power.horizon>6791]
    ax.axhline(6791/252,color='0.60',lw=1,ls='--')
    for sub, color, label in [(obs,'#202020','Original horizon'),
                              (ext,'#32739b','Extended horizon')]:
        cens = sub[sub.median_censored]
        ax.plot(100*cens.g,cens.horizon/252,'^',fillstyle='none',
                color=color,ms=7,label=label+' (unreached)')
        reached = sub[~sub.median_censored]
        if len(reached):
            ax.plot(100*reached.g,reached.median_day/252,'o',color=color)
    vertex = obs[obs.vertex_kkt]
    ax.plot(100 * vertex.g, vertex.implied_day / 252, 'D', ms=6,
            color='#ad6334', label='Vertex projection, replay 0')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set(xlabel=r'Annual best-to-worst log-growth gap (\%)',
           ylabel='Detection time (trading years)')
    ax.set_xticks([5,10,20,40,80],labels=['5','10','20','40','80'])
    ax.set_xlim(4.5,90)
    ax.set_ylim(22, 135)
    ax.set_yticks([25, 40, 60, 100], labels=['25', '40', '60', '100'])
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.legend(frameon=False,fontsize=9)
    fig.savefig(ROOT/'figure_B.pdf')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--figures', action='store_true')
    mode.add_argument('--summaries', action='store_true')
    options = parser.parse_args()

    with threadpool_limits(1):
        if options.figures:
            frame = pd.read_csv(ROOT / '5a_power_summary.csv')
        else:
            null_report()
            frame = power_report()

        if not options.summaries:
            figures(frame)
