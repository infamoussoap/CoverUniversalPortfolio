"""Input inventory and observed-market tests for Experiment 5a."""

import os
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from flint import arb, ctx

from particle_universal.interval_cover import outward_float
from particle_universal.paths import RESULTS
from study import DATA, PERIOD, evidence, load_panel, tape


ROOT = Path(__file__).resolve().parent
CACHE = Path(os.environ.get('COVER_INFERENCE_CACHE', ROOT / 'cache'))
MARKETS = ('sectors', 'nyse-o', 'nyse-n', 'sp500', 'djia', 'tse', 'msci',
           'stocks')
NAMES = ('Sector ETFs', 'NYSE(O)', 'NYSE(N)', 'SP500', 'DJIA', 'TSE', 'MSCI',
         'Yahoo 50')
TICKERS = 'XLB XLE XLF XLI XLK XLP XLU XLV XLY'.split()
LEVEL = np.log(20.)


def first_rejection(lower):
    threshold = arb(20).log()

    for t in np.flatnonzero(lower >= np.nextafter(LEVEL, -np.inf)):
        if arb(float(lower[t])) >= threshold:
            return int(t) + 1

    return None


def random_market(x, rep, length, g=0., tickers=TICKERS):
    T, d = x.shape
    passes = []
    for turn in range((length+T-1)//T):
        rng = np.random.default_rng([583211, d, rep, turn])
        passes.append(rng.permuted(x[:min(T, length-turn*T)], axis=1))
    raw = np.concatenate(passes)
    if g:
        order = np.random.default_rng(583210).permutation(len(tickers))
        annual = np.empty(d)
        annual[order] = g*(np.arange(d)/(d-1)-.5)
        raw *= np.exp(annual/252)
    return np.ascontiguousarray(raw)


def preflight():
    markets, flags = [], []
    for name, label in zip(MARKETS, NAMES):
        x = tape(name)
        T, d = x.shape
        dates = np.full(T, '', dtype=object)
        names = ['']*d
        if name in PERIOD:
            file = DATA / (name+'.mat')
            start, end = PERIOD[name]
        elif name == 'sectors':
            file = DATA / 'sectors.csv'
            dates = pd.read_csv(file).Date.values[1:]
            start, end = dates[0], dates[-1]
            names = TICKERS
        else:
            panel = load_panel('full')
            dates = panel['dates'].astype(str)
            names = panel['tickers']
            start, end = dates[0], dates[-1]
            file = RESULTS / 'yahoo_fixed50.npz'
        if not np.isfinite(x).all() or x.min() < 0 or (x.max(1) == 0).any():
            raise ValueError(label)
        flag = (x > 2) | (x < .5)
        for t, j in zip(*np.where(flag)):
            flags.append((label,t+1,dates[t],j+1,names[j],x[t,j]))
        markets.append(dict(market=label,d=d,T=T,start=start,end=end,
                            min_m=np.min(x.min(1)/x.mean(1)),
                            n_zero_rows=(x.min(1)==0).sum(),
                            n_zero_relatives=(x==0).sum(),
                            n_extreme_rows=flag.any(1).sum(),
                            checksum=sha256(file.read_bytes()).hexdigest(),
                            tape_sha256=sha256(x.tobytes()).hexdigest()))
    frame = pd.DataFrame(markets)
    frame.to_csv(ROOT/'5a_markets.csv', index=False)
    pd.DataFrame(flags, columns=['market','t','date','asset_index','asset',
                                'raw_relative']).to_csv(ROOT/'5a_extremes.csv',
                                                       index=False)
    print(frame.iloc[:,:9].to_string(index=False))


def real(recompute=False):
    ctx.prec = 128
    results, paths = [], []
    for name, label in zip(MARKETS, NAMES):
        x = tape(name)
        T, d = x.shape
        if x.min() <= 0:
            raise ValueError('Unexpected exact row on market tape')
        fresh = CACHE / f'observed_{name}.npz'
        if recompute or not fresh.exists():
            box, _, sec = evidence(x, 1)
            pending = fresh.with_suffix('.part.npz')
            np.savez(pending, bounds=box, seconds=sec)
            pending.replace(fresh)
        with np.load(fresh) as cached:
            lo, hi = cached['bounds'][:, :2].T
        dates = np.full(T, '', dtype=object)
        if name == 'sectors':
            dates = pd.read_csv(DATA/'sectors.csv').Date.to_numpy()[1:]
        elif name == 'stocks':
            dates = load_panel('full')['dates'].astype(str)
        a = x / x.mean(1, keepdims=True) - 1
        score = a.cumsum(0)
        ss = (a*a).sum(1).cumsum()
        vr = np.divide((score*score).sum(1), ss, out=np.full(T,np.nan),
                       where=ss>0)
        M = np.r_[0., np.einsum('ij,ij->i', a[1:],score[:-1])].cumsum()
        pred = M / (d*(d+1))
        peak = lo.argmax()
        ix = first_rejection(lo)
        e_lo = outward_float(arb(lo[-1]).exp().lower(),False)
        e_hi = outward_float(arb(hi[-1]).exp().upper(),True)
        e_max = outward_float(arb(lo[peak]).exp().lower(),False)
        results.append(dict(market=label,d=d,T=T,term_e_lo=e_lo,term_e_hi=e_hi,
                            max_e_lo=e_max,max_day=peak+1,max_date=dates[peak],
                            reject_day=ix,VR_T=vr[-1],
                            M_T=M[-1],pred_logZ=pred[-1],z_lo=lo[-1],
                            z_hi=hi[-1]))
        paths.append(pd.DataFrame(dict(market=label,t=np.arange(1,T+1),
                                       date=dates,z_lo=lo,z_hi=hi,M=M,VR=vr,
                                       pred_logZ=pred)))
    df = pd.DataFrame(results)
    order = np.argsort(-df.term_e_lo.to_numpy())
    ranks = np.arange(1, len(df)+1)
    lower_values = df.term_e_lo.to_numpy()[order]
    ok = np.array([arb(float(value)) * int(rank) >= 20 * len(df)
                   for rank, value in zip(ranks, lower_values)])
    df['ebh_reject'] = False
    if ok.any():
        df.loc[order[:max(ranks[ok])], 'ebh_reject'] = True
    df.to_csv(ROOT/'5a_summary.csv', index=False)
    pd.concat(paths).to_csv(ROOT/'5a_paths.csv', index=False)


if __name__ == '__main__':
    import sys
    if sys.argv[1:] == ['--preflight']:
        preflight()
    else:
        real(sys.argv[1:] == ['--recompute'])
