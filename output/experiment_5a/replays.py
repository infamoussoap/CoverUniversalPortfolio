"""Permutation calibration and paired drift power for Experiment 5a."""

import argparse

import numpy as np
import pandas as pd
from flint import arb, ctx
from numba import njit
from scipy.stats import qmc
from threadpoolctl import threadpool_limits

from market import CACHE, LEVEL, first_rejection, random_market, tape
from particle_universal.interval_cover import outward_float
from particle_universal.market_reference import _prior_particles
from particle_universal.tilted_interval_cover import TiltedCover
from reference import mass_residual
from screen import screen


GAPS = (.05,.1,.2,.4,.8)


@njit(cache=True)
def control_mass(a):
    T, d = a.shape
    g, cubes, weighted, diag = np.zeros((4, d))
    B = np.zeros((d,d))
    r2 = d*(d+1)
    r3 = r2*(d+2)
    r4 = r3*(d+3)
    ek = 0.
    out = np.empty(T)
    for t in range(T):
        row = a[t]
        sq = row*row
        size = sq.sum()
        g += row
        cubes += sq*row
        weighted += size*row
        diag += sq
        ek += (3*size**2+6*(sq @ sq))/r4
        frob, bilinear = 0., 0.
        for i in range(d):
            for j in range(d):
                B[i,j] += row[i]*row[j]
                frob += B[i,j]**2
                bilinear += g[i]*B[i,j]*g[j]
        tr = diag.sum()
        gg = g @ g
        q = (gg-tr)/(2*r2)
        cubic = ((g**3).sum()/3-g @ diag+2*cubes.sum()/3)/r3
        g4 = (3*gg**2+6*(g**4).sum())/r4
        gj = (3*(g @ weighted)+6*(g @ cubes))/r4
        q2 = (tr**2+2*frob+6*(diag @ diag))/r4
        g2q = (gg*tr+2*bilinear+6*((g*g) @ diag))/r4
        out[t] = 1+q+cubic+g4/24-g2q/4+q2/8+gj/3-ek/4
    return out


def estimate_maxima(raw, supports):
    a = raw / raw.mean(1,keepdims=True) - 1
    a -= a.mean(1,keepdims=True)
    control = control_mass(a)
    residual = []
    for points in supports:
        sums = np.zeros(len(a))
        for i in range(0,len(points),128):
            sums += mass_residual(a @ points[i:i+128].T)
        residual.append(sums/len(points))
    log_ref = np.log(control+np.mean(residual,axis=0))
    logs = np.log(control+np.array(residual))
    peak = log_ref.argmax()
    return log_ref[peak], peak+1, np.max(abs(logs-log_ref))


def verified(raw, rep, name, full=False):
    pre = screen(raw,full)
    if full:
        with np.load(CACHE/f'arb_{name}_{rep}.npz') as saved:
            lo = saved['lo']
        gap = max(abs(lo-pre))
        return first_rejection(lo), len(raw), gap
    candidates = np.flatnonzero(pre >= LEVEL-.05)
    if candidates.size == 0:
        return None, 0, np.nan
    ctx.prec, ctx.threads = 128, 1
    mon = TiltedCover(raw.shape[1], exact_pairs=True)
    w = np.ones(raw.shape[1])
    prev = 0
    max_diff = 0.
    for n, i in enumerate(candidates,1):
        for row in raw[prev:i+1]:
            mon.update(row,w)
        prev = i+1
        lo = mon.evidence()[0]
        max_diff = max(max_diff,abs(float(lo)-pre[i]))
        if lo >= arb(20).log():
            return int(i)+1, n, max_diff
    return None, len(candidates), max_diff


def save(records, path):
    temp = path.with_suffix('.part')
    pd.DataFrame(records).to_csv(temp,index=False)
    temp.replace(path)


def null(name, chunk, size):
    x = tape(name)
    R = 1000 if name == 'sectors' else 200
    dest = CACHE/f'null_{name}_{chunk}.csv'
    if dest.exists():
        return
    d = x.shape[1]
    supports = [_prior_particles(d,9,np.random.default_rng([583213,d,0]))]
    rng = np.random.default_rng([583213,d,1])
    u = qmc.Sobol(d,scramble=True,rng=rng).random_base2(9)
    v = -np.log1p(-u)
    supports.append(v/v.sum(1,keepdims=True))
    records = []
    for rep in range(chunk*size,min(R,(chunk+1)*size)):
        raw = random_market(x,rep,len(x))
        peak, day, spread = estimate_maxima(raw,supports)
        hit, n, gap = verified(raw,rep,name,rep<10)
        records.append(dict(market=name,rep=rep,max_logZ=peak,max_day=day,
                            reject=hit is not None,reject_day=hit,
                            certified_in_arb=n>0,arb_closes=n,full_arb=rep<10,
                            max_binary_arb_diff=gap,
                            max_reference_deviation=spread))
    save(records,dest)
    print('null',name,chunk,flush=True)


def power(chunk, size, gap, horizon):
    x = tape('sectors')
    dest = CACHE/f'power_{gap:g}_{horizon}_{chunk}.csv'
    if dest.exists():
        return
    records = []
    for rep in range(chunk*size,min(200,(chunk+1)*size)):
        raw = random_market(x,rep,horizon,gap)
        hit, n, diff = verified(raw,rep,'sectors')
        records.append(dict(g=gap,rep=rep,horizon=horizon,
                            reject=hit is not None,detect_day=hit,arb_closes=n,
                            max_binary_arb_diff=diff))
    save(records,dest)
    print('power',gap,horizon,chunk,flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('job', choices=['null','power','extended'])
    p.add_argument('--partition',type=int,default=0)
    p.add_argument('--partitions',type=int,default=1)
    opt = p.parse_args()
    ctx.prec, ctx.threads = 128, 1
    with threadpool_limits(1):
        if opt.job == 'null':
            for name, count in [('sectors',1000),('djia',200)]:
                for i in range(opt.partition,count//20,opt.partitions):
                    null(name,i,20)
        else:
            T = len(tape('sectors'))
            gaps = GAPS
            if opt.job == 'extended':
                candidates = []
                for gap in gaps:
                    paths = sorted(CACHE.glob(f'power_{gap:g}_{T}_*.csv'))
                    df = pd.concat([pd.read_csv(path) for path in paths])
                    assert len(df)==200 and df.rep.nunique()==200
                    if df.reject.mean() < .5:
                        candidates.append(gap)
                gaps, T = candidates[:2], 25000
            for gap in gaps:
                for i in range(opt.partition,10,opt.partitions):
                    power(i,20,gap,T)
