"""Bound the exact mixture independently on every relabeled path."""

import numpy as np
import pandas as pd
from flint import arb, ctx

from market import CACHE, ROOT, random_market, tape
from particle_universal.interval_cover import moment_bounds, outward_float


def cap(block):
    # See docs/inference.md for the cumulative score error bound.
    pad = arb(1)/50000
    g = [arb(v)+pad for v in block.max(0)]
    simple = max(v.upper() for v in g)
    if simple < arb(20).log():
        return simple
    return moment_bounds(g)[0][1].log().upper()


def certify(raw):
    T, d = raw.shape
    assert T <= 25000 and d <= 30 and .1 <= raw.min() <= raw.max() <= 5
    score = np.cumsum(raw/raw.mean(1,keepdims=True)-1,axis=0)
    threshold = arb(20).log()
    peak = arb(0)
    possible = 0
    for i in range(0,T,256):
        block = score[i:i+256]
        high = cap(block)
        if high < threshold:
            peak = max(peak,high)
            continue
        for j in range(0,len(block),16):
            short = block[j:j+16]
            local = cap(short)
            if local < threshold:
                peak = max(peak,local)
                continue
            for row in short:
                value = cap(row[None,:])
                peak = max(peak,value)
                possible += int(not value < threshold)
    return outward_float(peak,True), possible


def run():
    ctx.prec, ctx.threads = 128, 1
    jobs = [('sectors',0.,6791,1000), ('djia',0.,507,200)]
    jobs += [('sectors',g,6791,200) for g in (.05,.1,.2,.4,.8)]
    for g in (.05,.1,.2,.4,.8):
        if list(CACHE.glob(f'power_{g:g}_25000_*.csv')):
            jobs.append(('sectors',g,25000,200))
    rows = []
    for name,g,T,R in jobs:
        x = tape(name)
        for rep in range(R):
            high, possible = certify(random_market(x,rep,T,g))
            rows.append(dict(market=name,g=g,horizon=T,rep=rep,
                             max_logZ_upper=high,unresolved_closes=possible))
        print(name,g,T,'unresolved',sum(r['unresolved_closes'] for r in
                                         rows[-R:]),flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'5a_threshold_audit.csv',index=False)


if __name__=='__main__':
    run()
