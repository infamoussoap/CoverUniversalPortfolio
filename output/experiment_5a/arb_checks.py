"""Full-path interval checks and the independent two-asset integral."""

import argparse

import numpy as np
import pandas as pd
from flint import ctx
from scipy.special import roots_legendre
from threadpoolctl import threadpool_limits

from evalues import positive_pair
from market import CACHE, ROOT, random_market, tape
from study import evidence


def null_checks(name):
    x = tape(name)
    for rep in range(10):
        path = CACHE / f'arb_{name}_{rep}.npz'
        if path.exists():
            continue
        raw = random_market(x, rep, len(x))
        box, _, sec = evidence(raw, 1)
        pending = path.with_suffix('.part.npz')
        np.savez(pending, lo=box[:, 0], hi=box[:, 1], seconds=sec)
        pending.replace(path)
        print(name, rep, sec, flush=True)


def gauss(x):
    n = (len(x)+2)//2
    nodes, weights = roots_legendre(n)
    nodes = (nodes+1)/2
    weights /= weights.sum()
    relative = x / x.mean(1, keepdims=True)
    log_value = np.zeros(n)
    result = []
    for a, b in relative:
        log_value += np.log(b+(a-b)*nodes)
        result.append(np.log(weights @ np.exp(log_value)))
    return np.asarray(result)


def pair():
    x = tape('utilities_tech')
    results = []
    for g in (0., .8):
        raw = x if g == 0 else random_market(x, 0, len(x), g, ['XLK','XLU'])
        exact = positive_pair(raw)
        gl = gauss(raw)
        box, _, _ = evidence(raw, 1)
        data = pd.DataFrame(dict(path='observed' if g == 0 else 'planted',
                                 g=g,t=np.arange(1,len(x)+1),
                                 exact_logZ=exact.mean(1),gl_logZ=gl,
                                 z_lo=box[:,0],z_hi=box[:,1],
                                 exact_lo=exact[:,0],exact_hi=exact[:,1]))
        data['exact_inside'] = ((data.exact_lo >= data.z_lo)
                                & (data.exact_hi <= data.z_hi))
        if not data.exact_inside.all():
            raise ArithmeticError('Bernstein interval outside the band.')

        results.append(data)
        print(g,data.exact_inside.sum(),
              abs(data.gl_logZ-data.exact_logZ).max(), flush=True)
    pd.concat(results).to_csv(ROOT/'5a_exact_d2.csv', index=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('name')
    cmd = p.parse_args()
    ctx.prec, ctx.threads = 128, 1
    with threadpool_limits(1):
        if cmd.name == 'pair':
            pair()
        else:
            null_checks(cmd.name)
