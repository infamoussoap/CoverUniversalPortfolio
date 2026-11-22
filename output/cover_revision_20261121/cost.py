"""Measure nested prefix costs in fresh, single-threaded worker processes."""

import argparse
import json
import os
import subprocess
import sys
from time import perf_counter

import numpy as np
from flint import arb, ctx
from threadpoolctl import threadpool_limits

from particle_universal.interval_cover import outward_float
from particle_universal.tilted_interval_cover import TiltedCover
from study import CACHE, peak_mib, tape


def measure(name, d, ends, rep, pairs, alpha=1):
    ctx.prec, ctx.threads = 128, 1
    x = tape(name)[:, :d]
    w = np.ones(d)
    start = perf_counter()
    monitor = TiltedCover(d, alpha, exact_pairs=pairs)
    degree = 0
    for t, row in enumerate(x[:max(ends)], start=1):
        monitor.update(row, w)
        lo, hi = monitor.evidence()
        span = max(v.upper() for v in monitor.g) - min(
            v.lower() for v in monitor.g)
        n = (max(30, 4 * int(float(span) + 1) + 16) if pairs
             else max(24, int(4 * float(span)) + 1))
        degree = max(degree, n)
        if t in ends:
            kind = 'pair' if pairs else 'taylor'
            record = dict(dataset=name, d=d, T=t, method=kind, rep=rep,
                          seconds=perf_counter()-start, N_max=degree,
                          peak_mem_mb=peak_mib(),
                          term_width=outward_float((hi-lo).upper(), True))
            suffix = '' if alpha == 1 else f'_alpha{alpha:g}'
            dest = CACHE / f'cost_{name}_{d}_{t}_{kind}_{rep}{suffix}.json'
            record['alpha'] = alpha
            temp = dest.with_suffix('.part')
            temp.write_text(json.dumps(record))
            temp.replace(dest)
            print(name, d, t, rep, kind, record['seconds'], flush=True)


def coordinate():
    cases = [('finol_nyse', d, (1000, 2000, 4000, 6430), True)
             for d in (25, 50, 100, 200, 403)]
    cases += [('finol_sp', 339, (1268, len(tape('finol_sp'))), True)]
    cases += [('finol_nyse', 25, (1000,), False)]
    for rep in range(3):
        for name, d, ends, pairs in cases:
            kind = 'pair' if pairs else 'taylor'
            if (CACHE / f'cost_{name}_{d}_{max(ends)}_{kind}_{rep}.json').exists():
                continue
            args = [sys.executable, __file__, '--name', name, '--d', str(d),
                    '--ends', *map(str, ends), '--rep', str(rep)]
            if not pairs:
                args.append('--taylor')
            subprocess.run(args, check=True, env=os.environ.copy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name')
    parser.add_argument('--d', type=int)
    parser.add_argument('--ends', type=int, nargs='+')
    parser.add_argument('--rep', type=int)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--taylor', action='store_true')
    opt = parser.parse_args()
    with threadpool_limits(1):
        if opt.name:
            measure(opt.name, opt.d, opt.ends, opt.rep, not opt.taylor,
                    opt.alpha)
        else:
            coordinate()
