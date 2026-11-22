"""Measure terminal Cover enclosures on nested exact-row sets in FinOL NYSE."""

import csv
from time import perf_counter

import numpy as np
from flint import arb, ctx
from threadpoolctl import threadpool_limits

from extra_checks import OUT, ends
from particle_universal.tilted_interval_cover import TiltedCover
from study import DATA


def run():
    panel = np.load(DATA / 'finol_nyse.npz')
    x = panel['returns']
    d = x.shape[1]
    approx = x.min(axis=1) / x.mean(axis=1)
    order = np.argsort(approx, kind='stable')
    chosen = list(order[:5])
    # Verify the sorting certificate, including the sixth-smallest cutoff.
    def exact_minimum(t):
        row = [arb(float(v)) for v in x[t]]
        return min(row) * d / sum(row)

    ordered = [exact_minimum(t) for t in order[:6]]
    assert all(z < y for z, y in zip(ordered, ordered[1:]))
    assert all(ordered[-1] < exact_minimum(t) for t in order[6:])

    start = perf_counter()
    weights = np.ones(d)
    mon = TiltedCover(d, exact_pairs=True)
    for t, row in enumerate(x):
        if t not in chosen:
            mon.update(row, weights)

    print('Complement accumulated', perf_counter() - start, flush=True)
    from finol_product_moments import product_endpoints, validate_moments
    validate_moments()
    out = OUT / 'finol_nyse_exact_sets.csv'
    records = []

    for k in range(5, 0, -1):
        active = chosen[:k]
        start = perf_counter()
        lo, hi = product_endpoints(mon, x[active])
        a, b = ends(lo, hi)
        row = dict(dataset='finol_nyse', d=d, T=len(x), S_size=k,
                   exact_days='|'.join(str(t + 1) for t in active),
                   dates='|'.join(str(panel['dates'][t]) for t in active),
                   weakest_assets='|'.join(str(panel['tickers'][x[t].argmin()])
                                           for t in active),
                   z_lo=a, z_hi=b, width=ends(lo, hi - lo)[1],
                   checkpoint_seconds=perf_counter() - start, bits=ctx.prec)
        records.append(row)
        print(k, a, b, row['width'], row['checkpoint_seconds'], flush=True)
        with out.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=row)
            writer.writeheader()
            writer.writerows(reversed(records))

        mon.update(x[active[-1]], weights)

    assert abs(records[-1]['width'] - 2.832528830814004) < 1e-10

    controls = []
    for method in ('default', 'matched'):
        start = perf_counter()
        if method == 'default':
            lo, hi = mon.evidence()
        else:
            lo, hi = product_endpoints(mon, [])

        a, b = ends(lo, hi)
        controls.append(dict(dataset='finol_nyse', S_size=0, method=method,
                             z_lo=a, z_hi=b, width=ends(lo, hi - lo)[1],
                             checkpoint_seconds=perf_counter() - start))

    with (OUT / 'finol_nyse_empty_set_control.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=controls[0])
        writer.writeheader()
        writer.writerows(controls)


if __name__ == '__main__':
    ctx.prec, ctx.threads = 128, 1
    with threadpool_limits(1):
        run()
