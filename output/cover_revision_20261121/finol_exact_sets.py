"""Compare the empty and five nested exact-row sets at the terminal close."""

import csv
from time import perf_counter

import numpy as np
from flint import arb, ctx
from threadpoolctl import threadpool_limits

from extra_checks import OUT, ends
from finol_product_moments import product_endpoints, validate_moments
from particle_universal.tilted_interval_cover import TiltedCover
from study import CLASSIC, DATA, tape


def lowest_rows(x):
    """Certify the five lowest row minima after cross-sectional scaling.

    Parameters
    ----------
    x : (T, d) np.ndarray
        Daily gross returns for T closes and d assets.

    Returns
    -------
    indices : (5,) np.ndarray
        Zero-based row indices in increasing order of normalized minimum.
    """
    d = x.shape[1]
    approx = x.min(axis=1) / x.mean(axis=1)
    order = np.argsort(approx, kind='stable')

    def exact_minimum(t):
        row = [arb(float(v)) for v in x[t]]
        return min(row) * d / sum(row)

    ordered = [exact_minimum(t) for t in order[:6]]
    if not all(a < b for a, b in zip(ordered, ordered[1:])):
        raise ArithmeticError('Rounded inputs did not separate the six minima')
    if not all(ordered[-1] < exact_minimum(t) for t in order[6:]):
        raise ArithmeticError('Sixth row is not an isolated cutoff')

    return order[:5]


def terminal_sets(name, control_output=None):
    """Enclose terminal evidence with up to five extreme rows held exact.

    Parameters
    ----------
    name : str
        Market identifier accepted by study.tape.
    control_output : pathlib.Path or None
        Optional destination for the empty-set cubic-bound ablation.

    Returns
    -------
    records : list of dict
        Six terminal enclosures ordered by the number of exact rows.
    """
    x = tape(name)
    n, d = x.shape
    chosen = lowest_rows(x)
    dates, assets = [''] * n, [str(i + 1) for i in range(d)]
    if name == 'finol_nyse':
        with np.load(DATA / 'finol_nyse.npz') as data:
            dates, assets = data['dates'], data['tickers']

    monitor = TiltedCover(d, exact_pairs=True)
    w = np.ones(d)
    omitted = set(chosen)
    for t, row in enumerate(x):
        if t not in omitted:
            monitor.update(row, w)

    records = []
    for k in range(5, -1, -1):
        active = chosen[:k]
        start = perf_counter()
        if k:
            lo, hi = product_endpoints(monitor, x[active])
        elif control_output is None:
            lo, hi = monitor.evidence()
        else:
            previous = []
            lo, hi = monitor.evidence(
                legacy_bounds=lambda a, b: previous.extend((a, b)))
            controls = []
            for method, a, b in [('without_cubic_bound', *previous),
                                  ('default', lo, hi)]:
                lower, upper = ends(a, b)
                controls.append(dict(dataset=name, S_size=0, method=method,
                                     z_lo=lower, z_hi=upper,
                                     width=ends(a, b - a)[1]))
            save(controls, control_output)
        lower, upper = ends(lo, hi)
        weakest = [assets[x[t].argmin()] for t in active]

        records.append(dict(
            dataset=name, d=d, T=n, S_size=k,
            exact_days='|'.join(str(t + 1) for t in active),
            dates='|'.join(str(dates[t]) for t in active),
            weakest_assets='|'.join(map(str, weakest)),
            z_lo=lower, z_hi=upper, width=ends(lo, hi - lo)[1],
            checkpoint_seconds=perf_counter() - start, bits=ctx.prec))
        if k:
            monitor.update(x[active[-1]], w)

    # S=0 and S=1..5 all enclose the same normalized evidence.
    low = max(r['z_lo'] for r in records)
    high = min(r['z_hi'] for r in records)
    if not low <= high:
        raise ArithmeticError(f'Disjoint enclosures on {name}')

    return list(reversed(records))


def save(records, path):
    temp = path.with_suffix('.part')
    with temp.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)
    temp.replace(path)


def run():
    validate_moments()
    records = []
    for name in (*CLASSIC, 'finol_nyse'):
        control = None
        if name == 'finol_nyse':
            control = OUT / 'finol_nyse_empty_set_control.csv'
        rows = terminal_sets(name, control_output=control)
        print(name, [r['width'] for r in rows], flush=True)
        records.extend(rows)
        if name == 'finol_nyse':
            save(rows[1:], OUT / 'finol_nyse_exact_sets.csv')

    save(records, OUT / 'exact_sets_all.csv')


if __name__ == '__main__':
    ctx.prec, ctx.threads = 128, 1
    with threadpool_limits(1):
        run()
