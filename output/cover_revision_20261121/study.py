"""Replay the November experiment on exact binary64 input tapes."""

import argparse
import hashlib
import json
import resource
import sys
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
import pandas as pd
from flint import arb, ctx
from scipy.io import loadmat
from threadpoolctl import threadpool_limits

from particle_universal.interval_cover import outward_float
from particle_universal.tilted_interval_cover import TiltedCover
from particle_universal.yahoo_daily_portfolios import load_panel


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / 'cache'
DATA = ROOT / 'data'
CLASSIC = ('djia', 'msci', 'sp500', 'nyse-o', 'nyse-n', 'tse')
PERIOD = {'nyse-o': ('1962-07-03', '1984-12-31'),
          'nyse-n': ('1985-01-01', '2010-06-30'),
          'djia': ('2001-01-14', '2003-01-14'),
          'sp500': ('1998-01-02', '2003-01-31'),
          'msci': ('2006-04-01', '2010-03-31'),
          'tse': ('1994-01-04', '1998-12-31')}
GRID = (0.25, 0.5, 1, 2, 4)


def peak_mib():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024**2 if sys.platform == 'darwin' else 1024)


def tape(name):
    if name in CLASSIC:
        filename = DATA / (name + '.mat')
        if h5py.is_hdf5(filename):
            with h5py.File(filename) as handle:
                return np.ascontiguousarray(handle['data'][:].T)

        return np.ascontiguousarray(loadmat(filename)['data'])

    if name in ('stocks', 'pair'):
        panel = load_panel('full')
        x = panel['returns']
        cols = [panel['tickers'].tolist().index(s) for s in ('AAPL', 'MSFT')]
        return x if name == 'stocks' else x[:, cols]

    if name in ('sectors', 'utilities_tech'):
        df = pd.read_csv(DATA / 'sectors.csv', index_col=0)
        if name == 'utilities_tech':
            df = df[['XLK', 'XLU']]

        prices = df.to_numpy()
        return np.ascontiguousarray(prices[1:] / prices[:-1])

    return np.load(DATA / (name + '.npz'))['returns']


def check_inputs(names):
    """Bind existing checkpoint names to the exact market inputs."""
    CACHE.mkdir(exist_ok=True)
    for name in names:
        x = tape(name)
        fingerprint = hashlib.sha256()
        fields = [x]
        if name in ('stocks', 'pair'):
            panel = load_panel('full')
            fields.extend((panel['dates'], panel['tickers']))
        elif name in ('finol_nyse', 'finol_sp'):
            with np.load(DATA / (name + '.npz')) as data:
                fields.extend((data['dates'], data['tickers']))
        elif name in ('sectors', 'utilities_tech'):
            dates = pd.read_csv(DATA / 'sectors.csv', usecols=['Date'])
            fields.append(dates.Date.to_numpy(dtype='U'))

        for field in fields:
            array = np.ascontiguousarray(field)
            if array.dtype.hasobject:
                array = array.astype('U')
            fingerprint.update(str((array.dtype.str, array.shape)).encode())
            fingerprint.update(array.tobytes())

        key = fingerprint.hexdigest()
        record = CACHE / f'input_{name}.sha256'
        if record.exists():
            if record.read_text().strip() == key:
                continue
        else:
            prefixes = (f'{name}_*', f'w_{name}_*', f'cost_{name}_*',
                        f'g_w_{name}_*', f'worst_row_{name}.csv')
            if not any(path for pat in prefixes for path in CACHE.glob(pat)):
                record.write_text(key + '\n')
                continue

        raise RuntimeError(
            f'Inputs for {name} differ from the saved checkpoints, or the '
            'checkpoints have no input signature. Move the benchmark cache '
            'directory aside before running on a new vintage: ' + str(CACHE))


def preflight():
    summary, flags = [], []
    for name in CLASSIC:
        x = tape(name)
        if np.any(x < 0) or not np.isfinite(x).all() or np.any(x.max(1) == 0):
            raise ValueError(name)

        a = x / x.mean(axis=1, keepdims=True) - 1
        extreme = (x < 0.5) | (x > 2)
        for t, j in zip(*np.where(extreme)):
            flags.append(dict(dataset=name, t=t + 1, asset=j + 1,
                              date='', price_relative=x[t, j]))

        digest = hashlib.sha256((DATA / (name + '.mat')).read_bytes())
        start, end = PERIOD[name]
        g = np.cumsum(a, axis=0)
        summary.append(dict(dataset=name, d=x.shape[1], T=len(x),
                            start=start, end=end, min_m=1 + a.min(),
                            min_raw=x.min(), n_zero_rows=(x.min(1) == 0).sum(),
                            n_extreme_rows=extreme.any(1).sum(), S_size=0,
                            score_range=np.ptp(g[-1]), checksum=digest.hexdigest()))
    pd.DataFrame(summary).to_csv(ROOT / 'bench_datasets.csv', index=False)
    pd.DataFrame(flags).to_csv(ROOT / 'bench_extremes.csv', index=False)


def evidence(x, alpha, pairs=True):
    ctx.prec, ctx.threads = 128, 1
    start = perf_counter()
    d = x.shape[1]
    monitor = TiltedCover(d, alpha, exact_pairs=pairs)
    w = np.ones(d)
    bounds = np.empty((len(x), 4))
    degrees = np.empty(len(x), int)
    log_norm = arb(0)

    for i, row in enumerate(x):
        monitor.update(row, w)
        low, high = monitor.evidence()
        span = max(v.upper() for v in monitor.g) - min(
            v.lower() for v in monitor.g)
        degrees[i] = (max(30, 4 * int(float(span) + 1) + 16) if pairs
                      else max(24, int(4 * float(span)) + 1))
        log_norm += (sum(arb(float(v)) for v in row) / d).log()
        bounds[i] = (outward_float(low.lower(), False),
                     outward_float(high.upper(), True),
                     outward_float(log_norm.lower(), False),
                     outward_float(log_norm.upper(), True))

    return bounds, degrees, perf_counter() - start


def cache_path(name, alpha, kind='pair'):
    return CACHE / f'{name}_{float(alpha):g}_{kind}.npz'


def run(name, alpha, repeats):
    dest = cache_path(name, alpha)
    times = []
    if dest.exists():
        with np.load(dest) as prev:
            times = prev['times'].tolist()

    x = tape(name)
    while len(times) < repeats:
        bounds, degrees, seconds = evidence(x, alpha)
        if dest.exists():
            with np.load(dest) as prev:
                np.testing.assert_array_equal(bounds, prev['bounds'])

        times.append(seconds)
        rss = peak_mib()
        temp = dest.with_suffix('.part.npz')
        np.savez(temp, bounds=bounds, degrees=degrees, times=times, rss=rss)
        temp.replace(dest)
        print(name, alpha, len(times), seconds, bounds[-1], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('names', nargs='*', default=[])
    parser.add_argument('--alphas', nargs='+', type=float, default=[1])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()

    with threadpool_limits(1):
        if args.preflight:
            preflight()
        for a in args.alphas:
            for name in args.names:
                run(name, a, args.repeats)
