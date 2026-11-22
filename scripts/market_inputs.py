"""Read the OLPS, FinOL and Yahoo source formats used in the paper."""

import hashlib
import json
import re
import shutil
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat


def classic_matrix(path):
    if h5py.is_hdf5(path):
        with h5py.File(path) as handle:
            return np.ascontiguousarray(handle['data'][:].T)

    return np.ascontiguousarray(loadmat(path)['data'])


def build_finol(folder, spec, dest, test):
    dates = None
    close = []
    files = sorted(folder.glob('*.xlsx'))
    if [p.name for p in files] != sorted(spec['files']):
        raise ValueError(f'Unexpected workbook inventory in {folder}')

    for path in files:
        with path.open('rb') as handle:
            checksum = hashlib.file_digest(handle, 'sha256').hexdigest()
        if checksum != spec['files'][path.name]:
            raise ValueError(f'Workbook checksum mismatch: {path}')

        with ZipFile(path) as book:
            xml = book.read('xl/worksheets/sheet1.xml')

        if b'<c r="E1"' not in xml or b'<t>CLOSE</t>' not in xml:
            raise ValueError(f'Missing CLOSE header in {path}')

        cells = re.findall(rb'<c r="([AE])(\d+)"[^>]*><v>([^<]+)</v>', xml)
        cols = [[(int(i), float(v)) for c, i, v in cells if c == k]
                for k in (b'A', b'E')]
        days, prices = (np.asarray(values) for values in cols)
        np.testing.assert_array_equal(days[:, 0], prices[:, 0])
        if dates is None:
            dates = days[:, 1]
        np.testing.assert_array_equal(days[:, 1], dates)
        close.append(prices[:, 1])

    close = np.array(close).T
    if not np.isfinite(close).all() or close.min() <= 0:
        raise ValueError('Invalid FinOL CLOSE entry')

    returns = np.ascontiguousarray(close[1:] / close[:-1])
    returns = np.vstack((returns, np.ones(returns.shape[1])))
    calendar = np.datetime64('1899-12-30') + dates.astype('timedelta64[D]')
    tickers = np.asarray([p.stem for p in files])
    released = loadmat(test)['data']
    np.testing.assert_array_equal(returns[-len(released):], released)

    with TemporaryDirectory(dir=dest.parent) as staging:
        temp = Path(staging) / dest.name
        np.savez(temp, returns=returns, dates=calendar, tickers=tickers)
        temp.replace(dest)


def yahoo_fetch(name, spec, path):
    import yfinance as yf

    tickers = spec['tickers']
    query = tickers + ['SPY'] if name == 'stocks' else tickers
    frame = yf.download(query, start=spec['base_date'], end='2026-01-01',
                        interval='1d', auto_adjust=False, repair=False,
                        actions=False, threads=False, progress=False)
    if frame is None or frame.empty:
        raise RuntimeError(f'Yahoo returned no adjusted closes for {name}')

    prices = frame['Adj Close']
    if name == 'stocks':
        prices = prices.loc[prices['SPY'].dropna().index, tickers]
    else:
        prices = prices.loc[:, tickers]

    if prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise ValueError(f'Invalid date order in {name}')
    values = prices.to_numpy(dtype='float64')
    if not np.isfinite(values).all() or values.min() <= 0:
        raise ValueError(f'Missing or nonpositive history in {name}')

    if name == 'sectors':
        prices.to_csv(path, index_label='Date')
    else:
        dates = prices.index.strftime('%Y-%m-%d').to_numpy(dtype='U10')
        returns = values[1:] / values[:-1]
        np.savez_compressed(path, tickers=np.asarray(tickers),
                            price_dates=dates, dates=dates[1:],
                            adjusted_close=values, returns=returns)


def yahoo_identity(name, path, spec):
    if name == 'stocks':
        with np.load(path, allow_pickle=False) as tape:
            r = tape['returns']
            dates = tape['price_dates'].astype('U10')
            tickers = list(tape['tickers'])
            closes = tape['adjusted_close']
            np.testing.assert_array_equal(tape['dates'], dates[1:])
            np.testing.assert_array_equal(r, closes[1:] / closes[:-1])
    else:
        # Match the decimal-to-binary64 conversion in the paper's sector run.
        frame = pd.read_csv(path, index_col=0)
        tickers = list(frame.columns)
        dates = np.asarray(frame.index, dtype='U10')
        closes = frame.to_numpy()
        r = closes[1:] / closes[:-1]

    calendar = hashlib.sha256(dates.astype('S10').tobytes()).hexdigest()
    if (list(r.shape) != spec['shape'] or tickers != spec['tickers']
            or calendar != spec['dates_sha256'] or not np.isfinite(r).all()
            or not np.isfinite(closes).all() or closes.min() <= 0):
        raise ValueError(f'{name} has different constituents, dates or shape,'
                         ' or contains invalid data')

    price_hash = hashlib.sha256(np.asarray(r, dtype='<f8').tobytes())
    with path.open('rb') as src:
        file_hash = hashlib.file_digest(src, 'sha256').hexdigest()
    return dict(returns_sha256=price_hash.hexdigest(), source_sha256=file_hash)


def prepare_yahoo(manifest, data, stock, args, verify):
    datasets = [('stocks', stock, args.stock_snapshot),
                ('sectors', data / 'sectors.csv', args.sector_snapshot)]
    provenance = {}

    if not verify and any(not snap and not dest.exists()
                          for _, dest, snap in datasets):
        import yfinance as yf
        yf.set_tz_cache_location(str(data / 'yfinance-cache'))

    for name, dest, snapshot in datasets:
        spec = manifest[name]
        if verify and not dest.is_file():
            raise FileNotFoundError(dest)
        if not verify:
            dest.parent.mkdir(parents=True, exist_ok=True)

        fresh = not verify and (snapshot or not dest.is_file())
        stage = TemporaryDirectory(dir=dest.parent) if fresh else nullcontext()
        with stage as staging:
            source = dest
            acquired = 'existing local file'
            if fresh:
                source = Path(staging) / dest.name
                if snapshot:
                    shutil.copyfile(snapshot, source)
                    acquired = 'imported snapshot'
                else:
                    yahoo_fetch(name, spec, source)
                    acquired = 'downloaded via yfinance'

            identity = yahoo_identity(name, source, spec)
            exact = identity['returns_sha256'] == spec['returns_sha256']
            if not exact and not args.allow_revised_yahoo:
                raise ValueError(
                    f'{name} return hash: {identity["returns_sha256"]}; paper '
                    f'{spec["returns_sha256"]}. To study revised Yahoo prices '
                    'use --allow-revised-yahoo in a fresh checkout.')

            if source != dest:
                if (dest.exists()
                        and yahoo_identity(name, dest, spec) != identity):
                    raise ValueError(f'{dest} already contains another tape. '
                                     'Use a fresh checkout for a new vintage.')
                source.replace(dest)

        status = 'paper returns match' if exact else 'revised Yahoo returns'
        print(f'{name}: {status}, {identity["returns_sha256"]}')
        provenance[name] = dict(**identity, paper_returns_match=exact,
                                acquisition=acquired)
        if acquired == 'downloaded via yfinance':
            import yfinance as yf
            provenance[name]['retrieved_utc'] = datetime.now(
                timezone.utc).isoformat()
            provenance[name]['yfinance_version'] = yf.__version__

    if not verify:
        report = data / 'yahoo_provenance.json'
        prior = json.loads(report.read_text()) if report.is_file() else {}
        for name, rec in provenance.items():
            if (rec['acquisition'] != 'existing local file' or
                    rec['source_sha256'] != prior.get(name, {}).get(
                        'source_sha256')):
                prior[name] = rec
        stage = report.with_suffix('.tmp')
        stage.write_text(json.dumps(prior, indent=2) + '\n')
        stage.replace(report)
