"""Prepare the market inputs and verify their experiment fingerprints."""

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from market_inputs import build_finol, classic_matrix, prepare_yahoo


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'output' / 'cover_revision_20261121' / 'data'
STOCK = ROOT / 'particle_universal' / 'results' / 'yahoo_fixed50.npz'
VENDOR = ROOT / 'vendor'
MANIFEST = json.loads((ROOT / 'provenance' / 'inputs.json').read_text())
FINOL = json.loads((ROOT / 'provenance' / 'finol_sources.json').read_text())


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def array_hash(values):
    packed = np.asarray(values, dtype='<f8').tobytes()
    return hashlib.sha256(packed).hexdigest()


def check(label, actual, expected):
    if actual != expected:
        raise ValueError(f'{label}: expected {expected}, found {actual}')


def git(path, *args):
    return subprocess.check_output(['git', '-C', str(path), *args],
                                   text=True).strip()


def checkout(name, local_source):
    info = MANIFEST[name]
    target = VENDOR / {'olps': 'OLPS', 'finol': 'finol_data'}[name]
    if local_source and not local_source.is_dir():
        raise ValueError(f'Checkout directory does not exist: {local_source}')
    if local_source and name == 'finol':
        target = local_source.resolve()

    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target.parent) as staging:
            temp = Path(staging) / 'checkout'
            subprocess.run(['git', 'init', '-q', str(temp)], check=True)
            git(temp, 'remote', 'add', 'origin', info['url'])
            if name == 'finol':
                paths = ['datasets/NYSE(N)/*.xlsx', 'datasets/SP500/*.xlsx']
                for item in info['datasets'].values():
                    panel = item['panel']
                    paths.append('benchmark_results/other/price_relative/'
                                 f'price_relative_{panel}.mat')

                git(temp, 'sparse-checkout', 'set', '--no-cone',
                    *['/' + p for p in paths])
                git(temp, 'config', 'remote.origin.promisor', 'true')
                git(temp, 'config', 'remote.origin.partialclonefilter',
                    'blob:none')

            source = str(local_source.resolve()) if local_source else 'origin'
            args = ['fetch', '--depth=1']
            if name == 'finol':
                args.append('--filter=blob:none')

            git(temp, *args, source, info['commit'])
            git(temp, 'checkout', '--detach', info['commit'])
            temp.rename(target)

    check(str(target), git(target, 'rev-parse', 'HEAD'), info['commit'])
    if git(target, 'status', '--porcelain', '--untracked-files=no'):
        raise ValueError(f'Pinned checkout contains local edits: {target}')
    return target


def classic(verify, source):
    if not verify:
        source = checkout('olps', source)

    for name, spec in MANIFEST['olps']['datasets'].items():
        dest = DATA / (name + '.mat')
        if not verify:
            original = source / 'Data' / dest.name
            check(dest.name, digest(original), spec['file_sha256'])
            shutil.copyfile(original, dest)

        check(dest.name, digest(dest), spec['file_sha256'])
        returns = classic_matrix(dest)
        check(name, list(returns.shape), spec['shape'])
        check(name, array_hash(returns), spec['returns_sha256'])
        print(f'{name}: {returns.shape}, verified')


def finol(verify, source):
    if not verify:
        source = checkout('finol', source)

    for rec in FINOL:
        name = rec['name']
        spec = MANIFEST['finol']['datasets'][name]
        panel = spec['panel']
        dest = DATA / (name + '.npz')
        test = DATA / f'price_relative_{panel}.mat'

        if not verify:
            original = source / 'benchmark_results/other/price_relative'
            check(test.name, digest(original / test.name),
                  spec['test_file_sha256'])
            shutil.copyfile(original / test.name, test)
            if not dest.exists():
                build_finol(source / 'datasets' / panel, rec, dest, test)

        check(test.name, digest(test), spec['test_file_sha256'])
        with np.load(dest, allow_pickle=False) as data:
            ret = data['returns']
            check(name, list(ret.shape), rec['shape'])
            check(name, array_hash(ret), rec['returns_sha256'])
            names = sorted(Path(f).stem for f in rec['files'])
            check(name, list(data['tickers']), names)
            dates = data['dates']
            check(name, list(dates.astype(str)[[0, -1]]),
                  [rec['start'], rec['end']])
            check(name, hashlib.sha256(dates.astype('S10').tobytes())
                  .hexdigest(), spec['dates_sha256'])
            released = loadmat(test)['data']
            np.testing.assert_array_equal(ret[-len(released):], released)
        print(f'{name}: {ret.shape}, verified against released test suffix')

    if not verify:
        shutil.copyfile(ROOT / 'provenance/finol_sources.json',
                        DATA / 'finol.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    datasets = ('olps', 'finol', 'yahoo')
    parser.add_argument('task', choices=(*datasets, 'all', 'verify'))
    parser.add_argument('--only', choices=datasets, nargs='+',
                        help='Limit the datasets checked by verify.')
    parser.add_argument('--olps-source', type=Path,
                        help='Pinned local OLPS Git checkout to fetch from.')
    parser.add_argument('--finol-source', type=Path,
                        help='Use a local pinned FinOL checkout directly.')
    parser.add_argument('--stock-snapshot', type=Path,
                        help='Original fixed 50-stock Yahoo NPZ.')
    parser.add_argument('--sector-snapshot', type=Path,
                        help='Original nine-sector adjusted-close CSV.')
    parser.add_argument('--allow-revised-yahoo', action='store_true',
                        help='Accept revised prices on the same calendar.')
    args = parser.parse_args()
    verify = args.task == 'verify'
    if args.only and not verify:
        parser.error('--only applies to verify')
    if verify and (args.stock_snapshot or args.sector_snapshot):
        parser.error('Use yahoo to import snapshots before running verify')

    names = args.only or datasets
    if not verify and args.task != 'all':
        names = (args.task,)

    if not verify:
        DATA.mkdir(parents=True, exist_ok=True)

    for name in names:
        if name == 'olps':
            classic(verify, args.olps_source)
        elif name == 'finol':
            finol(verify, args.finol_source)
        else:
            prepare_yahoo(MANIFEST['yahoo'], DATA, STOCK, args, verify)


if __name__ == '__main__':
    main()
