"""Replay the paper's anytime-valid tests, permutation studies and subsets."""

import argparse
import json
import os
import subprocess
import sys
from hashlib import sha256
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BENCH = HERE.parent / 'cover_revision_20261121'
STAGES = {
    'markets': [('market.py', '--preflight'), ('market.py', '--recompute')],
    'exact': [('arb_checks.py', 'pair')],
    'null': [('arb_checks.py', 'sectors'), ('arb_checks.py', 'djia'),
             ('replays.py', 'null')],
    'power': [('replays.py', 'power'), ('replays.py', 'extended')],
    'audit': [('threshold_audit.py',)],
    'summaries': [('report.py', '--summaries')],
    'subsets': [('subset_design.py',), ('subsets.py', '0', '1'),
                ('subset_audit.py',), ('subset_report.py', '--summaries')],
    'figures': [('report.py', '--figures'), ('subset_report.py', '--figures')],
}


def prepare_cache():
    """Identify the inputs before resuming any existing replication chunks."""
    data = BENCH / 'data'
    names = ['djia.mat', 'msci.mat', 'nyse-o.mat', 'nyse-n.mat',
             'sp500.mat', 'tse.mat', 'sectors.csv']
    sources = [data / name for name in names]
    sources.append(ROOT / 'particle_universal/results/yahoo_fixed50.npz')
    missing = [p.name for p in sources if not p.is_file()]
    if missing:
        raise SystemExit('Missing market inputs: ' + ', '.join(missing)
                         + '. See docs/inference.md for data preparation.')

    digests = {p.name: sha256(p.read_bytes()).hexdigest() for p in sources}
    # Evidence endpoints, candidate days and verdicts depend on the monitor.
    version = {'inputs': digests, 'certificate': 'cubic-pair-abs-v1'}
    signature = json.dumps(version, sort_keys=True)
    run_id = sha256(signature.encode()).hexdigest()
    cache = HERE / 'cache' / run_id
    cache.mkdir(parents=True, exist_ok=True)
    (cache / 'inputs.json').write_text(json.dumps(digests, indent=2) + '\n')
    return cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choices = ('all', 'smoke', *STAGES)
    parser.add_argument('stage', nargs='?', choices=choices, default='all')
    stage = parser.parse_args().stage

    env = os.environ.copy()
    for name in ('OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
                 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS',
                 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[name] = '1'

    paths = [str(ROOT), str(BENCH)]
    if env.get('PYTHONPATH'):
        paths.append(env['PYTHONPATH'])
    env['PYTHONPATH'] = os.pathsep.join(paths)

    if stage == 'smoke':
        commands = [('smoke.py',)]
    else:
        if stage != 'figures':
            env['COVER_INFERENCE_CACHE'] = str(prepare_cache())

        steps = STAGES if stage == 'all' else [stage]
        commands = [cmd for step in steps for cmd in STAGES[step]]

    for script, *args in commands:
        print('Running', script, *args, flush=True)
        subprocess.run([sys.executable, str(HERE / script), *args],
                       cwd=ROOT, env=env, check=True)


if __name__ == '__main__':
    main()
