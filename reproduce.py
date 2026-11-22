"""Run the paper's experiments sequentially, starting from prepared inputs."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
BENCH = 'output/cover_revision_20261121/run_benchmarks.py'
INFERENCE = 'output/experiment_5a/run.py'
APPENDIX = ('scripts/alternating.py', '--out',
            'output/appendix/alternating.json')
VERIFY = ('scripts/check_results.py',)
STAGES = {
    'smoke': [(INFERENCE, 'smoke'), ('scripts/alternating.py', '--rounds',
                                     '160')],
    'benchmark': [(BENCH, 'all')],
    'inference': [(INFERENCE, 'all')],
    'appendix': [APPENDIX],
    'verify': [VERIFY],
    'all': [('scripts/prepare_data.py', 'verify'), (BENCH, 'all'),
            (INFERENCE, 'all'), APPENDIX, VERIFY],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=STAGES)
    stage = parser.parse_args().stage
    env = os.environ.copy()
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS',
                'NUMBA_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[key] = '1'

    paths = [str(ROOT), env.get('PYTHONPATH', '')]
    env['PYTHONPATH'] = os.pathsep.join(p for p in paths if p)
    env['MPLBACKEND'] = 'Agg'
    for command in STAGES[stage]:
        print('+', sys.executable, *command, flush=True)
        subprocess.run([sys.executable, *command], cwd=ROOT, env=env,
                       check=True)


if __name__ == '__main__':
    main()
