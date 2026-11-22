"""Reproduce the stock examples and six-market benchmark sequentially."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
OUT = HERE.parent / 'cover_experiments_final'
STAGES = ('examples', 'markets', 'matlab', 'tables', 'scaling', 'sensitivity')


def configure():
    sys.path.insert(0, str(PROJECT))
    env = os.environ
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS',
                'NUMBA_NUM_THREADS'):
        env[key] = '1'

    paths = [str(PROJECT), env.get('PYTHONPATH', '')]
    env['PYTHONPATH'] = os.pathsep.join(p for p in paths if p)
    (HERE / 'cache').mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)


def examples():
    import examples as stock
    from extra_checks import exact_pair
    from pair_figure import draw
    from reference import reference
    from study import run
    from verify_exports import verify_pair

    run('pair', 1, 1)
    run('stocks', 1, 3)
    reference('stocks', 1)
    exact_pair()
    stock.decompose()
    stock.figure()

    for name in ('table1_width.csv', 'fig2_50stock.csv', 'fig2_50stock.pdf'):
        shutil.copyfile(HERE / name, OUT / name)

    verify_pair()
    draw()


def markets():
    from bench_report import report
    from extra_checks import regret
    from reference import reference
    from strategies import run as allocations
    from study import CLASSIC, preflight, run

    kinds = ('first_order', 'UCRP', 'Soft-Bayes', 'VB-FTRL', 'EG', 'ONS',
             'PAMR', 'OLMAR1', 'sample_100', 'sample_1000', 'sample_10000',
             'sample_100000')
    preflight()
    for name in CLASSIC:
        for alpha in (1, 0.5):
            run(name, alpha, 3)
            reference(name, alpha)

        allocations(name, kinds, 3)

    report()
    regret()
    for name in ('bench_evidence', 'bench_datasets', 'bench_extremes',
                 'tableB'):
        shutil.copyfile(HERE / (name + '.csv'), OUT / (name + '.csv'))


def matlab():
    from scipy.io import savemat
    from study import CACHE, CLASSIC, tape

    olps = Path(os.environ.get('PAPER_OLPS_ROOT', PROJECT / 'vendor/OLPS'))
    executable = os.environ.get('MATLAB_BIN', 'matlab')
    if not (olps / 'Strategy/up_run.m').is_file():
        raise FileNotFoundError(f'Pinned OLPS checkout is missing at {olps}')

    provenance = json.loads((PROJECT / 'provenance/inputs.json').read_text())
    pin = provenance['olps']['commit']
    git = ['git', '-C', str(olps)]
    version = subprocess.check_output([*git, 'rev-parse', 'HEAD'], text=True)
    if version.strip() != pin:
        raise ValueError(f'OLPS checkout must be at {pin}.')
    subprocess.run([*git, 'diff', '--quiet', 'HEAD', '--', 'Strategy'],
                   check=True)

    if not shutil.which(executable):
        raise FileNotFoundError('Set MATLAB_BIN to the MATLAB executable.')

    for name in CLASSIC:
        savemat(CACHE / f'{name}_native_input.mat', {'x': tape(name)})

    env = dict(os.environ, COVER_STUDY=str(HERE),
               COVER_OLPS=str(olps.resolve()))
    names = '{' + ','.join("'" + n + "'" for n in CLASSIC) + '}'
    cmd = ("addpath(getenv('COVER_STUDY')); "
           "matlab_olps_replay(getenv('COVER_OLPS'), getenv('COVER_STUDY'), "
           + names + ')')
    subprocess.run([executable, '-singleCompThread', '-batch', cmd],
                   check=True, env=env)


def tables():
    import matplotlib.pyplot as plt

    from assemble_market import assemble_benchmarks, published_audit
    from report_tables import table_c, table_d, wealth_plot
    from verify_exports import verify_gaps

    plt.rcParams['text.usetex'] = True
    frame = assemble_benchmarks()
    audit = published_audit(frame)
    table_c(frame)
    table_d(audit)
    wealth_plot()
    verify_gaps()
    print('Verified market wealth and all 306 strategy histories.')


def scaling():
    import matplotlib.pyplot as plt

    from cost import coordinate
    from final_cost import save_costs
    from study import CACHE, CLASSIC, tape
    from verify_exports import subset_tapes

    coordinate()
    for name in CLASSIC:
        horizon, d = tape(name).shape
        for alpha in (1, 0.5):
            suffix = '' if alpha == 1 else '_alpha0.5'
            for rep in range(3):
                key = f'cost_{name}_{d}_{horizon}_pair_{rep}{suffix}.json'
                if (CACHE / key).is_file():
                    continue

                args = ['--name', name, '--d', str(d), '--ends', str(horizon),
                        '--alpha', str(alpha), '--rep', str(rep)]
                subprocess.run([sys.executable, HERE / 'cost.py', *args],
                               check=True)

    plt.rcParams['text.usetex'] = True
    save_costs()
    subset_tapes()


def sensitivity():
    from finol_exact_sets import run

    subprocess.run([sys.executable, HERE / 'extra_checks.py', 'product'],
                   check=True)
    run()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=(*STAGES, 'all'))
    args = parser.parse_args()
    configure()

    from flint import ctx
    from threadpoolctl import threadpool_limits
    from study import CLASSIC, check_inputs

    ctx.prec, ctx.threads = 128, 1
    actions = dict(zip(STAGES, (examples, markets, matlab, tables,
                                scaling, sensitivity)))
    inputs = dict(examples=('stocks', 'pair'), markets=CLASSIC,
                  matlab=CLASSIC, tables=CLASSIC,
                  scaling=(*CLASSIC, 'finol_nyse', 'finol_sp'),
                  sensitivity=(*CLASSIC, 'finol_nyse', 'finol_sp'))
    with threadpool_limits(1):
        for stage in STAGES if args.stage == 'all' else (args.stage,):
            print(f'Running {stage}', flush=True)
            check_inputs(inputs[stage])
            actions[stage]()


if __name__ == '__main__':
    main()
