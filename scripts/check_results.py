"""Compare independently regenerated summaries with the paper's records."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = ROOT / 'expected'
BENCH = ROOT / 'output' / 'cover_experiments_final'
INFERENCE = ROOT / 'output' / 'experiment_5a'
CASES = {
    'examples': (('table1_width', 'path'),),
    'markets': (('tableB', 'dataset alpha'), ('certified_regret', 'dataset')),
    'tables': (('bench_summary', 'dataset alpha strategy seed'),
               ('tableD', 'dataset')),
    'scaling': (('bench_cost', 'dataset alpha d T method'),),
    'sensitivity': (('finol_nyse_exact_sets', 'dataset S_size'),
                    ('finol_nyse_empty_set_control', 'dataset S_size method')),
    'inference': (('5a_summary', 'market'), ('5a_null_summary', 'market'),
                  ('5a_power_summary', 'g horizon'),
                  ('5a_subset_summary', 'group d')),
}
FIGURES = {'examples': ('aapl_msft_pair_series.pdf', 'fig2_50stock.pdf'),
           'tables': ('bench_wealth.pdf',), 'scaling': ('bench_scaling.pdf',),
           'inference': ('figure_A.pdf', 'figure_C.pdf')}


def check_table(folder, name, keys):
    filename = name + '.csv'
    columns = keys.split()
    expected = pd.read_csv(EXPECTED / filename).set_index(columns)
    actual = pd.read_csv(folder / filename).set_index(columns)
    assert expected.index.is_unique and actual.index.is_unique, filename
    pd.testing.assert_index_equal(expected.sort_index().index,
                                  actual.sort_index().index, exact=False)
    actual = actual.reindex(expected.index)

    for col in expected:
        if col in ('seconds', 'peak_mem_mb'):
            values = actual[col]
            if not np.isfinite(values).all() or not (values > 0).all():
                raise ValueError(f'{filename}: invalid measured {col}')
        elif pd.api.types.is_numeric_dtype(expected[col]):
            tolerant = pd.api.types.is_float_dtype(expected[col])
            rtol, atol = (5e-6, 5e-10) if tolerant else (0, 0)
            np.testing.assert_allclose(actual[col], expected[col],
                                       rtol=rtol, atol=atol, equal_nan=True,
                                       err_msg=f'{filename}: {col}')
        else:
            pd.testing.assert_series_equal(actual[col], expected[col],
                                           check_dtype=False)

    print(f'Passed {filename} ({len(actual)} records)')


def check_alternating():
    filename = ROOT / 'output' / 'appendix' / 'alternating.json'
    data = json.loads(filename.read_text())
    for key, expected in [('rounds', 12600), ('rational_checks', 160),
                          ('first_inconclusive_day', 56),
                          ('first_exact_breach_day', 78),
                          ('first_certified_breach_day', None)]:
        if data[key] != expected:
            raise ValueError(f'{key}: expected {expected}, found {data[key]}')

    if data['max_quadrature_difference'] > 3e-13:
        raise ArithmeticError('Alternating-market quadrature disagreement')
    print('Passed alternating market (12,600 closes)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', nargs='?', default='all',
                        choices=('all', 'appendix', *CASES))
    stage = parser.parse_args().stage
    stages = [*CASES, 'appendix'] if stage == 'all' else [stage]
    for item in stages:
        if item == 'appendix':
            check_alternating()
            continue

        folder = INFERENCE if item == 'inference' else BENCH
        for name, keys in CASES[item]:
            check_table(folder, name, keys)

        for name in FIGURES.get(item, ()):
            with (folder / name).open('rb') as pdf:
                if pdf.read(5) != b'%PDF-':
                    raise ValueError(f'{name} is not a PDF')


if __name__ == '__main__':
    main()
