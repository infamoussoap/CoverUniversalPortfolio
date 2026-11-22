"""Check the daily threshold independently of the subset screening rule."""

import numpy as np
import pandas as pd
from flint import ctx

from market import CACHE, ROOT, load_panel
from threshold_audit import certify


def audit():
    ctx.prec, ctx.threads = 128, 1
    design = pd.read_csv(ROOT / '5a_subset_design.csv')
    panel = load_panel('full')
    tickers = list(panel['tickers'])
    series = pd.read_csv(ROOT / '5a_paths.csv', float_precision='round_trip')
    full_path = series.loc[series.market == 'Yahoo 50', 'z_hi']
    full_cap = full_path.max()
    full_pending = int((full_path >= np.nextafter(np.log(20.), -np.inf)).sum())
    rows = []

    for item in design.itertuples():
        if item.d == 50:
            upper, pending = full_cap, full_pending
        else:
            cols = [tickers.index(v) for v in item.tickers.split()]
            raw = np.ascontiguousarray(panel['returns'][:, cols])
            upper, pending = certify(raw)

        rows.append((item.d, item.subset_id, upper, pending))

    names = ['d', 'subset_id', 'all_days_log_upper', 'unresolved_by_upper']
    target = CACHE / 'subset_audit.csv'
    pd.DataFrame(rows, columns=names).to_csv(target, index=False)


if __name__ == '__main__':
    audit()
