"""Freeze the Yahoo asset selection and the descriptive winner groups."""

from hashlib import sha256
from itertools import combinations

import numpy as np
import pandas as pd

from market import ROOT, load_panel


SEED = 583214


def freeze():
    panel = load_panel('full')
    raw = panel['returns']
    tickers = np.asarray(panel['tickers']).astype(str)
    scores = np.log(raw).sum(axis=0)
    ranking = np.lexsort((tickers, -scores))
    top = tickers[ranking[:5]]
    path = ROOT / '5a_stock_ranking.csv'
    frame = pd.DataFrame(dict(rank=np.arange(1, 51), ticker=tickers[ranking],
                              buy_hold_log_return=scores[ranking],
                              buy_hold_multiple=np.exp(scores[ranking])))
    frame.to_csv(path, index=False)

    rows = []
    for d in (2, 3, 5, 10, 20, 50):
        if d in (2, 50):
            draws = list(combinations(range(50), d))
        else:
            rng = np.random.default_rng([SEED, d])
            draws = {}
            while len(draws) < 200:
                key = tuple(sorted(rng.choice(50, d, replace=False)))
                draws[key] = None
        for i, cols in enumerate(draws, 1):
            names = tickers[list(cols)]
            rows.append((d, i, ' '.join(names), bool(set(names) & set(top))))

    dest = ROOT / '5a_subset_design.csv'
    pd.DataFrame(rows, columns=['d', 'subset_id', 'tickers', 'has_top5']
                 ).to_csv(dest, index=False)
    print('Ranking:', ' '.join(top), '; manifest:', sha256(
        dest.read_bytes()).hexdigest())


if __name__ == '__main__':
    freeze()
