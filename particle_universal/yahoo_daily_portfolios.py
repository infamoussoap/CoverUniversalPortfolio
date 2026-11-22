"""Load a date range of the frozen Yahoo stock universe."""

from hashlib import sha256
import numpy as np
from .paths import RESULTS


PANELS = {'2000': ('2000', '2010'), '2010': ('2010', '2020'),
          'full': ('2000', '2026')}


def load_panel(name):
    source = RESULTS / 'yahoo_fixed50.npz'
    start, end = PANELS[name]

    with np.load(source, allow_pickle=False) as data:
        dates = data['dates']
        selection = (dates >= start) & (dates < end)
        panel = {key: data[key][selection] for key in ('returns', 'dates')}
        panel['tickers'] = data['tickers']

    panel['source_sha256'] = sha256(source.read_bytes()).hexdigest()
    return panel
