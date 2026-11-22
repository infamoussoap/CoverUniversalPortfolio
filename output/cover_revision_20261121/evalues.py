"""Positive Bernstein integration on the two-asset simplex."""

import numpy as np
from flint import arb

from particle_universal.interval_cover import outward_float


def positive_pair(x):
    q = [arb(1)]
    normal = arb(0)
    out = []
    for t, row in enumerate(x):
        a, b = map(arb, row)
        normal += ((a+b)/2).log()
        aa, bb = a/(t+2), b/(t+2)
        left = [p*(i+1)*aa for i, p in enumerate(q)]
        right = [p*(t+1-i)*bb for i, p in enumerate(q)]
        q = [right[0], *(i+j for i, j in zip(left, right[1:])), left[-1]]
        z = sum(q).log()-normal
        out.append((outward_float(z.lower(), False),
                    outward_float(z.upper(), True)))
    return np.array(out)
