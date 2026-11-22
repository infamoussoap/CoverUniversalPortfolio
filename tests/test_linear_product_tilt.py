"""Check the one-factor recurrence against individually raised moments."""

from fractions import Fraction
from itertools import combinations_with_replacement

from flint import arb, ctx
import numpy as np

from particle_universal.linear_product_tilt import (linear_product_evidence,
                                                   weighted_pair_matrix)
from particle_universal.raised_pair_moments import RaisedSeries
from particle_universal.tilted_interval_cover import (TiltedCover,
                                                     exact_evidence)


def test_weighted_pairs():
    old_prec = ctx.prec
    ctx.prec = 128
    try:
        for alpha in (1, Fraction(1, 2), Fraction(5, 2)):
            for scores in ([0, 0, 0], [-1, 1, -1], [-1, 0, 3]):
                tilt = RaisedSeries([arb(v) for v in scores], alpha)
                keys = list(combinations_with_replacement(range(3), 3))
                individual = tilt.shifted_monomials(keys)

                for coeff in ([0, 1, 2], [0, 0, 3], [1, 1, 1]):
                    p = [arb(v) for v in coeff]
                    pairs = weighted_pair_matrix(tilt, p)
                    for i, j in combinations_with_replacement(range(3), 2):
                        value = sum(v * individual[tuple(sorted((k, i, j)))]
                                    for k, v in enumerate(p))
                        assert pairs[i, j].overlaps(value * tilt.shift.exp())
    finally:
        ctx.prec = old_prec


def test_zero_row_positions():
    before = np.array([1.25, 1., 0.75])
    loss = np.array([0., 1.5, 0.5])
    after = np.array([0.875, 1., 1.125])
    old_prec = ctx.prec
    ctx.prec = 128

    try:
        for alpha in (1, Fraction(1, 2)):
            for path in ([loss], [loss, before, after], [before, loss, after]):
                complement = TiltedCover(3, alpha, exact_pairs=True)
                for row in path:
                    if min(row) > 0:
                        complement.update(row, [1, 1, 1])

                low, high = linear_product_evidence(complement, loss)
                exact = list(exact_evidence(np.array(path), alpha))[-1]
                assert low <= exact.lower() <= exact.upper() <= high
    finally:
        ctx.prec = old_prec
