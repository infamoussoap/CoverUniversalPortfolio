"""Reproduce the five-stock alternating market in Appendix C.2."""

import argparse
import json
from fractions import Fraction
from math import comb
from pathlib import Path

import numpy as np
from flint import arb, ctx, fmpq
from scipy.special import logsumexp, roots_jacobi

from particle_universal.linear_cover import linear_cover
from particle_universal.tilted_interval_cover import TiltedCover


def quadrature(steps, edge, order):
    """Integrate (1 - edge² U²)^m under U ~ Beta(1,4) at each prefix."""
    nodes, weights = roots_jacobi(order, 3, 0)
    u = (1 + nodes) / 2
    log_mass = np.log(weights / weights.sum())
    log_pair = np.log1p(-(edge * u)**2)
    return logsumexp(log_mass + np.outer(steps // 2, log_pair), axis=1)


def exact_mass(pairs, edge):
    # E[U^(2j)] = 1 / binom(2j+4, 4) for d=5.
    return sum(comb(pairs, j) * (-edge**2)**j / comb(2 * j + 4, 4)
               for j in range(pairs + 1))


def run(rounds):
    if rounds < 1:
        raise ValueError("The number of sessions must be positive")

    ctx.prec, ctx.threads = 128, 1
    returns = np.ones((rounds, 5))
    returns[::2, :2] += [0.2, -0.2]
    returns[1::2, :2] += [-0.2, 0.2]
    edge = Fraction(float(returns[0, 0])) - 1
    assert Fraction(float(returns[0, 1])) == 1 - edge

    steps = np.arange(1, rounds + 1)
    short = quadrature(steps, float(edge), 96)
    long = quadrature(steps, float(edge), 192)
    reference_error = float(np.max(np.abs(short - long)))

    monitor = TiltedCover(5)
    weights = linear_cover(returns)
    threshold = arb(1) / 100
    first_open = first_breach = first_certified_breach = None

    for day, (row, w) in enumerate(zip(returns, weights), 1):
        monitor.update(row, w)
        lo, hi = monitor.evidence()
        gap_lo = (lo - monitor.log_played).lower()
        gap_hi = (hi - monitor.log_played).upper()
        if first_open is None and not (-threshold <= gap_lo <= gap_hi
                                       <= threshold):
            first_open = day

        if first_certified_breach is None and (gap_lo > threshold
                                               or gap_hi < -threshold):
            first_certified_breach = day

        if day <= 160:
            exact = exact_mass(day // 2, edge)
            log_z = arb(fmpq(exact.numerator, exact.denominator)).log()
            if not (lo <= log_z.lower() <= log_z.upper() <= hi):
                raise ArithmeticError(f"Exact evidence escaped on day {day}")

            gap = abs(log_z - monitor.log_played)
            if first_breach is None and gap > threshold:
                first_breach = day

    return {"rounds": rounds, "rational_checks": min(rounds, 160),
            "first_inconclusive_day": first_open,
            "first_exact_breach_day": first_breach,
            "first_certified_breach_day": first_certified_breach,
            "max_quadrature_difference": reference_error}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=12600)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = run(args.rounds)
    report = json.dumps(result, indent=2) + "\n"
    print(report, end="")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)
