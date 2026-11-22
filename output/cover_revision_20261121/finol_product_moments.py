"""Positive directional Dirichlet series for several exact return rows.

For alpha=1 let D_i=1-h_i*z-sum_r p[r,i]*u_r and F=prod_i D_i**(-1).
The coefficient [z^n u^b] F, times prod_r b_r!/(d)_(n+|b|), is
E[(h'W)^n/n! prod_r(p_r'W)^b_r]. Extra factors W_i W_j replace F
by (1+1_{i=j}) F/(D_i D_j) and raise the denominator by two.
These identities follow by integrating independent Exp(1) random
variables in total-mass and simplex coordinates.
"""

from itertools import product
from math import factorial, prod

from flint import arb, arb_mat

from particle_universal.interval_cover import nonnegative_upper
from particle_universal.raised_pair_moments import RaisedSeries, inverse_rising
from particle_universal.tilted_interval_cover import (
    dot, exponential_remainder, homogeneous, powers, score_quadratic,
    score_square)


def indices(k, cap):
    counts = list(product(range(cap + 1), repeat=k))
    positions = {c: i for i, c in enumerate(counts)}
    edges = []
    for c in counts:
        prev = [(r, positions[c[:r] + (n - 1,) + c[r + 1:]])
                for r, n in enumerate(c) if n]
        edges.append(prev)

    return counts, edges


def add_factor(base, h, p, edges):
    """Divide the coefficient series by one D_i with positive recurrences."""
    out = []
    for b, prev in zip(base, edges):
        v = b.copy()
        for r, pos in prev:
            c = p[r]
            v = [x + c * y for x, y in zip(v, out[pos])]

        for j in range(1, len(v)):
            v[j] += h * v[j - 1]

        out.append(v)

    return out


def score_series(tilt, cols, edges, extra=0):
    n = tilt.degree + extra
    base = [[arb(0)] * (n + 1) for _ in edges]
    base[0][0] = arb(1)

    for hi, p in zip(tilt.h, cols):
        base = add_factor(base, hi, p, edges)

    return base


def gen_scale(d, rank):
    return arb(1) / prod(range(d, d + rank))


def doubled_masses(scores, cols, peaks):
    k, d = len(peaks), len(cols)
    counts, edges = indices(k, 2)
    tilt = RaisedSeries([2 * v for v in scores])
    base = score_series(tilt, cols, edges)
    values = []
    for cap in range(3):
        rank = k * cap
        idx = counts.index((cap,) * k)
        norm = inverse_rising(d + rank, tilt.degree)
        mult = factorial(cap)**k * gen_scale(d, rank)
        prefix = mult * dot(norm, base[idx])
        values.append(tilt.shift.exp() *
                      (prefix + prod(peaks)**cap * tilt.tail))

    return values


def product_pairs(scores, cols, peaks):
    k, d = len(peaks), len(cols)
    counts, edges = indices(k, 1)
    tilt = RaisedSeries(scores)
    N = tilt.degree
    base = score_series(tilt, cols, edges, extra=k + 1)
    norm = inverse_rising(d + k + 2, N)
    scale = tilt.shift.exp() * gen_scale(d, k + 2)
    tail = tilt.shift.exp() * prod(peaks) * gen_scale(d, 2) * tilt.tail

    # (z(h_i-h_j) + (p_i-p_j)'u) F/(D_i D_j) = F/D_i - F/D_j.
    # The finite coefficient at order n uses single-raise coefficients
    # only through n+k+1. Near equal scores use the positive recurrence.
    singles, diag = [], []
    for h, p in zip(tilt.h, cols):
        one = add_factor(base, h, p, edges)
        totals = []
        for row, c in zip(one, counts):
            offset = k - sum(c) + 1
            totals.append(dot(norm, row[offset:offset + N + 1]))

        singles.append(totals)
        two = add_factor(one, h, p, edges)
        diag.append(2 * (scale * dot(norm, two[-1][:N + 1]) + tail))

    pairs = arb_mat(d, d)
    for i in range(d):
        pairs[i, i] = diag[i]
        one = None
        for j in range(i):
            diff = tilt.h[i] - tilt.h[j]
            if abs(diff) > arb(1) / 20000:
                delta = [y - x for x, y in zip(cols[i], cols[j])]
                inverse = 1 / diff
                coefs = [inverse]
                for c, prev in zip(counts[1:], edges[1:]):
                    r, pos = prev[0]
                    coefs.append(coefs[pos] * delta[r] * inverse)

                total = sum(factorial(sum(c)) * v * (a - b)
                            for c, v, a, b in zip(counts, coefs,
                                                  reversed(singles[i]),
                                                  reversed(singles[j])))
                entry = scale * total + tail
            else:
                entry = None

            if entry is None or entry.rel_accuracy_bits() < 50:
                if one is None:
                    one = add_factor(base, tilt.h[i], cols[i], edges)
                two = add_factor(one, tilt.h[j], cols[j], edges)
                entry = scale * dot(norm, two[-1][:N + 1]) + tail

            pairs[i, j] = pairs[j, i] = entry

    # This scalar calculation does not use divided differences or pairs.
    inv = inverse_rising(d + k, N)
    scalar = dot(inv, base[-1][:N + 1]) * gen_scale(d, k)
    mass = tilt.shift.exp() * (scalar + prod(peaks) * tilt.tail)
    if not sum(pairs.entries()).overlaps(mass):
        raise ArithmeticError('Directional mass and pair-matrix sum disagree')

    return mass, pairs


def product_endpoints(mon, raw):
    weights = []
    for row in raw:
        original = [arb(float(v)) for v in row]
        norm = sum(original) / len(original)
        weights.append([v / norm for v in original])

    cols = list(zip(*weights)) if weights else [()] * mon.d
    peaks = [max(v.upper() for v in row) for row in weights]
    m, g, alpha = mon.rising, mon.g, mon.alpha
    assert alpha == 1 and all(v >= 0 for row in weights for v in row)
    mass, matrix = product_pairs(g, cols, peaks)
    e2, p_e2, pp_e2 = doubled_masses(g, cols, peaks)
    pairs = matrix.entries()
    q = dot(pairs, mon.Q.entries())
    abs_j = dot(pairs, mon.C.entries()).upper()
    gp = powers(g, 4)
    h = homogeneous([len(g), 0] + [sum(v) for v in gp[2:]], 4)
    peak = max(v.upper() for v in g)

    # P may differ from one. Bound E[e^G (P-1) J] by Cauchy-Schwarz.
    moments = score_quadratic(mon.C, gp, h, m, alpha)
    cs = nonnegative_upper(2 * moments[2]) * nonnegative_upper(24 * moments[4])
    error = exponential_remainder(peak, 3) * cs.sqrt()
    j2 = score_square(mon.C, g, m, alpha)[0]
    error += nonnegative_upper((pp_e2 - 2 * p_e2 + e2) * j2).sqrt()
    cubic = mon.cubic_moments(gp)
    center = cubic[0] + cubic[1] + cubic[2] / 2
    j_lo = max(-abs_j, (center - error).lower())
    j_hi = min(abs_j, (center + error).upper())

    k_hi = min((prod(peaks) * peak.exp() * mon.R4).upper(),
               dot(pairs, mon.K.entries()).upper())
    second, fifth = score_square(mon.H, g, m, alpha)
    g4 = nonnegative_upper(24 * h[4] / m[4])
    mixed = g4.sqrt() * mon.H4_root**2
    taylor = prod(peaks) * (second + fifth
                            + exponential_remainder(peak, 2) * mixed)
    b_hi = min(taylor.upper(), (pp_e2.sqrt() * mon.H4_root**2).upper())

    a_lo = max(arb(0), (q - j_hi / 3).lower())
    a_hi = (q - j_lo / 3 + k_hi).upper()
    lo = mass.lower().log() - a_hi / mass.lower()
    hi = min(mass.upper(), (mass.upper() - a_lo + b_hi / 2).upper())
    assert mass > 0 and hi > 0
    return lo.lower(), hi.log().upper()


def validate_moments():
    """Cross-check all five product degrees with direct raised monomials."""
    g = [arb(1)/4, arb(1)/4, -arb(1)/2]
    rows = [[arb((j + i) % 3) + arb(1)/2 for j in range(3)]
            for i in range(5)]
    for k in range(1, 6):
        cols = list(zip(*rows[:k]))
        peaks = [max(row) for row in rows[:k]]
        mass, pairs = product_pairs(g, cols, peaks)
        tilt = RaisedSeries(g)
        poly = {(): arb(1)}
        for row in rows[:k]:
            new = {}
            for key, value in poly.items():
                for i, x in enumerate(row):
                    term = tuple(sorted(key + (i,)))
                    new[term] = new.get(term, 0) + value * x

            poly = new

        square = {}
        for key, coef in poly.items():
            for second, value in poly.items():
                term = tuple(sorted(key + second))
                square[term] = square.get(term, 0) + coef * value

        twice = RaisedSeries([2 * v for v in g])
        doubled = doubled_masses(g, cols, peaks)
        for result, terms in zip(doubled, [{(): arb(1)}, poly, square]):
            moment = twice.shifted_monomials(set(terms))
            direct = sum(v * moment[key] for key, v in terms.items())
            assert result.overlaps(direct * twice.shift.exp())
            assert result.rel_accuracy_bits() >= 50

        for i in range(3):
            for j in range(i + 1):
                terms = [(tuple(sorted(key + (i, j))), v)
                         for key, v in poly.items()]
                part = tilt.shifted_monomials({key for key, _ in terms})
                direct = sum(v * part[key] for key, v in terms)
                assert pairs[i, j].overlaps(direct * tilt.shift.exp())
                assert pairs[i, j].rel_accuracy_bits() >= 50


if __name__ == '__main__':
    validate_moments()
