"""Certify wealth when one return row is retained as a linear factor."""

from math import factorial

from flint import arb, arb_mat

from .interval_cover import nonnegative_upper
from .raised_pair_moments import RaisedSeries, inverse_rising, raise_series
from .tilted_interval_cover import (dot, exponential_remainder, homogeneous,
                                    powers, score_quadratic, score_square)


def weighted_pair_matrix(tilt, p):
    """Enclose E[exp(G) (p'W) WW'] by nonnegative Dirichlet series.

    Parameters
    ----------
    tilt : RaisedSeries
        Scores and truncated base series for a symmetric Dirichlet prior.
    p : list of arb, length d
        Nonnegative exact-row coefficients.

    Returns
    -------
    pairs : (d, d) arb_mat
        Matrix of weighted second moments.
    """
    d, alpha = tilt.d, tilt.alpha
    total = d * alpha
    if len(p) != d or any(not v >= 0 for v in p) or not max(p) > 0:
        raise ValueError('The factor requires nonnegative coefficients')

    degree = tilt.degree
    first = [raise_series(tilt.base, x) for x in tilt.h]
    mix = [alpha * sum(c * u[n] for c, u in zip(p, first))
           for n in range(degree + 1)]
    mass = alpha * sum(p)
    inv = inverse_rising(total + 3, degree)
    scale = tilt.shift.exp() * alpha / (total * (total + 1) * (total + 2))

    pairs = arb_mat(d, d)
    for i, hi in enumerate(tilt.h):
        base = [u + p[i] * v for u, v in zip(mix, first[i])]
        for j in range(i + 1):
            coeff = [u + p[j] * v for u, v in zip(base, first[j])]
            series = raise_series(raise_series(coeff, hi), tilt.h[j])
            tail = (mass + p[i] + p[j]) * tilt.tail
            count = alpha + int(i == j)
            value = scale * count * (dot(series, inv) + tail)
            pairs[i, j] = pairs[j, i] = value

    return pairs


def linear_product_evidence(monitor, row):
    """Enclose normalized Cover log wealth with one exact return row.

    Parameters
    ----------
    monitor : TiltedCover
        Streaming statistics for the strictly positive complement.
    row : (d,) np.ndarray
        Gross returns on the row kept exact, in binary64.

    Returns
    -------
    bounds : tuple of arb
        Lower and upper endpoints for normalized log evidence.
    """
    x = [arb(float(entry)) for entry in row]
    if len(x) != monitor.d or not all(v.is_finite() for v in x):
        raise ValueError('Expected a finite price relative for each stock')
    if min(x) < 0 or not max(x) > 0:
        raise ValueError('The exact row needs nonnegative, nonzero returns')

    normalizer = sum(x) / len(x)
    p = [v / normalizer for v in x]
    peak_p = max(v.upper() for v in p)
    g, alpha, m = monitor.g, monitor.alpha, monitor.rising
    peak = max(v.upper() for v in g)
    gp = powers(g, 4)
    h = homogeneous([len(g), 0] + [sum(v) for v in gp[2:]], 4, alpha)
    pairs = weighted_pair_matrix(RaisedSeries(g, alpha), p).entries()
    mass = sum(pairs)
    if not mass > 0:
        raise ArithmeticError('Could not enclose a positive tilted mass')

    q = dot(pairs, monitor.Q.entries())
    abs_j = dot(pairs, monitor.C.entries()).upper()
    cubic = monitor.cubic_moments(gp)
    center = cubic[0] + cubic[1] + cubic[2] / 2
    moments = score_quadratic(monitor.C, gp, h, m, alpha)
    cs = nonnegative_upper(2 * moments[2]) * nonnegative_upper(24 * moments[4])
    deviation = exponential_remainder(peak, 3) * cs.sqrt()

    # Bound the change from the plain cubic moment under the linear factor.
    centered = arb_mat(len(p), 1, [v - 1 for v in p])
    double = RaisedSeries([2 * v for v in g], alpha).pair_matrix()
    score_var = (centered.transpose() * double * centered)[0, 0]
    j_square = score_square(monitor.C, g, m, alpha)[0]
    deviation += nonnegative_upper(score_var * j_square).sqrt()
    j_lo = max(-abs_j, (center - deviation).lower())
    j_hi = min(abs_j, (center + deviation).upper())

    k_hi = (peak_p * peak.exp() * monitor.R4).upper()
    if monitor.K is not None:
        k_hi = min(k_hi, dot(pairs, monitor.K.entries()).upper())

    second, fifth = score_square(monitor.H, g, m, alpha)
    g4 = nonnegative_upper(factorial(4) * h[4] / m[4])
    mixed = g4.sqrt() * monitor.H4_root**2
    taylor = peak_p * (second + fifth + exponential_remainder(peak, 2) * mixed)
    coeff = arb_mat(len(p), 1, p)
    product_square = (coeff.transpose() * double * coeff)[0, 0]
    cs = nonnegative_upper(product_square).sqrt() * monitor.H4_root**2
    b_hi = min(cs.upper(), taylor.upper())

    a_low = max(arb(0), (q - j_hi / 3).lower())
    a_high = (q - j_lo / 3 + k_hi).upper()
    lo = mass.lower().log() - a_high / mass.lower()
    hi = min(mass.upper(), (mass.upper() - a_low + b_hi / 2).upper())
    if not hi > 0:
        raise ArithmeticError('Could not verify the upper endpoint')

    return lo.lower(), hi.log().upper()
