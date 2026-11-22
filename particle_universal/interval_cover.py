"""Directed log-wealth, Dirichlet mass and curvature bounds."""

from fractions import Fraction
from math import inf, nextafter
import numpy as np
from flint import arb, arb_mat, arb_poly, arb_series, ctx, fmpq


def exact_alpha(value):
    """ Interpret an integer, rational, or binary64 concentration exactly. """
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("Expected a positive Dirichlet concentration")
    if isinstance(value, (int, np.integer)):
        value = int(value)
    if isinstance(value, (float, np.float64)):
        value = float(value)
    if not isinstance(value, (int, float, Fraction, str)):
        raise ValueError("Alpha must be an exact rational or binary64 value")

    try:
        alpha = Fraction(value)
    except (ValueError, OverflowError, ZeroDivisionError) as error:
        raise ValueError("Invalid Dirichlet concentration") from error

    if alpha <= 0:
        raise ValueError("Expected a positive Dirichlet concentration")

    return alpha


def alpha_parameter(value):
    if isinstance(value, arb):
        if not value.is_finite() or not value > 0:
            raise ValueError("Expected a positive Dirichlet concentration")

        return value

    alpha = exact_alpha(value)
    return 1 if alpha == 1 else arb(fmpq(alpha.numerator, alpha.denominator))


def dirichlet_coefficients(shifted, alpha, degree):
    power = shifted
    sums = [arb(0)]
    for _ in range(degree):
        sums.append(alpha * sum(power))
        power = [left * right for left, right in zip(power, shifted)]

    coefficients = [arb(1)]
    for k in range(1, degree + 1):
        value = sum(sums[j] * coefficients[k - j] for j in range(1, k + 1))
        coefficients.append(value / k)

    return coefficients


def moment_bounds(g, alpha=1):
    shift = min(value.lower() for value in g)
    shifted = [value - shift for value in g]
    peak = max(value.upper() for value in shifted)
    degree = max(24, int(4 * float(peak)) + 1)
    if alpha == 1:
        poly = arb_poly([1])
        for value in shifted:
            poly *= arb_poly([1, -value])

        cap = ctx.cap
        try:
            ctx.cap = degree + 1
            series = arb_series(poly, prec=degree + 1).inv()
        finally:
            ctx.cap = cap
    else:
        alpha = alpha_parameter(alpha)
        series = dirichlet_coefficients(shifted, alpha, degree)

    concentration = len(g) * alpha
    rising = 1
    terms = [arb(1)]
    for k in range(1, degree + 1):
        rising *= concentration + k - 1
        terms.append(series[k] / rising)

    bounds = []
    for order in (1, 2):
        ratio = peak * order / (degree + 1)
        if not ratio < 1:
            raise ArithmeticError("Dirichlet series tail ratio is not < 1")

        scaled = [term * (order**k) for k, term in enumerate(terms)]
        multiplier = (order * shift).exp()
        prefix = multiplier * sum(scaled)
        tail = multiplier * scaled[-1] * ratio / (1 - ratio)
        bounds.append((prefix.lower(), (prefix + tail).upper()))

    return bounds


def quadratic_moments(B, diagonal, alpha=1):
    c = len(diagonal) * alpha
    trace = sum(diagonal)
    diagonal_square = sum(value * value for value in diagonal)
    square = sum(value * value for value in B.entries())
    if alpha == 1:
        first = trace / (c * (c + 1))
        fourth = trace * trace + 2 * square + 6 * diagonal_square
    else:
        first = alpha * trace / (c * (c + 1))
        fourth = alpha**2 * (trace * trace + 2 * square)
        fourth += 6 * alpha * diagonal_square

    second = fourth / (c * (c + 1) * (c + 2) * (c + 3))
    return first, second


def nonnegative_upper(x):
    return max(arb(0), x.upper())


def log_evidence_bounds(g, B, H, zero_floor=None, alpha=1):
    d = len(g)
    if alpha != 1:
        alpha = alpha_parameter(alpha)

    c = d * alpha
    (m1_low, m1_high), (_, m2_high) = moment_bounds(g, alpha)
    diagonal = [B[i, i] for i in range(d)]
    first, second = quadratic_moments(B, diagonal, alpha)
    upper = m1_high
    if first > 0:
        cross = 2 * alpha * sum(x * y for x, y in zip(g, diagonal))
        cross /= c * (c + 1) * (c + 2)
        a = (first * (cross / first).exp()).lower()
        b = (second * max(value.upper() for value in g).exp()).upper()
        if a > 0 and b >= 0:
            upper = min(upper, (m1_high - a * a / (a + b)).upper())

    log_upper = upper.log().upper()
    log_lower = None
    if zero_floor is not None:
        mean = alpha * sum(H[i, i] for i in range(d)) / (c * (c + 1))
        log_lower = (zero_floor - mean).lower()
    elif m1_low > 0:
        diagonal = [H[i, i] for i in range(d)]
        mean, second = quadratic_moments(H, diagonal, alpha)
        var_H = nonnegative_upper(second - mean * mean)
        var_exp = nonnegative_upper(m2_high - m1_low * m1_low)
        tilted = mean.upper() + (var_H * var_exp).sqrt() / m1_low
        log_lower = (m1_low.log() - tilted).lower()

    return log_lower, log_upper


def played_log_gain(row, weights, normalizer):
    played = [arb(float(value)) for value in weights]
    if len(played) != len(row) or min(played) < 0 or not max(played) > 0:
        raise ValueError("Need nonnegative weights and positive mass")

    gain = sum(x * w for x, w in zip(row, played)) / sum(played)
    if not gain > 0:
        raise ArithmeticError("Played portfolio has no positive payoff")

    return (gain / normalizer).log()


def gap_to_wealth(bounds, log_wealth):
    lower, upper = bounds
    if lower is not None:
        lower = (lower - log_wealth).lower()

    upper = (upper - log_wealth).upper()
    if not upper.is_finite() or (lower is not None and
                                (not lower.is_finite() or lower > upper)):
        raise ArithmeticError("Invalid Cover log-wealth enclosure")

    return lower, upper


class IntervalCover:
    """ Certify a played portfolio's cumulative log-wealth gap to Cover.

        The prior is symmetric Dirichlet with positive concentration alpha.
        Specify it as an exact integer or rational, or a binary64 value taken
        exactly. Input returns and weights are exact binary64 values, with
        played weights normalized in exact arithmetic.
    """

    def __init__(self, dimension, alpha=1):
        self.d = dimension
        self.alpha = alpha_parameter(exact_alpha(alpha))
        self.g = [arb(0) for _ in range(dimension)]
        self.B = arb_mat(dimension, dimension)
        self.H = arb_mat(dimension, dimension)
        self.log_played = arb(0)
        self.log_secondary = arb(0)
        self.zero_floor = None
        self.zero_log_means = {}

    def update(self, returns, weights, secondary=None):
        row = [arb(float(value)) for value in returns]
        low, high = min(row), max(row)
        if len(row) != self.d or low < 0 or not high > 0:
            raise ValueError("Each return row needs a positive maximum")

        normalizer = sum(row) / self.d
        self.log_played += played_log_gain(row, weights, normalizer)
        if secondary is not None:
            self.log_secondary += played_log_gain(row, secondary, normalizer)

        centered = [x / normalizer - 1 for x in row]
        self.g = [x + y for x, y in zip(self.g, centered)]
        col = arb_mat(self.d, 1, centered)
        outer = col * col.transpose()
        self.B += outer * (normalizer / (2 * high))

        if low > 0:
            self.H += outer * (normalizer / (2 * low))
        else:
            positive = [x for x in row if x > 0]
            k = len(positive)
            log_mean = self.zero_log_means.get(k)
            if log_mean is None:
                # Positive sleeves have total prior mass Beta(k*a,(d-k)*a).
                if self.alpha == 1:
                    log_mean = -sum(arb(1) / j for j in range(k, self.d))
                else:
                    log_mean = ((k * self.alpha).digamma()
                                - (self.d * self.alpha).digamma())
                self.zero_log_means[k] = log_mean

            floor = (min(positive) / normalizer).log() + log_mean
            if self.zero_floor is None:
                self.zero_floor = arb(0)

            self.zero_floor += floor

    def evidence(self):
        return log_evidence_bounds(self.g, self.B, self.H, self.zero_floor,
                                   self.alpha)

    def gap(self):
        return gap_to_wealth(self.evidence(), self.log_played)


def outward_float(endpoint, upper):
    direction = inf if upper else -inf
    value = nextafter(nextafter(float(endpoint), direction), direction)
    if not np.isfinite(value):
        raise OverflowError("Certificate endpoint exceeds binary64 range")

    representations = (arb(value), arb(str(value)))
    enclosed = all(x >= endpoint if upper else x <= endpoint
                   for x in representations)
    if not enclosed:
        raise ArithmeticError("Exported endpoint did not round outward")

    return value


def rounded_gap(bounds, log_wealth):
    lower, upper = gap_to_wealth(bounds, log_wealth)
    return {"lower": None if lower is None else outward_float(lower, False),
            "upper": outward_float(upper, True)}
