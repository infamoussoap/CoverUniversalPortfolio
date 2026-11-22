"""Streaming tilted enclosure and independent rational checks."""

from collections import defaultdict
from fractions import Fraction
from math import factorial, prod
from flint import arb, arb_mat, fmpq
from .interval_cover import (alpha_parameter, exact_alpha, moment_bounds,
                             nonnegative_upper, played_log_gain)
from .raised_pair_moments import RaisedSeries


def dot(x, y):
    return sum(a * b for a, b in zip(x, y))


def powers(x, degree):
    return [[value**j for value in x] for j in range(degree + 1)]


def homogeneous(p, degree, alpha=1):
    values = [arb(1)]

    for j in range(1, degree + 1):
        value = sum(p[k] * values[j - k] for k in range(1, j + 1))
        values.append(alpha * value / j)

    return values


def exponential_remainder(peak, degree):
    # exp Taylor remainder divided by x**degree is increasing in x.
    if peak > arb(1) / 1000:
        prefix = sum(peak**k / factorial(k) for k in range(degree))
        return (peak.exp() - prefix) / peak**degree

    return peak.exp() / factorial(degree)


def scalar_moments(a, rising, alpha=1):
    """ Return fourth and eighth moments of a centered row times W. """
    p = [len(a), 0] + [sum(x**k for x in a) for k in range(2, 9)]
    h = homogeneous(p, 8, alpha)
    return factorial(4) * h[4] / rising[4], factorial(8) * h[8] / rising[8]


def score_quadratic(matrix, gp, h, rising, alpha=1):
    """ Return E[G**k Q]/k! for symmetric A with A1=0 and centered g. """
    d = matrix.nrows()
    diag = [matrix[i, i] for i in range(d)]
    columns = arb_mat(d, 4, [gp[j][i] for i in range(d)
                             for j in range(1, 5)])
    bilinear = columns.transpose() * matrix * columns
    augmented = [alpha * (k + 1) * dot(diag, power)
                 + alpha**2 * sum(bilinear[i - 1, k - i - 1]
                                  for i in range(1, k))
                 for k, power in enumerate(gp)]
    return [sum(h[k - j] * augmented[j] for j in range(k + 1))
            / rising[k + 2] for k in range(5)]


def score_square(matrix, g, rising, alpha=1):
    """ Contract Q² and G Q² for symmetric A with A1=0 and centered g. """
    d = len(g)
    diag = [matrix[i, i] for i in range(d)]
    trace = sum(diag)
    # Arb multiplication also encloses squares of balls straddling zero.
    square = [sum(matrix[i, j] * matrix[i, j] for j in range(d))
              for i in range(d)]
    alpha2 = alpha**2
    raw_fourth = alpha2 * (trace**2 + 2 * sum(square))
    raw_fourth += 6 * alpha * dot(diag, diag)
    linear = matrix * arb_mat(d, 1, diag)
    raw_fifth = 24 * alpha * dot(g, [v**2 for v in diag])
    raw_fifth += (4 * alpha2 * dot(g, diag) * trace
                  + 8 * alpha2 * dot(g, square))
    raw_fifth += 8 * alpha2 * dot(g, linear.entries())
    return raw_fourth / rising[4], raw_fifth / rising[5]


class TiltedCover:
    """ Streaming Arb enclosure for tapes of strictly positive returns.

        Write G=sum(y), D=sum(y-log(1+y)), and y=a'W for each row.
        The upper bound uses exp(-D)<=1-D+D^2/2 and the lower uses
        Jensen under the exp(G) tilt. Tilted moments of Q=sum(y^2)/2
        and J=sum(y^3) use short expansions with rigorous remainders.
        A quadratic upper bound H>=D and Minkowski supply the square term.
        The prior is symmetric Dirichlet with positive concentration alpha.
        With exact_pairs=True, raised Dirichlet moments sharpen the terms
        in Q, the absolute cubic bound C and the quartic remainder. The square
        term also uses the second exponential moment. Rational inputs specify
        alpha exactly, including binary64 values.
    """

    def __init__(self, dimension, alpha=1, exact_pairs=True):
        self.d = dimension
        self.alpha = alpha_parameter(exact_alpha(alpha))
        total = dimension * self.alpha
        self.rising = [prod(total + j for j in range(k)) for k in range(9)]
        self.g = [arb(0) for _ in range(dimension)]
        self.K = arb_mat(dimension, dimension) if exact_pairs else None
        self.Q = arb_mat(dimension, dimension)
        self.C = arb_mat(dimension, dimension)
        self.H = arb_mat(dimension, dimension)
        self.J_matrix = arb_mat(dimension, dimension)
        self.J_diag = [arb(0) for _ in range(dimension)]
        self.J_weighted = [arb(0) for _ in range(dimension)]
        self.R4 = arb(0)
        self.H4_root = arb(0)
        self.log_played = arb(0)

    def update(self, returns, weights):
        row = [arb(float(value)) for value in returns]
        if (len(row) != self.d or not min(row) > 0
                or not all(x.is_finite() for x in row)):
            raise ValueError("Tilted bounds need finite positive returns")

        normalizer = sum(row) / self.d
        self.log_played += played_log_gain(row, weights, normalizer)
        relative = [x / normalizer for x in row]
        minimum = min(x.lower() for x in relative)
        a = [x - 1 for x in relative]
        squares = [x**2 for x in a]
        size = sum(squares)
        span = max(abs(x).upper() for x in a)
        self.g = [x + y for x, y in zip(self.g, a)]
        self.J_diag = [x + y * z for x, y, z in zip(self.J_diag, a, squares)]
        self.J_weighted = [x + size * y for x, y in zip(self.J_weighted, a)]

        col = arb_mat(self.d, 1, a)
        outer = col * col.transpose()
        if self.K is not None:
            self.K += outer * (span**2 / (4 * minimum))

        self.Q += outer / 2
        self.C += outer * span
        self.J_matrix += arb_mat(self.d, 1, squares) * col.transpose()

        low = minimum - 1
        h_factor = (low - minimum.log()) / low**2 if low < 0 else None
        if h_factor is None or not h_factor > 0:
            h_factor = 1 / (2 * minimum)

        self.H += outer * h_factor
        fourth, eighth = scalar_moments(a, self.rising, self.alpha)
        self.R4 += fourth / (4 * minimum)
        self.H4_root += h_factor * nonnegative_upper(eighth).sqrt().sqrt()

    def cubic_moments(self, gp):
        """ Return E[J], E[GJ], and E[G²J] from accumulated row sums. """
        m = self.rising
        alpha = self.alpha
        alpha2 = alpha**2
        g = self.g

        column = arb_mat(self.d, 1, g)
        cross = (column.transpose() * self.J_matrix * column)[0, 0]
        raw_third = 2 * alpha * sum(self.J_diag)
        raw_fourth = (3 * alpha2 * dot(g, self.J_weighted)
                      + 6 * alpha * dot(g, self.J_diag))
        raw_fifth = (2 * alpha2 * sum(gp[2]) * sum(self.J_diag)
                     + 12 * alpha2 * cross
                     + 6 * alpha2 * dot(gp[2], self.J_weighted)
                     + 24 * alpha * dot(gp[2], self.J_diag))
        return raw_third / m[3], raw_fourth / m[4], raw_fifth / m[5]

    def evidence(self, *, mgf_bounds=None, pair_series=None,
                 legacy_bounds=None):
        """Enclose log evidence, optionally reporting the unclipped control."""
        m = self.rising
        alpha = self.alpha
        g = self.g
        gp = powers(g, 4)
        h = homogeneous([self.d, 0] + [sum(row) for row in gp[2:]], 4, alpha)
        peak = max(x.upper() for x in g)
        factors = [exponential_remainder(peak, j) for j in (2, 3, 4)]
        if mgf_bounds is None:
            mgf_bounds = moment_bounds(g, alpha)
        (mgf_low, mgf_high), (_, mgf2_high) = mgf_bounds

        quad = score_quadratic(self.Q, gp, h, m, alpha)
        q_low = sum(quad[:4])
        q_high = q_low + factors[2] * factorial(4) * quad[4]

        ej, egj, eg2j = self.cubic_moments(gp)
        cubic = ej + egj + eg2j / 2
        absolute = score_quadratic(self.C, gp, h, m, alpha)
        deviation = factors[1] * (2 * nonnegative_upper(absolute[2])
                                  * 24 * nonnegative_upper(absolute[4])).sqrt()
        second, fifth = score_square(self.H, g, m, alpha)
        g4 = 24 * h[4] / m[4]
        mixed = nonnegative_upper(g4).sqrt() * self.H4_root**2
        b_high = (second + fifth + factors[0] * mixed).upper()
        k_high = (peak.exp() * self.R4).upper()

        if self.K is not None:
            if pair_series is None:
                pair_series = RaisedSeries(g, alpha)
            pairs = pair_series.pair_matrix()
            moments = pairs.entries()
            q = dot(moments, self.Q.entries())
            k = dot(moments, self.K.entries())
            q_low = max(q_low.lower(), q.lower())
            q_high = min(q_high.upper(), q.upper())
            k_high = min(k_high, k.upper())
            b_high = min(b_high, (mgf2_high.sqrt() * self.H4_root**2).upper())

            c_high = dot(moments, self.C.entries()).upper()
        else:
            # Taylor bound for I(C) when pair moments are not requested.
            c_high = (sum(absolute[:4])
                      + factors[2] * factorial(4) * absolute[4]).upper()

        # Intersect after mapping both cubic bounds into curvature bounds.
        # This also preserves the original outward rounding when it is tighter.
        base_low = max(arb(0), (q_low - (cubic + deviation) / 3).lower())
        base_high = nonnegative_upper(q_high - (cubic - deviation) / 3
                                      + k_high)

        if legacy_bounds is not None:
            upper = min(mgf_high, (mgf_high - base_low + b_high / 2).upper())
            lower = (mgf_low.log() - base_high / mgf_low).lower()
            legacy_bounds(lower, upper.log().upper())

        a_low = max(base_low, (q_low - c_high / 3).lower())
        a_high = min(base_high, nonnegative_upper(q_high + c_high / 3
                                                  + k_high))
        mass_upper = min(mgf_high, (mgf_high - a_low + b_high / 2).upper())
        log_lower = (mgf_low.log() - a_high / mgf_low).lower()
        return log_lower, mass_upper.log().upper()


def exact_product_moments(rows, alpha):
    """ Integrate prefix products of linear forms by rational expansion. """
    alpha = Fraction(alpha)
    d = len(rows[0])
    coefficients = {(0,) * d: Fraction(1)}
    masses = [Fraction(1)]
    denominator = Fraction(1)

    for degree, x in enumerate(rows, start=1):
        following = defaultdict(Fraction)
        for index, coefficient in coefficients.items():
            for i, value in enumerate(x):
                key = index[:i] + (index[i] + 1,) + index[i + 1:]
                following[key] += value * coefficient

        coefficients = following
        masses.append(masses[-1] * (alpha + degree - 1))
        denominator *= d * alpha + degree - 1
        evidence = sum(value * prod(masses[j] for j in index)
                       for index, value in following.items())
        yield evidence / denominator


def exact_evidence(returns, alpha=Fraction(1)):
    rows = []
    for row in returns:
        x = [Fraction(float(value)) for value in row]
        normalizer = sum(x) / len(x)
        rows.append([v / normalizer for v in x])

    for evidence in exact_product_moments(rows, alpha):
        yield arb(fmpq(evidence.numerator, evidence.denominator)).log()
