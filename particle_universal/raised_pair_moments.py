""" Exponential moments of Dirichlet monomials with certified series tails. """

from collections import Counter
from math import factorial, prod

from flint import arb, arb_mat, arb_poly

from .interval_cover import alpha_parameter


def raise_series(base, score):
    """ Multiply the series by (1 - score*z)^(-1) to its stored degree. """
    raised = base.copy()
    for j in range(1, len(base)):
        raised[j] += score * raised[j - 1]

    return raised


def inverse_rising(total, degree):
    values = [arb(1)]
    for j in range(degree):
        values.append(values[-1] / (total + j))

    return values


class RaisedSeries:
    """ Enclose exponential moments under successively raised Dirichlet laws.

        Shifting by the smallest score makes the exponential Taylor terms
        nonnegative. Raising one Dirichlet parameter adds a geometric
        factor to the generating series. The same absolute tail bound
        applies after any number of parameter raises.
    """

    def __init__(self, scores, alpha=1):
        self.d = len(scores)
        self.alpha = alpha_parameter(alpha)
        self.shift = min(x.lower() for x in scores)
        self.h = [x - self.shift for x in scores]
        radius = max(v.upper() for v in self.h)

        self.degree = max(30, 4 * int(float(radius) + 1) + 16)
        ratio = radius / (self.degree + 2)
        if not ratio < 1:
            raise ArithmeticError("Increase the Dirichlet series degree")

        first = radius**(self.degree + 1) / factorial(self.degree + 1)
        self.tail = arb(0).union((first / (1 - ratio)).upper())
        self.base = self.base_series()

    def base_series(self):
        base = [arb(1)] + [arb(0) for _ in range(self.degree)]
        if self.alpha == 1:
            for score in self.h:
                base = raise_series(base, score)

            return base

        # D(z) F'(z) = -alpha D'(z) F(z), D(z)=prod_i(1-h_i z).
        product = arb_poly([1])
        for score in self.h:
            product *= arb_poly([1, -score])

        terms = product.coeffs()
        for k in range(1, len(base)):
            base[k] = -sum((k + (self.alpha - 1) * j) * terms[j]
                           * base[k - j]
                           for j in range(1, min(k + 1, len(terms)))) / k

        return base

    def pair_matrix(self):
        """ Return all E[exp(G) W_i W_j] in one symmetric Arb matrix. """
        d, alpha = self.d, self.alpha
        total = d * alpha
        denom = inverse_rising(total + 2, self.degree)
        raised = [raise_series(self.base, x) for x in self.h]
        singles = [sum(x * v for x, v in zip(s, denom)) for s in raised]
        scale = self.shift.exp() * alpha / (total * (total + 1))

        pairs = arb_mat(d, d)
        for i, hi in enumerate(self.h):
            for j in range(i + 1):
                hj = self.h[j]
                difference = hi - hj
                # Direct raising avoids a nearly singular divided difference.
                if abs(difference) > (1 + abs(hi) + abs(hj)) / 1024:
                    prefix = (hi * singles[i] - hj * singles[j]) / difference
                else:
                    series = raise_series(raised[i], hj)
                    prefix = sum(x * v for x, v in zip(series, denom))

                moment = (alpha + int(i == j)) * scale * (prefix + self.tail)
                pairs[i, j] = pairs[j, i] = moment

        return pairs

    def shifted_monomials(self, keys):
        """ Return E[exp(G-shift) prod(W_i, i in key)] for sorted keys. """
        total = self.d * self.alpha
        previous = {(): self.base}
        moments = {}
        keys_by_degree = {0: {()}}
        for key in keys:
            for k in range(1, len(key) + 1):
                keys_by_degree.setdefault(k, set()).add(key[:k])

        for k, group in sorted(keys_by_degree.items()):
            denominator = inverse_rising(total + k, self.degree)
            prior_den = prod(total + n for n in range(k))
            current = {}
            for key in group:
                series = self.base if not k else raise_series(
                    previous[key[:-1]], self.h[key[-1]])
                prior_num = prod(prod(self.alpha + n for n in range(count))
                                 for count in Counter(key).values())
                prefix = sum(x * v for x, v in zip(series, denominator))
                moments[key] = (prefix + self.tail) * prior_num / prior_den
                current[key] = series

            previous = current

        return moments
