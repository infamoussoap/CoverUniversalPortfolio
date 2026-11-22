""" Evaluate the tilted certificate via raised moments in small dimensions. """

from collections import Counter
from itertools import combinations_with_replacement
from math import factorial, prod

from flint import arb


class RaisedTiltCover:
    """ Enclose uniform-prior Cover evidence using fourth-degree contractions.

        For positive paths in a small dimension, store coefficients of powers
        of the centered daily payoff. Compute their exponentially tilted means
        with one common positive series and its raised-parameter recurrences.
    """

    def __init__(self, dimension):
        self.d = dimension
        self.g = [arb(0) for _ in range(dimension)]
        assets = range(dimension)
        self.indices = [list(combinations_with_replacement(assets, k))
                        for k in range(5)]
        self.factors = []
        self.prior = []

        for k, group in enumerate(self.indices):
            counts = [prod(factorial(n) for n in Counter(key).values())
                      for key in group]
            self.factors.append([factorial(k) // v for v in counts])
            denominator = prod(range(dimension, dimension + k))
            self.prior.append([arb(v) / denominator for v in counts])

        pairs = self.indices[2]
        fourth = {key: j for j, key in enumerate(self.indices[4])}
        self.square = [(i, j, fourth[tuple(sorted(left + right))])
                       for i, left in enumerate(pairs)
                       for j, right in enumerate(pairs[:i + 1])]
        self.q = [arb(0) for _ in pairs]
        self.bound = self.q.copy()
        self.j = [arb(0) for _ in self.indices[3]]
        self.k = [arb(0) for _ in self.indices[4]]

    def update(self, row):
        row = [arb(float(v)) for v in row]
        if (len(row) != self.d or not min(row) > 0
                or not all(v.is_finite() for v in row)):
            raise ValueError("Raised moments require finite positive returns")

        equal = sum(row) / self.d
        a = [v / equal - 1 for v in row]
        minimum = min((v / equal).lower() for v in row)
        self.g = [v + h for v, h in zip(self.g, a)]
        for degree, accumulator in ((2, self.q), (3, self.j), (4, self.k)):
            for pos, key in enumerate(self.indices[degree]):
                value = self.factors[degree][pos] * prod(a[i] for i in key)
                if degree == 2:
                    accumulator[pos] += value / 2
                    self.bound[pos] += value / (2 * minimum)
                elif degree == 3:
                    accumulator[pos] += value / 3
                else:
                    accumulator[pos] += value / (4 * minimum)

    def moments(self):
        shift = min(v.lower() for v in self.g)
        h = [v - shift for v in self.g]
        radius = max(v.upper() for v in h)
        degree = max(30, 4 * int(float(radius) + 1) + 16)
        ratio = radius / (degree + 2)
        if not ratio < 1:
            raise ArithmeticError("Cannot enclose MGF tail")

        # The same positive absolute tail applies to every raised prior.
        remainder = radius**(degree + 1) / factorial(degree + 1) / (1 - ratio)
        tail = arb(0).union(remainder.upper())
        base = [arb(1)] + [arb(0) for _ in range(degree)]
        for v in h:
            for k in range(1, len(base)):
                base[k] += v * base[k - 1]

        previous = {(): base}
        means = []
        for k, group in enumerate(self.indices):
            denominator = [arb(1)]
            for j in range(degree):
                denominator.append(denominator[-1] / (self.d + k + j))

            current = {}
            values = []
            for idx, key in enumerate(group):
                if k:
                    series = previous[key[:-1]].copy()
                    v = h[key[-1]]
                    for j in range(1, len(series)):
                        series[j] += v * series[j - 1]
                else:
                    series = base

                prefix = sum(v * w for v, w in zip(series, denominator))
                values.append(self.prior[k][idx] * (prefix + tail))
                current[key] = series

            previous = current
            means.append(values)

        return shift, means

    def evidence(self):
        shift, moments = self.moments()
        mgf = moments[0][0]
        if not mgf > 0:
            raise ArithmeticError("Cannot verify a positive normalizer")

        squared = [arb(0) for _ in self.indices[4]]
        for i, j, pos in self.square:
            factor = 1 if i == j else 2
            squared[pos] += factor * self.bound[i] * self.bound[j]

        linear = sum(v * w for v, w in zip(self.q, moments[2]))
        linear -= sum(v * w for v, w in zip(self.j, moments[3]))
        remainder = sum(v * w for v, w in zip(self.k, moments[4]))
        second = sum(v * w for v, w in zip(squared, moments[4]))
        lower = shift + mgf.log() - (linear + remainder) / mgf
        curvature_lower = max(arb(0), linear.lower())
        taylor_mass = mgf - curvature_lower + second / 2
        upper_mass = min(mgf.upper(), taylor_mass.upper())
        upper = shift + upper_mass.log()
        return lower.lower(), upper.upper()
