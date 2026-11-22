""" Certify small-dimensional Cover wealth while keeping zero rows exact. """

from collections import defaultdict

from flint import arb

from .interval_cover import alpha_parameter, exact_alpha
from .raised_pair_moments import RaisedSeries
from .raised_tilt_cover import RaisedTiltCover


class ProductTiltCover(RaisedTiltCover):
    """ Enclose normalized Dirichlet mixture wealth for nonnegative returns.

        Store each row containing a zero as an exact linear factor and
        integrate their product against the exponential tilt. Coefficients
        use outward-rounded Arb arithmetic. Suitable for small dimensions
        and few exact rows: the required moments have degree up to four
        plus the number of rows kept exact.
    """

    def __init__(self, dimension, alpha=1):
        super().__init__(dimension)
        self.alpha = alpha_parameter(exact_alpha(alpha))
        self.polynomial = {(): arb(1)}
        self.exact_days = 0

    def update(self, row, keep_exact=False):
        """ Include a return row, optionally factoring it out of the tilt. """
        values = [arb(float(x)) for x in row]
        if len(values) != self.d or not all(x.is_finite() for x in values):
            raise ValueError("Expected a finite return for each asset")
        if min(values) < 0 or not max(values) > 0:
            raise ValueError("Expected a nonnegative row of positive mass")

        if min(values) > 0 and not keep_exact:
            super().update(row)
            return

        mean = sum(values) / self.d
        coefficients = [(j, x / mean) for j, x in enumerate(values) if x > 0]
        updated = defaultdict(arb)
        for monomial, value in self.polynomial.items():
            for j, coefficient in coefficients:
                key = tuple(sorted((*monomial, j)))
                updated[key] += value * coefficient

        self.polynomial = dict(updated)
        self.exact_days += 1

    def evidence(self):
        tilted = RaisedSeries(self.g, self.alpha)
        keys = [()] + self.indices[2] + self.indices[3] + self.indices[4]
        extensions = {key: [(tuple(sorted(key + power)), c)
                            for power, c in self.polynomial.items()]
                      for key in keys}
        required = {k for terms in extensions.values() for k, c in terms}
        raw = tilted.shifted_monomials(required)
        moments = {key: sum(c * raw[k] for k, c in terms)
                   for key, terms in extensions.items()}

        mass = moments[()]
        linear = sum(v * moments[k] for v, k in zip(self.q, self.indices[2]))
        linear -= sum(v * moments[k] for v, k in zip(self.j, self.indices[3]))
        fourth_keys = self.indices[4]
        remainder = sum(v * moments[k] for v, k in zip(self.k, fourth_keys))
        square = sum((1 if i == j else 2) * self.bound[i] * self.bound[j]
                     * moments[fourth_keys[pos]] for i, j, pos in self.square)

        if not mass > 0:
            raise ArithmeticError("Cannot enclose the positive tilted mass")

        lower = mass.log() - (linear + remainder) / mass
        curvature_lower = max(arb(0), linear.lower())
        taylor_mass = mass - curvature_lower + square / 2
        upper = min(mass.upper(), taylor_mass.upper())
        lower_log = (tilted.shift + lower).lower()
        return lower_log, (tilted.shift + upper.log()).upper()
