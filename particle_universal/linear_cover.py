"""First-order Cover allocation and the tilted-mean fallback."""

import numpy as np
from .market_reference import _normalize


def tilt_mean(score, alpha=1, tolerance=1e-14):
    """ Evaluate the mean of a linearly tilted symmetric Dirichlet prior.

        Parameters
        ----------
        score : (d,) np.ndarray
            Coefficients in the log density tilt.
        alpha : float
            Positive concentration for each coordinate.
        tolerance : float
            Relative Taylor-tail target in binary64 arithmetic.

        Returns
        -------
        weights : (d,) np.ndarray
            Normalized posterior mean for the linear exponential tilt.
    """
    score = np.asarray(score, dtype=float)
    if (score.ndim != 1 or not score.size or not np.isfinite(score).all()
            or not np.isfinite(alpha) or alpha <= 0
            or not 0 < tolerance < 1):
        raise ValueError("Invalid score, concentration, or series tolerance")

    h = score - score.min()
    radius = h.max()
    d = len(h)
    term = np.ones(d)
    total = term.copy()
    k = 0

    while True:
        k += 1
        shifted = h * term / (d * alpha + k)
        term = shifted + alpha * shifted.sum() / k
        total += term

        # Each coordinate series is a raised-parameter Dirichlet MGF.
        # Since 0 <= W.h <= radius, later terms decay at least geometrically.
        ratio = radius / (k + 1)
        if ratio < 1 and np.all(term * ratio <= tolerance * (1 - ratio)
                                * total):
            return total / total.sum()

        scale = total.max()
        if scale > 1e100:
            total /= scale
            term /= scale


def linear_cover(returns, alpha=1, fallback_equal=None):
    """ Approximate the predictive simplex mean by its linear response.

        Parameters
        ----------
        returns : (T, d) np.ndarray
            Gross returns for T dates and d assets.
        alpha : float
            Common concentration of the symmetric Dirichlet prior.
        fallback_equal : bool or None
            By default use the linearly tilted mean if any coordinate leaves
            the interior. True reproduces the historical equal-weight fallback
            and False requests an exception. All choices update the score.

        Returns
        -------
        weights : (T, d) np.ndarray
            Pre-round predictions.
    """
    market, _ = _normalize(returns)
    T, d = market.shape
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Dirichlet concentration must be positive and finite")

    weights = np.empty_like(market)
    weights[0] = 1 / d
    gradient = np.zeros(d)
    concentration = d * alpha + 1
    for t, row in enumerate(market[:-1]):
        gradient += row / row.mean() - 1
        weights[t + 1] = (1 + gradient / concentration) / d
        if weights[t + 1].min() <= 0:
            if fallback_equal is None:
                weights[t + 1] = tilt_mean(gradient, alpha)
            elif fallback_equal:
                weights[t + 1] = 1 / d
            else:
                raise FloatingPointError(
                    f"Linear portfolio left simplex at {t+1}")

    return weights
