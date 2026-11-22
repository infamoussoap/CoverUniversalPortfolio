"""Horizon-tuned Soft-Bayes allocations."""

import numpy as np
from .market_reference import _normalize


def learning_rate(horizon, d):
    eta_bar = np.sqrt(np.log(d) / (d * horizon))
    return float(eta_bar / (1 + eta_bar))


def soft_bayes(returns, horizon=None):
    """ Run horizon-tuned Soft-Bayes on nonnegative gross returns.

        Uses the fixed-rate procedure and Theorem 3 of Orseau, Lattimore,
        and Legg (2017), https://proceedings.mlr.press/v76/orseau17a.html,
        with the total number of assets as the number of experts.

        Parameters
        ----------
        returns : (T, d) np.ndarray
            Gross returns with d ≥ 2. Each row has a positive maximum.
        horizon : int or None
            Declared horizon, at least T. Defaults to T for a full replay.

        Returns
        -------
        weights : (T, d) np.ndarray
            Long-only allocations chosen before each return.
    """
    market, _ = _normalize(returns)
    T, d = market.shape
    horizon = T if horizon is None else horizon
    if not isinstance(horizon, (int, np.integer)) or horizon < T:
        raise ValueError("Declared horizon must be an integer at least T")

    eta = learning_rate(horizon, d)
    retain = 1 - eta
    w = np.full(d, 1 / d)
    weights = np.empty_like(market)
    for t, row in enumerate(market):
        weights[t] = w
        gain = w @ row
        if not gain > 0:
            raise FloatingPointError("Soft-Bayes portfolio payoff underflowed")

        w *= retain + (eta / gain) * row

    return weights
