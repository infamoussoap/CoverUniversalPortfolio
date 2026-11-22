"""VB-FTRL solver with the published finite-iteration schedule."""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from .market_reference import _normalize


def quasi_newton(gradients, strength, mu, identity):
    hessian = gradients.T @ (strength[:, None] * gradients)
    factor = cho_factor(hessian, lower=True, check_finite=False)
    inverse = cho_solve(factor, identity, check_finite=False)
    leverage = strength * np.einsum("ij,ij->i", gradients @ inverse,
                                    gradients)

    gradient = (strength + mu * leverage) @ gradients
    metric = hessian + 3 * mu * (gradients.T @ (leverage[:, None] * gradients))
    factor = cho_factor(metric, lower=True, check_finite=False)
    direction = cho_solve(factor, gradient, check_finite=False)
    decrement = float(np.sqrt(gradient @ direction))
    return direction, decrement


def solve_round(history, tangent, w, A, gradients, strength, identity,
                mu, steps, tolerance, skip_cycles=False):
    d = len(w)
    seen = {} if skip_cycles else None
    step = 0
    evaluations = 0
    while step < steps:
        if seen is not None:
            seen[w.tobytes()] = step

        gradients[:d] = -A / w[:, None]
        gradients[d:] = -tangent / (history @ w)[:, None]
        direction, decrement = quasi_newton(gradients, strength, mu, identity)
        evaluations += 1

        if decrement <= tolerance:
            return evaluations, decrement, True

        w[:-1] -= direction
        w[-1] = 1 - w[:-1].sum()
        if not np.all(w > 0):
            raise FloatingPointError("VB-FTRL iterate left the open simplex")

        step += 1
        if seen is not None:
            previous = seen.get(w.tobytes())
            if previous is not None:
                period = step - previous
                step += (steps - step) // period * period
                # The remaining partial cycle needs no further shortcuts.
                seen = None

    return evaluations, decrement, False


def vb_ftrl(returns, regularization=560, mu=2, steps=16, tolerance=1e-8,
            skip_cycles=False):
    """ Run the volumetric-barrier strategy with warm-started updates.

        Implements the direction in Algorithm 1 of Jézéquel, Ostrovskii,
        and Gaillard (2025), https://doi.org/10.1287/moor.2023.0175.
        Early stopping uses the pre-step quasi-Newton decrement.

        Parameters
        ----------
        returns : (T, d) np.ndarray
            Gross returns on T dates for d assets.
        regularization : float
            Multiplier of the cumulative logarithmic simplex barrier.
        mu : float
            Multiplier of the volumetric barrier.
        steps : int
            Maximum quasi-Newton steps per round.
        tolerance : float
            Stop when the quasi-Newton decrement is at most this value.
            A negative value executes all requested iterations.
        skip_cycles : bool
            Skip whole periods after a bitwise state repeat. This applies
            the same scheduled update map under deterministic arithmetic.

        Returns
        -------
        result : dict
            Pre-return portfolio weights, gains, and optimizer diagnostics.
    """
    market, scale = _normalize(returns)
    if regularization <= 0 or mu <= 0 or steps < 1:
        raise ValueError("Invalid barrier weights or step budget")

    T, d = market.shape
    identity = np.eye(d - 1)
    A = np.vstack((identity, -np.ones(d - 1)))
    tangent = market[:, :-1] - market[:, -1:]
    w = np.full(d, 1 / d)
    gradients = np.empty((d + T, d - 1))
    strength = np.ones(d + T)
    strength[:d] = regularization

    weights = np.empty_like(market)
    evaluations = np.zeros(T, dtype=int)
    decrements = np.zeros(T)
    converged = np.ones(T, dtype=bool)
    weights[0] = w
    for t in range(1, T):
        count, decrement, done = solve_round(
            market[:t], tangent[:t], w, A, gradients[:d+t], strength[:d+t],
            identity, mu, steps, tolerance, skip_cycles)
        weights[t] = w
        evaluations[t] = count
        decrements[t] = decrement
        converged[t] = done

    log_gains = np.log(np.einsum("ij,ij->i", market, weights)) + scale
    return {"weights": weights, "log_gains": log_gains,
            "evaluations": evaluations, "decrements": decrements,
            "converged": converged}


def published_steps(T, d):
    return int(18 * np.ceil(np.log(T + d + 164)) + 7)
