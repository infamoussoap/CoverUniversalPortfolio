"""Decompose the lower and upper evidence widths."""

import numpy as np


def slack_sample(centered, particles):
    """ Estimate the slack of the ideal tilted bounds in mass units.

        Parameters
        ----------
        centered : (T, d) np.ndarray
            Daily returns relative to the equal-weight payoff, less one.
        particles : (N, d) np.ndarray
            Uniform-prior portfolios from one scrambled net.

        Returns
        -------
        estimates : (5,) np.ndarray
            The two lower gaps and the three upper gaps in the theorem.
    """
    minimum = 1 + centered.min(axis=1)
    shift = minimum - 1
    eta = np.full_like(shift, 0.5)
    varied = shift != 0
    eta[varied] = (shift[varied] - np.log1p(shift[varied])) / shift[varied]**2

    chunks = []
    for batch in np.array_split(particles, len(particles) // 256):
        y = centered @ batch.T
        square = y * y
        D = (y - np.log1p(y)).sum(axis=0)
        D3 = (square * (0.5 - y / 3)).sum(axis=0)
        majorant = eta @ square
        K = (0.25 / minimum) @ (square * square)
        weight = np.exp(y.sum(axis=0))

        if np.any(D < D3 - 1e-14) or np.any(D > D3 + K + 1e-14):
            raise ArithmeticError('Cubic remainder escaped its bounds')

        upper = np.column_stack((D - D3, (majorant - D) * (majorant + D) / 2,
                                 D * D / 2 - D - np.expm1(-D), K))
        direct = -D3 + majorant**2 / 2 - np.expm1(-D)
        np.testing.assert_allclose(upper[:, :3].sum(axis=1), direct,
                                   atol=1e-14, rtol=1e-12)
        if np.any(majorant < D - 1e-14):
            raise ArithmeticError('Quadratic majorant fell below curvature')

        chunks.append((weight, D, upper))

    weights, curvature, terms = (np.concatenate(a) for a in zip(*chunks))
    weights /= weights.sum()
    mean = weights @ curvature
    jensen = np.exp(-mean) * (weights @ np.expm1(mean - curvature))
    cubic, majorant, remainder, quartic = weights @ terms
    lower_remainder = -np.exp(-mean) * np.expm1(cubic - quartic)

    return np.array([jensen, lower_remainder, cubic, majorant, remainder])
