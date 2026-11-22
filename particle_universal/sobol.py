"""Sobol reference settings and analytic control polynomial."""

import numpy as np


SEED = 841120
POWER = 16
SCRAMBLES = 8


def polynomial_mass(a):
    """ Integrate the degree-four truncation of the product over the simplex.

        Parameters
        ----------
        a : (T, d) np.ndarray
            Centered relative return rows.

        Returns
        -------
        masses : (T, 2) np.ndarray
            Exact polynomial integrals through degrees two and four at
            each prefix, evaluated in floating point.
    """
    T, d = a.shape
    r2 = d * (d + 1)
    r3 = r2 * (d + 2)
    r4 = r3 * (d + 3)
    g = np.zeros(d)
    B = np.zeros((d, d))
    cubes = np.zeros(d)
    weighted_rows = np.zeros(d)
    e_k = 0
    masses = np.empty((T, 2))

    for i, row in enumerate(a):
        squares = row * row
        norm = squares.sum()
        g += row
        B += np.outer(row, row)
        cubes += squares * row
        weighted_rows += norm * row
        e_k += (3 * norm**2 + 6 * (squares @ squares)) / r4

        diagonal = B.diagonal()
        trace = diagonal.sum()
        g2 = g @ g
        quadratic = (g2 - trace) / (2 * r2)
        cubic = ((g**3).sum() / 3 - g @ diagonal + 2 * cubes.sum() / 3) / r3
        e_g4 = (3 * g2**2 + 6 * (g**4).sum()) / r4
        e_gj = (3 * (g @ weighted_rows) + 6 * (g @ cubes)) / r4
        e_q2 = (trace**2 + 2 * (B * B).sum() + 6 * (diagonal @ diagonal)) / r4
        e_g2q = (g2 * trace + 2 * (g @ B @ g) + 6 * (g**2 @ diagonal)) / r4
        fourth = e_g4 / 24 - e_g2q / 4 + e_q2 / 8 + e_gj / 3 - e_k / 4
        masses[i] = 1 + quadratic, 1 + quadratic + cubic + fourth

    return masses
