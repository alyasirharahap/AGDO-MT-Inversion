"""
Levy flight utilities.
"""

from __future__ import annotations

import math

import numpy as np


def levy_step(
    ndim: int,
    beta: float = 1.5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate a Levy flight step using Mantegna's algorithm.

    Parameters
    ----------
    ndim : int
        Number of dimensions.

    beta : float, default=1.5
        Levy distribution exponent.

    rng : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    ndarray
        Levy flight step.
    """

    if beta <= 0 or beta >= 2:
        raise ValueError(
            "beta must satisfy 0 < beta < 2."
        )

    if rng is None:
        rng = np.random.default_rng()

    sigma = (
        math.gamma(1 + beta)
        * np.sin(np.pi * beta / 2)
        / (
            math.gamma((1 + beta) / 2)
            * beta
            * 2 ** ((beta - 1) / 2)
        )
    ) ** (1 / beta)

    u = rng.normal(0, sigma, ndim)
    v = rng.normal(0, 1, ndim)

    step = u / (np.abs(v) ** (1 / beta))

    return step