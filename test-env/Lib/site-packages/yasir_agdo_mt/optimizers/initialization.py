"""
Population initialization utilities for optimization algorithms.
"""

from __future__ import annotations

import numpy as np


def initialize_population(
    npop: int,
    lb: np.ndarray,
    ub: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate an initial population uniformly within parameter bounds.

    Parameters
    ----------
    npop : int
        Population size.

    lb : ndarray
        Lower bounds.

    ub : ndarray
        Upper bounds.

    rng : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    ndarray
        Initial population with shape (npop, ndim).
    """

    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)

    if lb.shape != ub.shape:
        raise ValueError(
            "Lower and upper bounds must have the same shape."
        )

    if np.any(lb >= ub):
        raise ValueError(
            "Each lower bound must be smaller than upper bound."
        )

    if rng is None:
        rng = np.random.default_rng()

    population = rng.uniform(
        low=lb,
        high=ub,
        size=(npop, lb.size),
    )

    return population