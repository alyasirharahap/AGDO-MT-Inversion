from __future__ import annotations

import numpy as np


def add_gaussian_noise(
    data: np.ndarray,
    noise_level: float = 0.05,
    *,
    seed: int | None = None,
) -> np.ndarray:
    """
    Add relative Gaussian noise to synthetic data.

    Parameters
    ----------
    data : np.ndarray
        Input data.

    noise_level : float, default=0.05
        Relative standard deviation of the Gaussian noise.
        For example, 0.05 represents a 5% noise level.

    seed : int or None, optional
        Random seed for reproducible noise generation.

    Returns
    -------
    np.ndarray
        Data with Gaussian noise added.
    """

    data = np.asarray(
        data,
        dtype=float,
    )

    if data.ndim != 1:
        raise ValueError(
            "data must be one-dimensional."
        )

    if noise_level < 0:
        raise ValueError(
            "noise_level must be non-negative."
        )

    if not np.all(np.isfinite(data)):
        raise ValueError(
            "data must contain only finite values."
        )

    rng = np.random.default_rng(seed)

    noise = rng.normal(
        loc=0.0,
        scale=noise_level,
        size=data.shape,
    )

    return data * (1.0 + noise)