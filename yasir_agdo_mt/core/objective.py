"""
Objective functions for 1D Magnetotelluric inversion.
"""

from __future__ import annotations

import numpy as np

from .forward import (
    mt1d,
    apparent_resistivity,
    phase,
)

def rmse_mt(
    rho_obs: np.ndarray,
    phase_obs: np.ndarray,
    rho_cal: np.ndarray,
    phase_cal: np.ndarray,
) -> float:
    """
    Compute the root mean square error (RMSE) between observed and
    calculated MT responses.

    Parameters
    ----------
    rho_obs : np.ndarray
        Observed apparent resistivity (ohm.m).
    phase_obs : np.ndarray
        Observed phase (degrees).
    rho_cal : np.ndarray
        Calculated apparent resistivity (ohm.m).
    phase_cal : np.ndarray
        Calculated phase (degrees).

    Returns
    -------
    float
        RMSE value.
    """

    # Validate input shapes
    if rho_obs.shape != rho_cal.shape:
        raise ValueError(
            "Observed and calculated resistivity must have the same shape."
        )

    if phase_obs.shape != phase_cal.shape:
        raise ValueError(
            "Observed and calculated phase must have the same shape."
        )

    # Compute residuals
    rho_residual = np.log10(rho_obs / rho_cal)
    phase_residual = np.deg2rad(phase_obs - phase_cal)

    # Root Mean Square Error
    rmse = np.sqrt(
        np.mean(
            rho_residual**2 +
            phase_residual**2
        )
    )

    return float(rmse)

def evaluate_mt_model(
    model: np.ndarray,
    frequency: np.ndarray,
    rho_obs: np.ndarray,
    phase_obs: np.ndarray,
) -> float:
    """
    Evaluate a 1D magnetotelluric model by computing the RMSE
    between observed and calculated responses.

    Parameters
    ----------
    model : np.ndarray
        Model vector containing layer resistivities followed by
        layer thicknesses.
    frequency : np.ndarray
        Frequency array (Hz).
    rho_obs : np.ndarray
        Observed apparent resistivity.
    phase_obs : np.ndarray
        Observed phase (degrees).

    Returns
    -------
    float
        Root mean square error (RMSE).
    """

    model = np.asarray(model, dtype=float)

    if model.ndim != 1:
        raise ValueError("Model vector must be one-dimensional.")

    if len(model) % 2 == 0:
        raise ValueError(
            "Model vector must contain n resistivities and n-1 thicknesses."
        )

    nlayer = (len(model) + 1) // 2

    resistivity = model[:nlayer]
    thickness = model[nlayer:]

    Z = mt1d(
        resistivity,
        thickness,
        frequency,
    )

    rho_cal = apparent_resistivity(
        Z,
        frequency,
    )

    phase_cal = phase(Z)

    return rmse_mt(
        rho_obs,
        phase_obs,
        rho_cal,
        phase_cal,
    )