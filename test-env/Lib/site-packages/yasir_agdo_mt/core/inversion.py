from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .forward import (
    mt1d,
    apparent_resistivity,
    phase,
)

from ..optimizers.agdo import (
    AGDOConfig,
    AGDOResult,
    agdo_mt,
)


@dataclass(slots=True)
class MTInversionResult:
    """
    Result of a one-dimensional magnetotelluric inversion.

    Attributes
    ----------
    resistivity : np.ndarray
        Estimated layer resistivities in ohm.m.

    thickness : np.ndarray
        Estimated layer thicknesses in meters.

    rho_cal : np.ndarray
        Calculated apparent resistivity response.

    phase_cal : np.ndarray
        Calculated phase response in degrees.

    best_fitness : float
        Best objective-function value obtained by AGDO.

    convergence : np.ndarray
        Best fitness value recorded at each iteration.

    population_history : np.ndarray
        Mean population fitness recorded at each iteration.

    iterations : int
        Number of optimization iterations performed.

    converged : bool
        Whether the configured early-stopping criterion was reached.
    """

    resistivity: np.ndarray
    thickness: np.ndarray

    rho_cal: np.ndarray
    phase_cal: np.ndarray

    best_fitness: float

    convergence: np.ndarray
    population_history: np.ndarray

    iterations: int
    converged: bool

def invert_mt1d(
    frequency: np.ndarray,
    rho_obs: np.ndarray,
    phase_obs: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    *,
    config: AGDOConfig | None = None,
) -> MTInversionResult:
    """
    Perform one-dimensional magnetotelluric inversion using AGDO.

    Parameters
    ----------
    frequency : np.ndarray
        Frequency array in Hz.

    rho_obs : np.ndarray
        Observed apparent resistivity in ohm.m.

    phase_obs : np.ndarray
        Observed phase in degrees.

    lb : np.ndarray
        Lower bounds of the model parameters.

        The model vector must follow the structure:

        [rho_1, rho_2, ..., rho_n,
         h_1, h_2, ..., h_(n-1)]

    ub : np.ndarray
        Upper bounds of the model parameters.

    config : AGDOConfig, optional
        Configuration of the AGDO optimizer.
        If None, the default AGDO configuration is used.

    Returns
    -------
    MTInversionResult
        Inversion result containing the estimated model,
        calculated MT responses, fitness history,
        and optimization information.
    """

    # ------------------------------------------------------
    # Convert inputs to NumPy arrays
    # ------------------------------------------------------

    frequency = np.asarray(
        frequency,
        dtype=float,
    )

    rho_obs = np.asarray(
        rho_obs,
        dtype=float,
    )

    phase_obs = np.asarray(
        phase_obs,
        dtype=float,
    )

    lb = np.asarray(
        lb,
        dtype=float,
    )

    ub = np.asarray(
        ub,
        dtype=float,
    )

    # ------------------------------------------------------
    # Run AGDO inversion
    # ------------------------------------------------------

    optimization_result: AGDOResult = agdo_mt(
        frequency=frequency,
        rho_obs=rho_obs,
        phase_obs=phase_obs,
        lb=lb,
        ub=ub,
        config=config,
    )

    # ------------------------------------------------------
    # Separate resistivity and thickness
    # ------------------------------------------------------

    best_model = optimization_result.best_model

    nlayer = (
        best_model.size + 1
    ) // 2

    resistivity = best_model[:nlayer].copy()

    thickness = best_model[nlayer:].copy()

    # ------------------------------------------------------
    # Calculate response of the inverted model
    # ------------------------------------------------------

    Z = mt1d(
        resistivity=resistivity,
        thickness=thickness,
        frequency=frequency,
    )

    rho_cal = apparent_resistivity(
        Z,
        frequency,
    )

    phase_cal = phase(Z)

    # ------------------------------------------------------
    # Return inversion result
    # ------------------------------------------------------

    return MTInversionResult(
        resistivity=resistivity,
        thickness=thickness,
        rho_cal=rho_cal,
        phase_cal=phase_cal,
        best_fitness=optimization_result.best_fitness,
        convergence=optimization_result.convergence.copy(),
        population_history=(
            optimization_result.population_history.copy()
        ),
        iterations=optimization_result.iterations,
        converged=optimization_result.converged,
    )