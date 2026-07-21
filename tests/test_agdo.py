import numpy as np

from yasir_agdo_mt.core.forward import (
    mt1d,
    apparent_resistivity,
    phase,
)

from yasir_agdo_mt.optimizers.agdo import (
    AGDOConfig,
    AGDOResult,
    agdo_mt,
)


def create_synthetic_data():
    """
    Create a simple three-layer synthetic MT dataset.
    """

    frequency = np.logspace(-3, 4, 20)

    true_resistivity = np.array([
        100.0,
        10.0,
        1000.0,
    ])

    true_thickness = np.array([
        500.0,
        1000.0,
    ])

    Z = mt1d(
        true_resistivity,
        true_thickness,
        frequency,
    )

    rho_obs = apparent_resistivity(
        Z,
        frequency,
    )

    phase_obs = phase(Z)

    return (
        frequency,
        rho_obs,
        phase_obs,
    )


def test_agdo_returns_result():

    frequency, rho_obs, phase_obs = (
        create_synthetic_data()
    )

    lb = np.array([
        1.0,
        1.0,
        1.0,
        10.0,
        10.0,
    ])

    ub = np.array([
        3000.0,
        3000.0,
        3000.0,
        2000.0,
        2000.0,
    ])

    config = AGDOConfig(
        npop=10,
        niter=5,
        seed=42,
    )

    result = agdo_mt(
        frequency=frequency,
        rho_obs=rho_obs,
        phase_obs=phase_obs,
        lb=lb,
        ub=ub,
        config=config,
    )

    assert isinstance(
        result,
        AGDOResult,
    )


def test_agdo_model_shape():

    frequency, rho_obs, phase_obs = (
        create_synthetic_data()
    )

    lb = np.array([
        1.0,
        1.0,
        1.0,
        10.0,
        10.0,
    ])

    ub = np.array([
        3000.0,
        3000.0,
        3000.0,
        2000.0,
        2000.0,
    ])

    config = AGDOConfig(
        npop=10,
        niter=5,
        seed=42,
    )

    result = agdo_mt(
        frequency=frequency,
        rho_obs=rho_obs,
        phase_obs=phase_obs,
        lb=lb,
        ub=ub,
        config=config,
    )

    assert result.best_model.shape == lb.shape


def test_agdo_fitness_is_finite():

    frequency, rho_obs, phase_obs = (
        create_synthetic_data()
    )

    lb = np.array([
        1.0,
        1.0,
        1.0,
        10.0,
        10.0,
    ])

    ub = np.array([
        3000.0,
        3000.0,
        3000.0,
        2000.0,
        2000.0,
    ])

    config = AGDOConfig(
        npop=10,
        niter=5,
        seed=42,
    )

    result = agdo_mt(
        frequency=frequency,
        rho_obs=rho_obs,
        phase_obs=phase_obs,
        lb=lb,
        ub=ub,
        config=config,
    )

    assert np.isfinite(
        result.best_fitness
    )


def test_agdo_convergence_non_increasing():

    frequency, rho_obs, phase_obs = (
        create_synthetic_data()
    )

    lb = np.array([
        1.0,
        1.0,
        1.0,
        10.0,
        10.0,
    ])

    ub = np.array([
        3000.0,
        3000.0,
        3000.0,
        2000.0,
        2000.0,
    ])

    config = AGDOConfig(
        npop=10,
        niter=10,
        seed=42,
    )

    result = agdo_mt(
        frequency=frequency,
        rho_obs=rho_obs,
        phase_obs=phase_obs,
        lb=lb,
        ub=ub,
        config=config,
    )

    difference = np.diff(
        result.convergence
    )

    assert np.all(
        difference <= 1e-12
    )