import numpy as np

from yasir_agdo_mt.core.objective import rmse_mt


def test_rmse_mt():

    rho_obs = np.array([100.0, 200.0, 300.0])
    phase_obs = np.array([45.0, 50.0, 55.0])

    rho_cal = np.array([102.0, 198.0, 301.0])
    phase_cal = np.array([44.5, 50.5, 54.8])

    rmse = rmse_mt(
        rho_obs,
        phase_obs,
        rho_cal,
        phase_cal,
    )

    assert isinstance(rmse, float)
    assert rmse >= 0