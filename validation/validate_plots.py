import matplotlib.pyplot as plt
import numpy as np

from yasir_agdo_mt.visualization import (
    plot_convergence,
    plot_model,
    plot_mt_response,
)

frequency = np.logspace(-3, 3, 40)

rho_obs = np.linspace(80, 600, 40)
rho_cal = rho_obs * 1.03

phase_obs = np.linspace(15, 70, 40)
phase_cal = phase_obs + 0.8

plot_mt_response(
    frequency,
    rho_obs,
    phase_obs,
    rho_cal,
    phase_cal,
    title="Synthetic MT Response",
    calculated_label="AGDO",
)

plot_model(
    [100, 20, 900],
    [300, 700],
    title="Recovered Resistivity Model",
    label="AGDO",
    bottom_factor=2.0,
)

plot_convergence(
    np.exp(-np.linspace(0,5,100)),
    title="AGDO Convergence",
    ylabel="RMSE",
)

plt.show()