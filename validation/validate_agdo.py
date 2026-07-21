import numpy as np
import matplotlib.pyplot as plt

from yasir_agdo_mt.optimizers.agdo import (
    AGDOConfig,
    agdo_mt,
)

from yasir_agdo_mt.core.forward import (
    mt1d,
    apparent_resistivity,
    phase,
)


# ==========================================================
# Load synthetic MT data
# ==========================================================

data = np.loadtxt(
    "data/Data_Synthetic_Model_2.txt",
    skiprows=1,
)

frequency = data[:, 0]
rho_obs = data[:, 1]
phase_obs = data[:, 2]

print("=" * 50)
print("Synthetic MT data loaded")
print("Total frequencies :", len(frequency))
print("=" * 50)


# ==========================================================
# Inversion bounds
# ==========================================================

lb = np.array(
    [
        1,
        1,
        1,
        10,
        10,
    ],
    dtype=float,
)

ub = np.array(
    [
        3000,
        3000,
        3000,
        2000,
        2000,
    ],
    dtype=float,
)


# ==========================================================
# AGDO configuration
# ==========================================================

config = AGDOConfig(

    npop=120,

    niter=150,

    learning_rate=0.05,

    beta1=0.9,

    beta2=0.999,

    eps=1e-8,

    levy_beta=1.5,

    seed=42,

    verbose=True,

)


# ==========================================================
# Run inversion
# ==========================================================

result = agdo_mt(

    frequency=frequency,

    rho_obs=rho_obs,

    phase_obs=phase_obs,

    lb=lb,

    ub=ub,

    config=config,

)

print("\n========== FINAL RESULT ==========")

print("\nBest model")

print(result.best_model)

print("\nBest fitness")

print(result.best_fitness)


# ==========================================================
# Forward response of best model
# ==========================================================

nlayer = (len(result.best_model) + 1) // 2

resistivities = result.best_model[:nlayer]
thicknesses = result.best_model[nlayer:]

# Calculate complex MT impedance
Z = mt1d(
    resistivity=resistivities,
    thickness=thicknesses,
    frequency=frequency,
)

# Convert impedance to apparent resistivity
rho_cal = apparent_resistivity(
    Z,
    frequency,
)

# Calculate phase
phase_cal = phase(Z)


# ==========================================================
# Apparent resistivity
# ==========================================================

plt.figure(figsize=(10, 6))

plt.loglog(
    1 / frequency,
    rho_obs,
    "ob",
    label="Observed",
)

plt.loglog(
    1 / frequency,
    rho_cal,
    "r-",
    linewidth=2,
    label="Calculated",
)

plt.xlabel("Period (s)")

plt.ylabel("Apparent Resistivity (Ohm.m)")

plt.title("AGDO Inversion")

plt.legend()

plt.grid(True, which="both")

plt.show()


# ==========================================================
# Phase
# ==========================================================

plt.figure(figsize=(10, 6))

plt.semilogx(
    1 / frequency,
    phase_obs,
    "ob",
    label="Observed",
)

plt.semilogx(
    1 / frequency,
    phase_cal,
    "r-",
    linewidth=2,
    label="Calculated",
)

plt.xlabel("Period (s)")

plt.ylabel("Phase (deg)")

plt.title("AGDO Inversion")

plt.legend()

plt.grid(True)

plt.show()


# ==========================================================
# Convergence
# ==========================================================

plt.figure(figsize=(8, 5))

plt.plot(
    result.convergence,
    linewidth=2,
)

plt.xlabel("Iteration")

plt.ylabel("RMSE (%)")

plt.title("Convergence Curve")

plt.grid(True)

plt.show()