"""
Example 3
---------

One-dimensional MT inversion using synthetic data.

This example:

1. Creates a synthetic MT dataset
2. Adds no noise (for simplicity)
3. Runs AGDO inversion
4. Prints the estimated model
"""

import numpy as np

from yasir_agdo_mt.core.forward import (
    mt1d,
    apparent_resistivity,
    phase,
)

from yasir_agdo_mt.core.inversion import (
    invert_mt1d,
)

# ----------------------------------------------------
# True layered-earth model
# ----------------------------------------------------

true_resistivity = np.array([
    100,
    10,
    500,
])

true_thickness = np.array([
    200,
    800,
])

frequency = np.logspace(
    -3,
    4,
    56,
)

# ----------------------------------------------------
# Generate synthetic MT responses
# ----------------------------------------------------

Z = mt1d(
    resistivity=true_resistivity,
    thickness=true_thickness,
    frequency=frequency,
)

rho_obs = apparent_resistivity(
    Z,
    frequency,
)

phase_obs = phase(Z)

# ----------------------------------------------------
# Parameter bounds
# ----------------------------------------------------

lb = np.array([
    1,
    1,
    1,
    10,
    10,
])

ub = np.array([
    3000,
    3000,
    3000,
    2000,
    2000,
])

# ----------------------------------------------------
# Run inversion
# ----------------------------------------------------

result = invert_mt1d(
    frequency=frequency,
    rho_obs=rho_obs,
    phase_obs=phase_obs,
    lb=lb,
    ub=ub,
)

# ----------------------------------------------------
# Results
# ----------------------------------------------------

print("\nEstimated Resistivity (Ohm.m)")
print(result.resistivity)

print("\nEstimated Thickness (m)")
print(result.thickness)

print(f"\nBest Fitness : {result.best_fitness:.6f}")

print(f"Iterations   : {result.iterations}")

print(f"Converged    : {result.converged}")

print("\nExample completed successfully.")