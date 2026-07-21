"""
Example 5
---------

Customize AGDO optimization parameters.

This example demonstrates how to configure
the AGDO optimizer using AGDOConfig.
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

from yasir_agdo_mt.optimizers.agdo import (
    AGDOConfig,
)

# ----------------------------------------------------
# 1. Define synthetic model
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
# 2. Generate synthetic responses
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
# 3. Define parameter bounds
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
# 4. Configure AGDO
# ----------------------------------------------------

config = AGDOConfig(
    npop=200,
    niter=300,
    learning_rate=0.05,
    beta1=0.9,
    beta2=0.999,
    tolerance=1e-3,
    patience=25,
    seed=42,
    verbose=True,
)

# ----------------------------------------------------
# 5. Run inversion
# ----------------------------------------------------

result = invert_mt1d(
    frequency=frequency,
    rho_obs=rho_obs,
    phase_obs=phase_obs,
    lb=lb,
    ub=ub,
    config=config,
)

# ----------------------------------------------------
# 6. Display results
# ----------------------------------------------------

print("=" * 60)
print("AGDO Configuration")
print("=" * 60)

print(config)

print()

print("=" * 60)
print("Inversion Result")
print("=" * 60)

print(f"Best fitness : {result.best_fitness:.6f}")
print(f"Iterations   : {result.iterations}")
print(f"Converged    : {result.converged}")

print("\nEstimated Resistivity (Ohm.m)")
print(result.resistivity)

print("\nEstimated Thickness (m)")
print(result.thickness)

print("\nExample completed successfully.")