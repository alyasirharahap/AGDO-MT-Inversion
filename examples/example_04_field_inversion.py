"""
Example 4
---------

Invert field MT data from a SEG-EDI file.
"""

from pathlib import Path
import numpy as np

from yasir_agdo_mt.io.edi import read_edi
from yasir_agdo_mt.core.forward import (
    apparent_resistivity,
    phase,
)
from yasir_agdo_mt.core.inversion import invert_mt1d

# ----------------------------------------------------
# Locate data file
# ----------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

edi_file = ROOT / "data" / "edi" / "L09S10_edt.edi"

# ----------------------------------------------------
# Read EDI file
# ----------------------------------------------------

edi = read_edi(edi_file)

# ----------------------------------------------------
# Compute MT responses from impedance (Zxy)
# ----------------------------------------------------

rho_obs = apparent_resistivity(
    edi.zxy,
    edi.frequency,
)

phase_obs = phase(
    edi.zxy,
)

# ----------------------------------------------------
# Define inversion bounds
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
    frequency=edi.frequency,
    rho_obs=rho_obs,
    phase_obs=phase_obs,
    lb=lb,
    ub=ub,
)

# ----------------------------------------------------
# Display results
# ----------------------------------------------------

print("=" * 60)
print("Field MT Inversion")
print("=" * 60)

print(f"Station        : {edi.station}")
print(f"Best fitness   : {result.best_fitness:.6f}")
print(f"Iterations     : {result.iterations}")
print(f"Converged      : {result.converged}")

print("\nEstimated Resistivity (Ohm.m)")
print(result.resistivity)

print("\nEstimated Thickness (m)")
print(result.thickness)

print("\nExample completed successfully.")