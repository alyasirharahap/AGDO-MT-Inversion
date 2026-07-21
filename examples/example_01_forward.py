"""
Example 1
---------

Forward modelling of a layered-earth MT model.

This example computes the MT response
(apparent resistivity and phase)
from a three-layer Earth model.
"""

import numpy as np

from yasir_agdo_mt.core.forward import (
    mt1d,
    apparent_resistivity,
    phase,
)

# -----------------------------------------
# Define layered-earth model
# -----------------------------------------

resistivity = np.array([
    100,
    10,
    500,
])

thickness = np.array([
    200,
    800,
])

frequency = np.logspace(
    -3,
    4,
    56,
)

# -----------------------------------------
# Forward modelling
# -----------------------------------------

Z = mt1d(
    resistivity=resistivity,
    thickness=thickness,
    frequency=frequency,
)

rho = apparent_resistivity(
    Z,
    frequency,
)

phi = phase(Z)

# -----------------------------------------
# Results
# -----------------------------------------

print("Apparent resistivity")
print(rho)

print()

print("Phase")
print(phi)