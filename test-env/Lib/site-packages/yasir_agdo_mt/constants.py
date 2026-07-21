"""
Physical constants used throughout Yasir AGDO-MT.
"""

import numpy as np

# Vacuum magnetic permeability (H/m)
MU0 = 4 * np.pi * 1e-7

# Numerical tolerance
EPSILON = 1e-12

# Default optimizer parameters
DEFAULT_BETA1 = 0.9
DEFAULT_BETA2 = 0.999