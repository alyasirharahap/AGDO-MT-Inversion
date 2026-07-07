import numpy as np

from yasir_agdo_mt.core.forward import intrinsic_impedance

omega = 2 * np.pi

rho = 100

Z = intrinsic_impedance(omega, rho)

print(Z)