"""
Forward modelling for One-Dimensional Magnetotellurics (MT).

This module contains functions used to compute the MT response
of horizontally layered Earth models.

Author
------
Muhammad Fadhilah Al Yasir Harahap

Project
-------
Yasir AGDO-MT
"""

from __future__ import annotations

import numpy as np

from ..constants import MU0


def intrinsic_impedance(
    angular_frequency: float | np.ndarray,
    resistivity: float,
    mu: float = MU0,
) -> complex | np.ndarray:
    """
    Compute the intrinsic impedance of a homogeneous layer.

    Parameters
    ----------
    angular_frequency : float or ndarray
        Angular frequency (rad/s).

    resistivity : float
        Electrical resistivity (Ohm.m).

    mu : float, optional
        Magnetic permeability.
        Default is the free-space permeability (MU0).

    Returns
    -------
    complex or ndarray
        Intrinsic impedance.
    """

    return np.sqrt(1j * angular_frequency * mu * resistivity)


def propagation_constant(
    angular_frequency: float | np.ndarray,
    resistivity: float,
    mu: float = MU0,
) -> complex | np.ndarray:
    """
    Compute the propagation constant.

    Parameters
    ----------
    angular_frequency : float or ndarray

    resistivity : float

    mu : float

    Returns
    -------
    complex or ndarray
    """

    return np.sqrt(1j * angular_frequency * mu / resistivity)


def reflection_coefficient(
    lower_impedance: complex,
    upper_impedance: complex,
) -> complex:
    """
    Compute the reflection coefficient between two layers.

    Parameters
    ----------
    lower_impedance : complex

    upper_impedance : complex

    Returns
    -------
    complex
    """

    return (lower_impedance - upper_impedance) / (
        lower_impedance + upper_impedance
    )


def recursive_impedance(
    intrinsic: complex,
    reflection: complex,
    propagation: complex,
    thickness: float,
) -> complex:
    """
    Compute recursive impedance.

    Parameters
    ----------
    intrinsic : complex

    reflection : complex

    propagation : complex

    thickness : float

    Returns
    -------
    complex
    """

    exponential = np.exp(-2 * propagation * thickness)

    return intrinsic * (
        (1 + reflection * exponential)
        /
        (1 - reflection * exponential)
    )

def mt1d(
    resistivity: np.ndarray,
    thickness: np.ndarray,
    frequency: np.ndarray,
) -> np.ndarray:
    """
    Compute the complex surface impedance of a 1D layered Earth model.

    Parameters
    ----------
    resistivity : ndarray
        Layer resistivities (Ohm.m).

    thickness : ndarray
        Layer thicknesses (m). The last layer is assumed to be a half-space.

    frequency : ndarray
        Frequencies (Hz).

    Returns
    -------
    ndarray
        Complex surface impedance.
    """

    omega = 2 * np.pi * np.asarray(frequency)

    resistivity = np.asarray(resistivity, dtype=float)
    thickness = np.asarray(thickness, dtype=float)

    nlayer = len(resistivity)

    impedance = np.zeros(len(frequency), dtype=complex)

    for i, w in enumerate(omega):

        # Half-space impedance
        z = intrinsic_impedance(
            angular_frequency=w,
            resistivity=resistivity[-1],
        )

        # Recursive upward calculation
        for j in range(nlayer - 2, -1, -1):

            z0 = intrinsic_impedance(
                angular_frequency=w,
                resistivity=resistivity[j],
            )

            k = propagation_constant(
                angular_frequency=w,
                resistivity=resistivity[j],
            )

            r = reflection_coefficient(
                lower_impedance=z,
                upper_impedance=z0,
            )

            z = recursive_impedance(
                intrinsic=z0,
                reflection=r,
                propagation=k,
                thickness=thickness[j],
            )

        impedance[i] = z

    return impedance

def apparent_resistivity(
    impedance: np.ndarray,
    frequency: np.ndarray,
    mu: float = MU0,
) -> np.ndarray:
    """
    Compute apparent resistivity from the MT impedance.

    Parameters
    ----------
    impedance : ndarray
        Complex surface impedance.

    frequency : ndarray
        Frequencies in Hz.

    mu : float, optional
        Magnetic permeability (default: MU0).

    Returns
    -------
    ndarray
        Apparent resistivity (Ohm·m).
    """

    frequency = np.asarray(frequency, dtype=float)
    impedance = np.asarray(impedance, dtype=complex)

    omega = 2 * np.pi * frequency

    return np.abs(impedance) ** 2 / (mu * omega)

def phase(
    impedance: np.ndarray,
) -> np.ndarray:
    """
    Compute MT phase from the complex impedance.

    Parameters
    ----------
    impedance : ndarray
        Complex surface impedance.

    Returns
    -------
    ndarray
        Phase in degrees.
    """

    impedance = np.asarray(impedance, dtype=complex)

    return np.degrees(np.angle(impedance))