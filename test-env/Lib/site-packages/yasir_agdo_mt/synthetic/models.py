from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class LayeredModel:
    """
    One-dimensional layered-earth resistivity model.

    Parameters
    ----------
    resistivity : np.ndarray
        Electrical resistivity of each layer in ohm.m.

    thickness : np.ndarray
        Thickness of each layer in meters, excluding
        the bottom half-space.
    """

    resistivity: np.ndarray
    thickness: np.ndarray

    def __post_init__(self) -> None:

        self.resistivity = np.asarray(
            self.resistivity,
            dtype=float,
        )

        self.thickness = np.asarray(
            self.thickness,
            dtype=float,
        )

        if self.resistivity.ndim != 1:
            raise ValueError(
                "resistivity must be one-dimensional."
            )

        if self.thickness.ndim != 1:
            raise ValueError(
                "thickness must be one-dimensional."
            )

        if self.resistivity.size != self.thickness.size + 1:
            raise ValueError(
                "The number of resistivity values must be "
                "one greater than the number of thickness values."
            )

        if np.any(self.resistivity <= 0):
            raise ValueError(
                "All resistivity values must be positive."
            )

        if np.any(self.thickness <= 0):
            raise ValueError(
                "All thickness values must be positive."
            )

    @property
    def nlayer(self) -> int:
        """
        Number of layers including the bottom half-space.
        """

        return self.resistivity.size

    @property
    def model_vector(self) -> np.ndarray:
        """
        Return the model as a single parameter vector.

        Structure:
        [rho_1, ..., rho_n, h_1, ..., h_(n-1)]
        """

        return np.concatenate(
            (
                self.resistivity,
                self.thickness,
            )
        )


def create_layered_model(
    resistivity: np.ndarray | list[float],
    thickness: np.ndarray | list[float],
) -> LayeredModel:
    """
    Create a validated one-dimensional layered-earth model.
    """

    return LayeredModel(
        resistivity=np.asarray(
            resistivity,
            dtype=float,
        ),
        thickness=np.asarray(
            thickness,
            dtype=float,
        ),
    )