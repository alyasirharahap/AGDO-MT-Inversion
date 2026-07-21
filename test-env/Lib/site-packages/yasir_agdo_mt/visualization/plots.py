from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def _validate_1d(
    array: Sequence[float] | np.ndarray,
    name: str,
) -> np.ndarray:
    """
    Validate a one-dimensional numeric array.
    """

    array = np.asarray(array, dtype=float)

    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")

    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")

    return array


def plot_mt_response(
    frequency,
    rho_obs,
    phase_obs,
    rho_cal=None,
    phase_cal=None,
    *,
    figsize=(7, 8),
    title=None,
    observed_label="Observed",
    calculated_label="Calculated",
    axes=None,
):
    """
    Plot MT apparent resistivity and phase responses.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : tuple(matplotlib.axes.Axes, matplotlib.axes.Axes)
    """

    frequency = _validate_1d(frequency, "frequency")
    rho_obs = _validate_1d(rho_obs, "rho_obs")
    phase_obs = _validate_1d(phase_obs, "phase_obs")

    if len(frequency) != len(rho_obs):
        raise ValueError("frequency and rho_obs must have equal length.")

    if len(frequency) != len(phase_obs):
        raise ValueError("frequency and phase_obs must have equal length.")

    if rho_cal is not None:
        rho_cal = _validate_1d(rho_cal, "rho_cal")

        if len(rho_cal) != len(frequency):
            raise ValueError("rho_cal has incorrect length.")

    if phase_cal is not None:
        phase_cal = _validate_1d(phase_cal, "phase_cal")

        if len(phase_cal) != len(frequency):
            raise ValueError("phase_cal has incorrect length.")

    if axes is None:
        fig, axes = plt.subplots(
            2,
            1,
            figsize=figsize,
            sharex=True,
        )
    else:
        fig = axes[0].figure

    ax_rho, ax_phase = axes

    ax_rho.loglog(
        frequency,
        rho_obs,
        "o",
        label=observed_label,
    )

    if rho_cal is not None:
        ax_rho.loglog(
            frequency,
            rho_cal,
            "-",
            linewidth=2,
            label=calculated_label,
        )

    ax_rho.set_ylabel("Apparent Resistivity (Ωm)")
    ax_rho.grid(True, which="both")
    ax_rho.legend()

    ax_phase.semilogx(
        frequency,
        phase_obs,
        "o",
        label=observed_label,
    )

    if phase_cal is not None:
        ax_phase.semilogx(
            frequency,
            phase_cal,
            "-",
            linewidth=2,
            label=calculated_label,
        )

    ax_phase.set_xlabel("Frequency (Hz)")
    ax_phase.set_ylabel("Phase (°)")
    ax_phase.grid(True, which="both")
    ax_phase.legend()

    if title is not None:
        fig.suptitle(title)

    fig.tight_layout()

    return fig, (ax_rho, ax_phase)


def plot_model(
    resistivity,
    thickness,
    *,
    ax=None,
    label=None,
    title=None,
    bottom_factor=1.5,
):
    """
    Plot layered resistivity model.
    """

    resistivity = _validate_1d(resistivity, "resistivity")
    thickness = _validate_1d(thickness, "thickness")

    if len(resistivity) != len(thickness) + 1:
        raise ValueError(
            "Number of resistivities must equal number of thicknesses + 1."
        )

    if bottom_factor <= 1:
        raise ValueError(
            "bottom_factor must be greater than 1."
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 7))
    else:
        fig = ax.figure

    depth = np.concatenate(
        (
            [0.0],
            np.cumsum(thickness),
        )
    )

    bottom_depth = depth[-1] * bottom_factor

    depth_plot = np.append(
        depth,
        bottom_depth,
    )

    rho_plot = np.append(
        resistivity,
        resistivity[-1],
    )

    ax.step(
        rho_plot,
        depth_plot,
        where="post",
        linewidth=2,
        label=label,
    )

    ax.set_xscale("log")
    ax.invert_yaxis()

    ax.set_xlabel("Resistivity (Ωm)")
    ax.set_ylabel("Depth (m)")
    ax.grid(True, which="both")

    if label is not None:
        ax.legend()

    if title is not None:
        ax.set_title(title)

    return fig, ax


def plot_convergence(
    convergence,
    *,
    ax=None,
    title=None,
    ylabel="Objective Function",
):
    """
    Plot optimization convergence curve.
    """

    convergence = _validate_1d(
        convergence,
        "convergence",
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    ax.plot(
        np.arange(
            1,
            len(convergence) + 1,
        ),
        convergence,
        linewidth=2,
    )

    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)

    ax.grid(True)

    if title is not None:
        ax.set_title(title)

    return fig, ax