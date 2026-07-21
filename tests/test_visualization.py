from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.axes
import matplotlib.figure
import numpy as np
import pytest

from yasir_agdo_mt.visualization import (
    plot_convergence,
    plot_model,
    plot_mt_response,
)


@pytest.fixture
def sample_mt_data():

    frequency = np.logspace(-3, 3, 30)

    rho_obs = np.linspace(100, 500, 30)

    phase_obs = np.linspace(20, 70, 30)

    rho_cal = rho_obs * 1.02

    phase_cal = phase_obs + 0.5

    return (
        frequency,
        rho_obs,
        phase_obs,
        rho_cal,
        phase_cal,
    )


@pytest.fixture
def sample_model():

    resistivity = np.array(
        [100.0, 20.0, 800.0]
    )

    thickness = np.array(
        [300.0, 700.0]
    )

    return resistivity, thickness


def test_plot_mt_response(sample_mt_data):

    (
        frequency,
        rho_obs,
        phase_obs,
        rho_cal,
        phase_cal,
    ) = sample_mt_data

    fig, axes = plot_mt_response(
        frequency,
        rho_obs,
        phase_obs,
        rho_cal,
        phase_cal,
    )

    assert isinstance(
        fig,
        matplotlib.figure.Figure,
    )

    assert len(axes) == 2

    for ax in axes:
        assert isinstance(
            ax,
            matplotlib.axes.Axes,
        )


def test_plot_model(sample_model):

    resistivity, thickness = sample_model

    fig, ax = plot_model(
        resistivity,
        thickness,
    )

    assert isinstance(
        fig,
        matplotlib.figure.Figure,
    )

    assert isinstance(
        ax,
        matplotlib.axes.Axes,
    )


def test_plot_convergence():

    convergence = np.linspace(
        1,
        0.05,
        100,
    )

    fig, ax = plot_convergence(
        convergence,
    )

    assert isinstance(
        fig,
        matplotlib.figure.Figure,
    )

    assert isinstance(
        ax,
        matplotlib.axes.Axes,
    )


def test_invalid_mt_length():

    with pytest.raises(
        ValueError
    ):

        plot_mt_response(
            np.arange(10),
            np.arange(9),
            np.arange(10),
        )


def test_invalid_model():

    with pytest.raises(
        ValueError
    ):

        plot_model(
            [100, 10],
            [100, 200],
        )


def test_empty_convergence():

    with pytest.raises(
        ValueError
    ):

        plot_convergence([])

def test_invalid_bottom_factor():

    with pytest.raises(ValueError):

        plot_model(
            [100, 10, 500],
            [100, 200],
            bottom_factor=1.0,
        )

def test_custom_labels(sample_mt_data):

    (
        frequency,
        rho_obs,
        phase_obs,
        rho_cal,
        phase_cal,
    ) = sample_mt_data

    fig, axes = plot_mt_response(
        frequency,
        rho_obs,
        phase_obs,
        rho_cal,
        phase_cal,
        observed_label="Observed",
        calculated_label="AGDO",
    )

    legend = axes[0].get_legend()

    labels = [
        text.get_text()
        for text in legend.get_texts()
    ]

    assert "Observed" in labels

    assert "AGDO" in labels