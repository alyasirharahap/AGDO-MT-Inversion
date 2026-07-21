import numpy as np
import pytest

from yasir_agdo_mt.synthetic import (
    LayeredModel,
    create_layered_model,
    add_gaussian_noise,
)


def test_create_layered_model():

    model = create_layered_model(
        resistivity=[
            100,
            10,
            1000,
        ],
        thickness=[
            500,
            1000,
        ],
    )

    assert isinstance(
        model,
        LayeredModel,
    )

    assert model.nlayer == 3

    assert model.resistivity.shape == (3,)
    assert model.thickness.shape == (2,)


def test_model_vector():

    model = create_layered_model(
        resistivity=[
            100,
            10,
            1000,
        ],
        thickness=[
            500,
            1000,
        ],
    )

    expected = np.array([
        100,
        10,
        1000,
        500,
        1000,
    ])

    assert np.allclose(
        model.model_vector,
        expected,
    )


def test_invalid_layer_structure():

    with pytest.raises(ValueError):

        create_layered_model(
            resistivity=[
                100,
                10,
                1000,
            ],
            thickness=[
                500,
            ],
        )


def test_gaussian_noise_shape():

    data = np.ones(20)

    noisy = add_gaussian_noise(
        data,
        noise_level=0.05,
        seed=42,
    )

    assert noisy.shape == data.shape


def test_gaussian_noise_reproducible():

    data = np.ones(20)

    noisy_1 = add_gaussian_noise(
        data,
        noise_level=0.05,
        seed=42,
    )

    noisy_2 = add_gaussian_noise(
        data,
        noise_level=0.05,
        seed=42,
    )

    assert np.allclose(
        noisy_1,
        noisy_2,
    )


def test_zero_noise():

    data = np.array([
        10.0,
        20.0,
        30.0,
    ])

    noisy = add_gaussian_noise(
        data,
        noise_level=0.0,
        seed=42,
    )

    assert np.allclose(
        noisy,
        data,
    )