from pathlib import Path

import numpy as np
import pytest

from yasir_agdo_mt.io import (
    EDIData,
    EDIResponse,
    read_edi,
    edi_response,
)


@pytest.fixture
def edi_file():
    """
    Return the path to the example EDI file.
    """

    root = Path(__file__).resolve().parents[1]

    return root / "data" / "edi" / "L09S10_edt.edi"


def test_read_edi_returns_data(edi_file):

    data = read_edi(edi_file)

    assert isinstance(
        data,
        EDIData,
    )


def test_edi_frequency(edi_file):

    data = read_edi(edi_file)

    assert data.frequency.ndim == 1

    assert data.frequency.size > 0

    assert np.all(
        np.isfinite(data.frequency)
    )

    assert np.all(
        data.frequency > 0
    )


def test_edi_impedance_components(edi_file):

    data = read_edi(edi_file)

    components = [
        data.zxx,
        data.zxy,
        data.zyx,
        data.zyy,
    ]

    available = [
        component
        for component in components
        if component is not None
    ]

    assert len(available) > 0

    for component in available:

        assert (
            component.shape
            == data.frequency.shape
        )

        assert np.iscomplexobj(
            component
        )


def test_edi_station(edi_file):

    data = read_edi(edi_file)

    assert isinstance(
        data.station,
        str,
    )

    assert len(data.station) > 0


def test_edi_file_not_found():

    with pytest.raises(
        FileNotFoundError
    ):

        read_edi(
            "file_that_does_not_exist.edi"
        )

def test_edi_response_returns_data(edi_file):

    data = read_edi(edi_file)

    response = edi_response(
        data,
        component="zxy",
    )

    assert isinstance(
        response,
        EDIResponse,
    )


def test_edi_response_shape(edi_file):

    data = read_edi(edi_file)

    response = edi_response(
        data,
        component="zxy",
    )

    assert (
        response.frequency.shape
        == data.frequency.shape
    )

    assert (
        response.apparent_resistivity.shape
        == data.frequency.shape
    )

    assert (
        response.phase.shape
        == data.frequency.shape
    )


def test_edi_response_is_finite(edi_file):

    data = read_edi(edi_file)

    response = edi_response(
        data,
        component="zxy",
    )

    assert np.all(
        np.isfinite(
            response.apparent_resistivity
        )
    )

    assert np.all(
        np.isfinite(
            response.phase
        )
    )


def test_invalid_edi_component(edi_file):

    data = read_edi(edi_file)

    with pytest.raises(
        ValueError
    ):

        edi_response(
            data,
            component="invalid",
        )