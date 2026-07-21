from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

from ..core.forward import (
    apparent_resistivity,
    phase,
)


@dataclass(slots=True)
class EDIData:
    """
    Magnetotelluric data read from an EDI file.
    """

    station: str
    frequency: np.ndarray

    zxx: np.ndarray | None = None
    zxy: np.ndarray | None = None
    zyx: np.ndarray | None = None
    zyy: np.ndarray | None = None

    latitude: str | None = None
    longitude: str | None = None
    elevation: float | None = None


def _read_numeric_block(
    text: str,
    block_name: str,
) -> np.ndarray | None:
    """
    Read a numeric data block from EDI text.
    """

    pattern = (
        rf">{re.escape(block_name)}"
        rf"[^\n]*\n"
        rf"(.*?)(?=\n>)"
    )

    match = re.search(
        pattern,
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if match is None:
        return None

    block = match.group(1)

    values = np.fromstring(
        block,
        sep=" ",
        dtype=float,
    )

    return values


def _read_header_value(
    text: str,
    key: str,
) -> str | None:
    """
    Read a value from the EDI header.
    """

    pattern = (
        rf"^\s*{re.escape(key)}\s*=\s*(.+?)\s*$"
    )

    match = re.search(
        pattern,
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )

    if match is None:
        return None

    return match.group(1).strip().strip('"')


def _build_complex_impedance(
    real: np.ndarray | None,
    imag: np.ndarray | None,
) -> np.ndarray | None:
    """
    Combine real and imaginary impedance components.
    """

    if real is None or imag is None:
        return None

    if real.shape != imag.shape:
        raise ValueError(
            "Real and imaginary impedance components "
            "must have identical shapes."
        )

    return real + 1j * imag


def read_edi(
    filename: str | Path,
) -> EDIData:
    """
    Read magnetotelluric data from a SEG EDI file.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the EDI file.

    Returns
    -------
    EDIData
        Parsed EDI data including frequency and
        available impedance tensor components.
    """

    path = Path(filename)

    if not path.exists():
        raise FileNotFoundError(
            f"EDI file not found: {path}"
        )

    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    # ------------------------------------------------------
    # Metadata
    # ------------------------------------------------------

    station = (
        _read_header_value(text, "DATAID")
        or path.stem
    )

    latitude = _read_header_value(
        text,
        "LAT",
    )

    longitude = _read_header_value(
        text,
        "LONG",
    )

    elevation_text = _read_header_value(
        text,
        "ELEV",
    )

    elevation = (
        float(elevation_text)
        if elevation_text is not None
        else None
    )

    # ------------------------------------------------------
    # Frequency
    # ------------------------------------------------------

    frequency = _read_numeric_block(
        text,
        "FREQ",
    )

    if frequency is None or frequency.size == 0:
        raise ValueError(
            "No FREQ block found in EDI file."
        )

    # ------------------------------------------------------
    # Impedance components
    # ------------------------------------------------------

    zxx = _build_complex_impedance(
        _read_numeric_block(text, "ZXXR"),
        _read_numeric_block(text, "ZXXI"),
    )

    zxy = _build_complex_impedance(
        _read_numeric_block(text, "ZXYR"),
        _read_numeric_block(text, "ZXYI"),
    )

    zyx = _build_complex_impedance(
        _read_numeric_block(text, "ZYXR"),
        _read_numeric_block(text, "ZYXI"),
    )

    zyy = _build_complex_impedance(
        _read_numeric_block(text, "ZYYR"),
        _read_numeric_block(text, "ZYYI"),
    )

    # ------------------------------------------------------
    # Validate component lengths
    # ------------------------------------------------------

    for name, component in (
        ("ZXX", zxx),
        ("ZXY", zxy),
        ("ZYX", zyx),
        ("ZYY", zyy),
    ):

        if (
            component is not None
            and component.size != frequency.size
        ):
            raise ValueError(
                f"{name} contains {component.size} values, "
                f"but FREQ contains {frequency.size}."
            )

    return EDIData(
        station=station,
        frequency=frequency,
        zxx=zxx,
        zxy=zxy,
        zyx=zyx,
        zyy=zyy,
        latitude=latitude,
        longitude=longitude,
        elevation=elevation,
    )

@dataclass(slots=True)
class EDIResponse:
    """
    Magnetotelluric response derived from an EDI impedance component.

    Attributes
    ----------
    frequency : np.ndarray
        Frequency values in Hz.

    apparent_resistivity : np.ndarray
        Apparent resistivity in ohm.m.

    phase : np.ndarray
        Impedance phase in degrees.

    component : str
        Impedance tensor component used to calculate the response.
    """

    frequency: np.ndarray
    apparent_resistivity: np.ndarray
    phase: np.ndarray
    component: str

def edi_response(
    data: EDIData,
    component: str = "zxy",
) -> EDIResponse:
    """
    Calculate apparent resistivity and phase from
    an impedance component stored in EDI data.

    Parameters
    ----------
    data : EDIData
        Parsed EDI data returned by read_edi().

    component : str, default="zxy"
        Impedance tensor component to use.

        Supported components are:
        "zxx", "zxy", "zyx", and "zyy".

    Returns
    -------
    EDIResponse
        Frequency, apparent resistivity, phase,
        and the selected impedance component.
    """

    component = component.lower()

    valid_components = {
        "zxx",
        "zxy",
        "zyx",
        "zyy",
    }

    if component not in valid_components:
        raise ValueError(
            "component must be one of: "
            "'zxx', 'zxy', 'zyx', or 'zyy'."
        )

    impedance = getattr(
        data,
        component,
    )

    if impedance is None:
        raise ValueError(
            f"Impedance component "
            f"{component.upper()} "
            "is not available in the EDI file."
        )

    rho = apparent_resistivity(
        impedance,
        data.frequency,
    )

    ph = phase(
        impedance,
    )

    return EDIResponse(
        frequency=data.frequency.copy(),
        apparent_resistivity=rho,
        phase=ph,
        component=component,
    )