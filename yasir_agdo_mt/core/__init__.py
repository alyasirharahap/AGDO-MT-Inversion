"""
Core functionality for one-dimensional
magnetotelluric modelling and inversion.
"""

from .forward import (
    mt1d,
    apparent_resistivity,
    phase,
)

from .objective import (
    rmse_mt,
    evaluate_mt_model,
)

from .inversion import (
    MTInversionResult,
    invert_mt1d,
)


__all__ = [
    "mt1d",
    "apparent_resistivity",
    "phase",
    "rmse_mt",
    "evaluate_mt_model",
    "MTInversionResult",
    "invert_mt1d",
]