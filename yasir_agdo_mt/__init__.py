"""
Yasir AGDO-MT

A Python package for one-dimensional magnetotelluric
forward modelling and inversion using the
Adam Gradient Descent Optimizer (AGDO).
"""

from .version import __version__

from .core import (
    mt1d,
    apparent_resistivity,
    phase,
    rmse_mt,
    evaluate_mt_model,
    MTInversionResult,
    invert_mt1d,
)

from .optimizers import (
    AGDOConfig,
    AGDOResult,
    agdo_mt,
)


__all__ = [
    "__version__",
    "mt1d",
    "apparent_resistivity",
    "phase",
    "rmse_mt",
    "evaluate_mt_model",
    "AGDOConfig",
    "AGDOResult",
    "agdo_mt",
    "MTInversionResult",
    "invert_mt1d",
    "plot_mt_response",
    "plot_model",
    "plot_convergence",
]