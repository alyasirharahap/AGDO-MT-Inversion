"""
Utilities for generating synthetic
magnetotelluric models and data.
"""

from .models import (
    LayeredModel,
    create_layered_model,
)

from .noise import (
    add_gaussian_noise,
)


__all__ = [
    "LayeredModel",
    "create_layered_model",
    "add_gaussian_noise",
]