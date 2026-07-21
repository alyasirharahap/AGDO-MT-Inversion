"""
Custom exceptions for Yasir AGDO-MT.
"""


class MTError(Exception):
    """Base exception for Yasir AGDO-MT."""


class InvalidModelError(MTError):
    """Raised when the resistivity model is invalid."""


class InvalidFrequencyError(MTError):
    """Raised when the frequency array is invalid."""