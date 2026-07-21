"""
Input and output utilities for magnetotelluric data.
"""

from .edi import (
    EDIData,
    EDIResponse,
    read_edi,
    edi_response,
)


__all__ = [
    "EDIData",
    "EDIResponse",
    "read_edi",
    "edi_response",
]