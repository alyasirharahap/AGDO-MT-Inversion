"""
Example 2
---------

Read a SEG-EDI file and inspect its contents.
"""

from pathlib import Path

from yasir_agdo_mt.io.edi import read_edi
from yasir_agdo_mt.core.forward import (
    apparent_resistivity,
    phase,
)

# ----------------------------------------------------
# Locate data file
# ----------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

edi_file = ROOT / "data" / "edi" / "L09S10_edt.edi"

# ----------------------------------------------------
# Read EDI
# ----------------------------------------------------

edi = read_edi(edi_file)

# ----------------------------------------------------
# Compute MT responses from Zxy
# ----------------------------------------------------

rho = apparent_resistivity(
    edi.zxy,
    edi.frequency,
)

phi = phase(edi.zxy)

# ----------------------------------------------------
# Display information
# ----------------------------------------------------

print("=" * 60)
print("EDI Information")
print("=" * 60)

print(f"Station              : {edi.station}")
print(f"Latitude             : {edi.latitude}")
print(f"Longitude            : {edi.longitude}")
print(f"Elevation            : {edi.elevation}")

print()

print(f"Number of frequencies : {len(edi.frequency)}")

print()

print("First five frequencies")
print(edi.frequency[:5])

print()

print("First five apparent resistivities")
print(rho[:5])

print()

print("First five phases")
print(phi[:5])

print("\nExample completed successfully.")