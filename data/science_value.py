"""Data-driven science value from CRISM mineralogy (see docs/DATA.md).

Turns 'why is this sample worth it?' into a real number from orbital mineral maps
(Jezero is carbonate/phyllosilicate-rich = astrobiology-grade), instead of an arbitrary value.

TODO(bob): implement (task B5).
"""
from __future__ import annotations

from common.types import Target


def load_mineral_map(path: str) -> "np.ndarray":  # noqa: F821
    """Load the DEM-aligned science-value raster derived from CRISM MTRDR."""
    raise NotImplementedError("TODO(bob): load CRISM-derived science-value raster")


def science_value_at(xy: tuple[float, float], mineral_map: "np.ndarray") -> float:  # noqa: F821
    raise NotImplementedError("TODO(bob): sample science value at a world coordinate")


def classify_targets(candidates: list[tuple[float, float]], mineral_map) -> list[Target]:
    """Attach data-derived science_value + mineral_class to candidate locations."""
    raise NotImplementedError("TODO(bob): build Targets with real science values")
