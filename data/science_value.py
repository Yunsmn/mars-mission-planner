"""Science value mapping from CRISM mineralogy to target science scores.

Maps CRISM mineral detections (carbonate, phyllosilicate, olivine) to science value scores
that drive the planner's target prioritization. High carbonate/clay = high astrobiology
potential = high science value.

See docs/DATA.md for CRISM data sources.

Authored by IBM Bob for the MARVIN mission planner.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def load_science_map(path: str | Path) -> np.ndarray:
    """Load the prepared science value map.
    
    Args:
        path: Path to jezero_science_value.npy
    
    Returns:
        Science value grid (0-1 scale)
    """
    path = Path(path)
    if not path.exists():
        logger.warning(f"Science value map not found at {path}, using default")
        return np.full((64, 64), 0.5, dtype=np.float32)
    
    science_map = np.load(path)
    logger.info(f"Loaded science value map: {science_map.shape}, "
               f"range [{science_map.min():.2f}, {science_map.max():.2f}]")
    return science_map


def get_science_value(
    xy: tuple[float, float],
    science_map: np.ndarray,
    world_radius: float = 5.0
) -> float:
    """Get science value at a world location from the CRISM-derived map.
    
    Args:
        xy: World coordinates (meters, in [-world_radius, world_radius])
        science_map: The loaded science value grid
        world_radius: World extent in meters (default: 5.0)
    
    Returns:
        Science value at that location (0-1 scale)
    """
    x, y = xy
    n = science_map.shape[0]
    
    # Convert world coords to grid indices
    j = int(np.clip((x + world_radius) / (2 * world_radius) * (n - 1), 0, n - 1))
    i = int(np.clip((y + world_radius) / (2 * world_radius) * (n - 1), 0, n - 1))
    
    return float(science_map[i, j])


def classify_mineral(science_value: float) -> str:
    """Classify mineral type based on science value.
    
    This is a simplified mapping for display purposes.
    Real CRISM data would provide actual mineral classes.
    
    Args:
        science_value: Science value score (0-1)
    
    Returns:
        Mineral class string
    """
    if science_value > 0.7:
        return "carbonate"  # High astrobiology potential
    elif science_value > 0.5:
        return "phyllosilicate"  # Clay minerals
    elif science_value > 0.3:
        return "olivine"  # Igneous
    else:
        return "basalt"  # Common background