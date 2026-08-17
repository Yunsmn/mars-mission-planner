"""Terrain: heightfield generation/loading + slope, and conversion to a MuJoCo hfield.

For now `synthesize` builds a gentle procedural Mars-like terrain so the sim runs without a
download. `load_dem` (real HiRISE/CTX DEM) drops in later via the same array contract — the
rest of the sim doesn't care where the grid came from. See docs/DATA.md.
"""
from __future__ import annotations

import numpy as np


def synthesize(n: int = 64, amplitude_m: float = 0.22, seed: int = 42) -> np.ndarray:
    """A gentle, long-wavelength terrain (meters), n x n — drivable by a basic rover.

    Long-wavelength dunes + heavily smoothed noise keep slopes low (~<12°) so wheel-terrain
    contact stays stable. Real Mars DEMs drop in later via load_dem().
    """
    rng = np.random.default_rng(seed)
    xs = np.linspace(0, 2 * np.pi, n)           # low frequency = gentle slopes
    gx, gy = np.meshgrid(xs, xs)
    dunes = 0.6 * np.sin(gx) * np.cos(0.7 * gy) + 0.4 * np.sin(0.5 * gx + 1.0)
    noise = rng.standard_normal((n, n))
    for _ in range(10):                         # heavy smoothing -> rolling, not jagged
        noise = (
            noise
            + np.roll(noise, 1, 0) + np.roll(noise, -1, 0)
            + np.roll(noise, 1, 1) + np.roll(noise, -1, 1)
        ) / 5.0
    field = dunes + 0.5 * noise
    field -= field.min()
    if field.max() > 0:
        field *= amplitude_m / field.max()
    return field.astype(np.float64)


def rolling(n: int = 64, seed: int = 7, relief_m: float = 0.6) -> np.ndarray:
    """Smooth, varied Mars-like terrain (meters), n x n: a field of rounded hills and hollows —
    ground that goes up AND down, not a wall in front of the rover. Built from a handful of Gaussian
    features of mixed sign, width, and height, then smoothed, so some rises are gentle (cheap to
    drive over) and a few are steep (worth going around). Fully procedural from `seed`.
    """
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    r = 5.0                                        # world half-extent (matches TERRAIN_RADIUS)
    xs = np.linspace(-r, r, n)
    xx, yy = np.meshgrid(xs, xs)
    z = np.zeros((n, n))
    for _ in range(9):                             # hills (+) and hollows (-) of varied size
        cx, cy = rng.uniform(-4.2, 4.2, size=2)
        amp = rng.uniform(-1.0, 1.0)
        sig = rng.uniform(0.8, 2.4)                # small sigma + big amp => a steep feature
        z += amp * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sig ** 2)))
    z = gaussian_filter(z, sigma=1.0)              # keep it rounded, never jagged
    z -= z.min()
    if z.max() > 0:
        z *= relief_m / z.max()
    return z.astype(np.float64)


def load_dem(path: str, meters_per_cell: float = 6.0) -> tuple[np.ndarray, dict]:
    """Load a prepared DEM (.npy or GeoTIFF).
    
    Args:
        path: Path to DEM file (.npy or .tif)
        meters_per_cell: Resolution in meters per cell
    
    Returns:
        Tuple of (dem array, metadata dict)
    """
    import os
    
    if not os.path.exists(path):
        # Fall back to synthetic if file doesn't exist
        import logging
        logging.getLogger(__name__).warning(f"DEM not found at {path}, using synthetic")
        dem = synthesize()
        return dem, {"meters_per_cell": meters_per_cell, "source": "synthetic"}
    
    if path.endswith(".npy"):
        dem = np.load(path)
        return dem, {"meters_per_cell": meters_per_cell, "source": path}
    
    elif path.endswith((".tif", ".tiff")):
        try:
            import rasterio
            with rasterio.open(path) as src:
                dem = src.read(1)
                transform = src.transform
                actual_res = abs(transform[0])
                return dem, {"meters_per_cell": actual_res, "source": path}
        except ImportError:
            raise ImportError("rasterio required for GeoTIFF. Install: pip install rasterio")
    
    raise ValueError(f"Unsupported DEM format: {path}")


def slope_map(dem: np.ndarray, meters_per_cell: float) -> np.ndarray:
    """Per-cell slope in degrees (gradient magnitude of the elevation grid)."""
    dzdy, dzdx = np.gradient(dem, meters_per_cell)
    return np.degrees(np.arctan(np.hypot(dzdx, dzdy)))


def dem_to_hfield(dem: np.ndarray) -> dict:
    """Normalized elevation buffer (0..1) + vertical scale for a MuJoCo <hfield>."""
    lo, hi = float(dem.min()), float(dem.max())
    span = max(hi - lo, 1e-6)
    normalized = ((dem - lo) / span).astype(np.float64)
    return {"nrow": dem.shape[0], "ncol": dem.shape[1],
            "elevation_m": span, "data": normalized}
