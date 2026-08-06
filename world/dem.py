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


def load_dem(path: str) -> tuple[np.ndarray, dict]:
    """Load a prepared DEM (.npy for now; rasterio GeoTIFF path added when real data lands)."""
    if path.endswith(".npy"):
        dem = np.load(path)
        return dem, {"meters_per_cell": 6.0, "source": path}
    raise NotImplementedError("TODO(bob/later): GeoTIFF via rasterio in data/prepare.py")


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
