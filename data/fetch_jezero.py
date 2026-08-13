"""Fetch a real Jezero-region DEM from the global MOLA mosaic and prepare it for the sim.

Reads only the Jezero *window* from NASA/USGS's global MOLA 463 m/px mosaic via GDAL's
/vsicurl/ HTTP range requests — no multi-GB download. The terrain *morphology* is real Jezero;
the amplitude is normalized to a rover-drivable range (the sim patch is abstracted to ~10 m, so
absolute relief is scaled, not the shape). Provenance is recorded next to the output.

Run:  .venv/bin/python -m data.fetch_jezero
"""
from __future__ import annotations

import json
import math
import os

os.environ.setdefault("GDAL_HTTP_UNSAFESSL", "YES")
os.environ.setdefault("CPL_VSIL_CURL_USE_HEAD", "YES")

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds

MOLA = "/vsicurl/https://planetarymaps.usgs.gov/mosaic/Mars_MGS_MOLA_DEM_mosaic_global_463m.tif"
MARS_RADIUS_M = 3396190.0
JEZERO_LON_DEG, JEZERO_LAT_DEG = 77.58, 18.44   # Jezero delta region (E, N)
HALF_WINDOW_M = 30000.0                          # 60 km box (captures crater + rim)
GRID = 64
DRIVABLE_AMPLITUDE_M = 0.22


def fetch(out_npy: str = "data/derived/jezero_dem.npy") -> str:
    # Equirectangular Mars: easting = R*lon_rad, northing = R*lat_rad (central meridian 0).
    cx = MARS_RADIUS_M * math.radians(JEZERO_LON_DEG)
    cy = MARS_RADIUS_M * math.radians(JEZERO_LAT_DEG)
    with rasterio.open(MOLA) as src:
        win = from_bounds(cx - HALF_WINDOW_M, cy - HALF_WINDOW_M,
                          cx + HALF_WINDOW_M, cy + HALF_WINDOW_M, src.transform)
        data = src.read(1, window=win, out_shape=(GRID, GRID),
                        resampling=Resampling.bilinear).astype(np.float64)

    real_lo, real_hi = float(data.min()), float(data.max())
    field = data - data.min()
    if field.max() > 0:
        field *= DRIVABLE_AMPLITUDE_M / field.max()

    os.makedirs(os.path.dirname(out_npy) or ".", exist_ok=True)
    np.save(out_npy, field)

    prov = {
        "source": "NASA/USGS MGS MOLA global DEM 463 m/px",
        "url": "https://planetarymaps.usgs.gov/mosaic/Mars_MGS_MOLA_DEM_mosaic_global_463m.tif",
        "region": "Jezero crater",
        "center_lon_e_deg": JEZERO_LON_DEG,
        "center_lat_n_deg": JEZERO_LAT_DEG,
        "window_km": 2 * HALF_WINDOW_M / 1000.0,
        "grid": GRID,
        "real_elevation_range_m": [round(real_lo, 1), round(real_hi, 1)],
        "note": "Real Jezero morphology; amplitude normalized to a rover-drivable range.",
    }
    with open(os.path.join(os.path.dirname(out_npy), "jezero_dem_provenance.json"), "w") as f:
        json.dump(prov, f, indent=2)

    print(f"real Jezero MOLA window: elevation {real_lo:.0f}..{real_hi:.0f} m "
          f"(relief {real_hi - real_lo:.0f} m) over {2 * HALF_WINDOW_M / 1000:.0f} km")
    print(f"normalized to {DRIVABLE_AMPLITUDE_M} m amplitude, {GRID}x{GRID} -> {out_npy}")
    return out_npy


if __name__ == "__main__":
    fetch()
