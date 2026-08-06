"""Prepare raw NASA rasters into the derived arrays referenced by config.yaml.

Reproject/crop/downsample DEM + ortho + CRISM to a common grid (CTX-scale to stay light on
disk); write data/derived/*. Record product IDs in data/raw/PROVENANCE.md.

TODO(bob): implement (task B5). Keep full rasters in data/raw/ (git-ignored).
"""
from __future__ import annotations


def prepare_all(config_path: str = "config.yaml") -> None:
    """Produce jezero_dem.npy, jezero_slope.npy, jezero_ortho.png,
    jezero_science_value.npy, dust_tau_series.csv."""
    raise NotImplementedError("TODO(bob): raw -> derived data pipeline")


if __name__ == "__main__":
    prepare_all()
