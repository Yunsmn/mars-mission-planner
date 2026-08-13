"""Prepare real Mars data for MARVIN — DEM, science value, dust opacity.

This script downloads/processes raw orbital data into the derived arrays the planner uses:
- jezero_dem.npy: elevation grid (meters) → MuJoCo hfield
- jezero_slope.npy: slope map (degrees) → hazard/energy
- jezero_ortho.png: orbital image → visualization
- jezero_science_value.npy: CRISM-derived science scores per cell
- dust_tau_series.csv: MEDA/REMS dust opacity time series

See docs/DATA.md for data sources and provenance.

Usage:
    python -m data.prepare --dem <path_to_geotiff> --output-dir data/derived

Authored by IBM Bob for the MARVIN mission planner.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_dem(
    input_path: str,
    output_dir: str,
    target_size: int = 64,
    meters_per_cell: float = 6.0
) -> None:
    """Process a GeoTIFF DEM into MARVIN's format.
    
    Args:
        input_path: Path to input GeoTIFF DEM
        output_dir: Directory for output files
        target_size: Target grid size (will downsample/crop)
        meters_per_cell: Target resolution in meters per cell
    """
    try:
        import rasterio
        from rasterio.enums import Resampling
        from scipy.ndimage import gaussian_filter
    except ImportError:
        logger.error("rasterio not installed. Install with: pip install rasterio")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading DEM from {input_path}")
    
    with rasterio.open(input_path) as src:
        # Read the DEM
        dem = src.read(1)
        
        # Get original resolution
        transform = src.transform
        original_res = abs(transform[0])  # meters per pixel
        
        logger.info(f"Original DEM: {dem.shape}, resolution: {original_res:.2f} m/px")
        
        # Calculate downsample factor
        downsample_factor = int(meters_per_cell / original_res)
        if downsample_factor < 1:
            downsample_factor = 1
        
        # Downsample if needed
        if downsample_factor > 1:
            new_shape = (dem.shape[0] // downsample_factor, 
                        dem.shape[1] // downsample_factor)
            
            # Use rasterio's resampling
            dem_downsampled = src.read(
                1,
                out_shape=new_shape,
                resampling=Resampling.average
            )
            dem = dem_downsampled
            logger.info(f"Downsampled to {dem.shape}, ~{meters_per_cell:.1f} m/px")
        
        # Crop to target size (center crop)
        if dem.shape[0] > target_size or dem.shape[1] > target_size:
            center_y, center_x = dem.shape[0] // 2, dem.shape[1] // 2
            half = target_size // 2
            dem = dem[
                center_y - half:center_y + half,
                center_x - half:center_x + half
            ]
            logger.info(f"Cropped to {dem.shape}")
        
        # Smooth slightly to reduce noise
        dem = gaussian_filter(dem, sigma=0.5)
        
        # Save DEM
        dem_path = output_path / "jezero_dem.npy"
        np.save(dem_path, dem.astype(np.float64))
        logger.info(f"Saved DEM to {dem_path}")
        
        # Calculate and save slope
        from world.dem import slope_map
        slope = slope_map(dem, meters_per_cell)
        slope_path = output_path / "jezero_slope.npy"
        np.save(slope_path, slope.astype(np.float32))
        logger.info(f"Saved slope map to {slope_path}")
        logger.info(f"Slope range: {slope.min():.1f}° to {slope.max():.1f}°")
        
        # Save metadata
        metadata = {
            "source": input_path,
            "shape": dem.shape,
            "meters_per_cell": meters_per_cell,
            "elevation_range_m": (float(dem.min()), float(dem.max())),
            "slope_range_deg": (float(slope.min()), float(slope.max()))
        }
        
        import json
        metadata_path = output_path / "jezero_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")


def prepare_science_value(
    crism_path: str | None,
    dem_shape: tuple[int, int],
    output_dir: str
) -> None:
    """Process CRISM mineralogy into science value map.
    
    Args:
        crism_path: Path to CRISM MTRDR or summary parameter map (optional)
        dem_shape: Shape to match (from DEM)
        output_dir: Directory for output files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if crism_path is None:
        # Generate synthetic science value map for now
        logger.info("No CRISM data provided, generating synthetic science value map")
        
        # Create a map with some high-value regions
        science_map = np.random.RandomState(42).uniform(0.3, 0.7, dem_shape)
        
        # Add some high-value "mineral deposits"
        for _ in range(3):
            cy = np.random.randint(10, dem_shape[0] - 10)
            cx = np.random.randint(10, dem_shape[1] - 10)
            y, x = np.ogrid[:dem_shape[0], :dem_shape[1]]
            dist = np.sqrt((y - cy)**2 + (x - cx)**2)
            science_map += 0.3 * np.exp(-dist**2 / 50)
        
        science_map = np.clip(science_map, 0.0, 1.0)
    else:
        # TODO: Load and process real CRISM data
        logger.info(f"Loading CRISM data from {crism_path}")
        raise NotImplementedError("Real CRISM processing not yet implemented")
    
    science_path = output_path / "jezero_science_value.npy"
    np.save(science_path, science_map.astype(np.float32))
    logger.info(f"Saved science value map to {science_path}")
    logger.info(f"Science value range: {science_map.min():.2f} to {science_map.max():.2f}")


def prepare_dust_data(output_dir: str) -> None:
    """Generate dust opacity time series.
    
    For now, generates a synthetic series. Real MEDA/REMS data integration TBD.
    
    Args:
        output_dir: Directory for output files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Generating synthetic dust opacity time series")
    
    # Generate a realistic dust tau series
    # Baseline ~0.5, with seasonal variation and occasional dust events
    sols = np.arange(0, 100)
    
    # Baseline with seasonal variation
    tau = 0.5 + 0.2 * np.sin(sols * 2 * np.pi / 50)
    
    # Add some dust events
    dust_events = [20, 60]
    for event_sol in dust_events:
        event_profile = 1.5 * np.exp(-((sols - event_sol)**2) / 20)
        tau += event_profile
    
    # Add noise
    tau += np.random.RandomState(42).normal(0, 0.05, len(sols))
    tau = np.clip(tau, 0.1, 3.0)
    
    # Write CSV manually (no pandas dependency)
    dust_path = output_path / "dust_tau_series.csv"
    with open(dust_path, 'w') as f:
        f.write("sol,dust_tau,source\n")
        for sol, tau_val in zip(sols, tau):
            f.write(f"{sol},{tau_val:.4f},synthetic\n")
    
    logger.info(f"Saved dust time series to {dust_path}")
    logger.info(f"Tau range: {tau.min():.2f} to {tau.max():.2f}")


def main():
    parser = argparse.ArgumentParser(description="Prepare Mars data for MARVIN")
    parser.add_argument("--dem", help="Path to input DEM GeoTIFF")
    parser.add_argument("--crism", help="Path to CRISM data (optional)")
    parser.add_argument("--output-dir", default="data/derived", 
                       help="Output directory for processed data")
    parser.add_argument("--target-size", type=int, default=64,
                       help="Target grid size (default: 64x64)")
    parser.add_argument("--meters-per-cell", type=float, default=6.0,
                       help="Target resolution in meters per cell (default: 6.0)")
    
    args = parser.parse_args()
    
    if args.dem:
        prepare_dem(args.dem, args.output_dir, args.target_size, args.meters_per_cell)
        
        # Load DEM shape for science value map
        dem = np.load(Path(args.output_dir) / "jezero_dem.npy")
        prepare_science_value(args.crism, dem.shape, args.output_dir)
    else:
        logger.info("No DEM provided, generating synthetic data only")
        # Generate synthetic data with default size
        prepare_science_value(None, (64, 64), args.output_dir)
    
    prepare_dust_data(args.output_dir)
    
    logger.info("Data preparation complete!")


if __name__ == "__main__":
    main()