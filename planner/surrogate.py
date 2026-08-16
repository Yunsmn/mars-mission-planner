"""The surrogate simulator — the planner's cheap 'imagination'.

Runs a candidate action sequence n times under injected uncertainty to expose tail risk.
Fully vectorized over all n rollouts (one batched NumPy pass, NO python loop over rollouts),
no rendering, no contact solve. Target: sub-millisecond for n=20, horizon=100.

This is NOT MuJoCo. MuJoCo (world/scene.py) is the real world; this is a reduced model:
path integrated over the DEM slope map, energy = f(dist, slope, payload),
slip/hazard = f(slope, roughness) with probability of getting stuck rising past a threshold.
Uncertainty per rollout: traction, localization drift, battery-draw multiplier.

SCALE: Real world is ±5m, targets ~1-2m away, battery in %, ~0.6 Wh/m base cost.

Authored by IBM Bob for the MARVIN mission planner.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common.types import ActionSeq, MissionState, RolloutBatch

# Real-world scale constants (match world/sim.py and rover/energy.py)
TERRAIN_RADIUS = 5.0  # world spans [-5, 5] m
BASE_WH_PER_M = 0.6   # flat-ground energy cost
SLOPE_WH_PER_M_PER_DEG = 0.05
SAMPLE_ENERGY_WH = 0.4  # battery % cost per sample
SCAN_ENERGY_WH = 0.1
OBSERVE_ENERGY_WH = 0.05


@dataclass(frozen=True)
class SurrogateEnv:
    """Lightweight terrain model for fast rollouts."""
    terrain: np.ndarray      # Coarse elevation grid (16-24 cells) — used for fast hazard rollouts
    meters_per_cell: float   # Grid resolution
    dust_tau: float
    traction_range: tuple[float, float]
    loc_drift_range: tuple[float, float]
    draw_mult_range: tuple[float, float]
    terrain_full: np.ndarray | None = None   # Full-res height grid, for A* pathing (finer than rollouts)
    mpc_full: float = 0.0


def rollout_batch(
    seq: ActionSeq,
    state: MissionState,
    env: SurrogateEnv,
    n: int,
    rng: np.random.Generator
) -> RolloutBatch:
    """Simulate `seq` from `state` n times under uncertainty. Returns per-rollout arrays.
    
    Vectorized implementation: all n rollouts computed in parallel using NumPy broadcasting.
    
    Args:
        seq: Action sequence to simulate
        state: Current mission state (pose, battery, etc.)
        env: Environment parameters (terrain, uncertainty ranges)
        n: Number of rollouts to perform
        rng: Random number generator for uncertainty injection
    
    Returns:
        RolloutBatch with success, energy, and hazard arrays of shape (n,)
    """
    # Initialize arrays for all rollouts (vectorized)
    positions = np.tile(state.pose.xy, (n, 1))  # (n, 2)
    headings = np.full(n, state.pose.heading_rad)  # (n,)
    battery_pct = np.full(n, state.battery_pct)  # (n,) - battery as percentage
    loc_uncertainty = np.full(n, state.localization_sigma)  # (n,)
    
    # Sample uncertainty parameters for each rollout
    traction_mult = rng.uniform(
        env.traction_range[0],
        env.traction_range[1],
        size=n
    )
    loc_drift_rate = rng.uniform(
        env.loc_drift_range[0],
        env.loc_drift_range[1],
        size=n
    )
    draw_mult = rng.uniform(
        env.draw_mult_range[0],
        env.draw_mult_range[1],
        size=n
    )
    
    # Track outcomes
    success = np.ones(n, dtype=bool)
    energy_used_pct = np.zeros(n)  # Track as battery percentage
    peak_hazard = np.zeros(n)
    
    # Simulate each action in the sequence
    for action in seq:
        if action.kind.name == 'DRIVE':
            target_xy = action.params.get('xy')
            if target_xy is None:
                success[:] = False
                continue
            
            # Clip target to terrain bounds
            target = np.array(target_xy)
            target = np.clip(target, -TERRAIN_RADIUS, TERRAIN_RADIUS)
            
            # Calculate distance for all rollouts
            delta = target - positions
            distances = np.linalg.norm(delta, axis=1)  # (n,)
            
            # Sample terrain along path using bilinear interpolation
            slopes = sample_terrain_along_path(positions, target, env)  # (n,)
            
            # Hazard model: increases sharply with slope
            # Base hazard from slope (normalized to [0, 1])
            slope_hazard = np.clip(slopes / 25.0, 0, 1)  # 25° = high risk
            
            # Combined hazard with traction uncertainty
            hazard = slope_hazard * (2.0 - traction_mult)  # poor traction increases hazard
            hazard = np.clip(hazard, 0, 1)
            
            # Update peak hazard
            peak_hazard = np.maximum(peak_hazard, hazard)
            
            # Failure probability increases with hazard
            failure_prob = hazard ** 2  # quadratic increase
            failures = rng.random(n) < failure_prob
            success[failures] = False
            
            # Energy model: matches rover/energy.py
            # Base energy per meter (0.6 Wh/m)
            per_m = BASE_WH_PER_M + SLOPE_WH_PER_M_PER_DEG * np.maximum(0, slopes)
            
            # Total energy with uncertainty (in Wh, then convert to %)
            energy_wh = distances * per_m * draw_mult
            energy_pct = energy_wh  # Nominal capacity ~100 Wh, so Wh ≈ %
            
            energy_used_pct += energy_pct
            battery_pct -= energy_pct
            
            # Update positions (with localization drift)
            drift = rng.normal(0, loc_drift_rate[:, np.newaxis], size=(n, 2))
            positions = target + drift
            
            # Clip positions to terrain bounds
            positions = np.clip(positions, -TERRAIN_RADIUS, TERRAIN_RADIUS)
            
            # Update localization uncertainty
            loc_uncertainty += distances * 0.01  # grows with distance traveled
            
            # Check battery depletion
            depleted = battery_pct <= 0
            success[depleted] = False
            
        elif action.kind.name == 'SAMPLE':
            # Sampling action: fixed energy cost (battery %)
            sample_energy_pct = SAMPLE_ENERGY_WH * draw_mult
            energy_used_pct += sample_energy_pct
            battery_pct -= sample_energy_pct
            
            # Small failure probability (instrument issues)
            sample_failures = rng.random(n) < 0.05
            success[sample_failures] = False
            
        elif action.kind.name == 'SCAN':
            # Scanning action: small energy cost
            scan_energy_pct = SCAN_ENERGY_WH * draw_mult
            energy_used_pct += scan_energy_pct
            battery_pct -= scan_energy_pct
            
        elif action.kind.name == 'OBSERVE':
            # Observation action: minimal energy
            obs_energy_pct = OBSERVE_ENERGY_WH * draw_mult
            energy_used_pct += obs_energy_pct
            battery_pct -= obs_energy_pct
            
        elif action.kind.name == 'HOLD':
            # Hold: no action, no energy
            pass
    
    return RolloutBatch(
        success=success,
        energy=energy_used_pct,  # Return as battery % used
        hazard=peak_hazard
    )


def _slope_at(points: np.ndarray, env: SurrogateEnv) -> np.ndarray:
    """Bilinear-interpolated terrain slope (degrees) at world-coordinate points, shape (n,)."""
    grid_size = env.terrain.shape[0]
    grid_x = np.clip((points[:, 0] + TERRAIN_RADIUS) / (2 * TERRAIN_RADIUS) * (grid_size - 1),
                     0, grid_size - 1)
    grid_y = np.clip((points[:, 1] + TERRAIN_RADIUS) / (2 * TERRAIN_RADIUS) * (grid_size - 1),
                     0, grid_size - 1)
    x0 = np.floor(grid_x).astype(int)
    x1 = np.minimum(x0 + 1, grid_size - 1)
    y0 = np.floor(grid_y).astype(int)
    y1 = np.minimum(y0 + 1, grid_size - 1)
    # local gradient from the coarse grid at each point's cell
    dx = (env.terrain[y0, x1] - env.terrain[y0, x0]) / env.meters_per_cell
    dy = (env.terrain[y1, x0] - env.terrain[y0, x0]) / env.meters_per_cell
    return np.degrees(np.arctan(np.sqrt(dx ** 2 + dy ** 2)))


def sample_terrain_along_path(
    start_positions: np.ndarray,
    target: np.ndarray,
    env: SurrogateEnv,
    n_samples: int = 6,
) -> np.ndarray:
    """Worst-case terrain slope (degrees) along each path, sampled at several points.

    Sampling the whole segment — not just the midpoint — is what lets the lightsim tell a route that
    *skirts* a dune from one that *clips* it: a single-point check misses the crossing entirely.

    Args:
        start_positions: (n, 2) array of starting positions
        target: (2,) target position
        env: SurrogateEnv with terrain grid
        n_samples: points sampled along each path (max slope over them is returned)

    Returns:
        (n,) array of worst-case slope values in degrees
    """
    target = np.asarray(target, dtype=float)
    worst = np.zeros(len(start_positions))
    for t in np.linspace(1.0 / n_samples, 1.0, n_samples):     # skip t=0 (the start point itself)
        points = start_positions * (1.0 - t) + target * t
        worst = np.maximum(worst, _slope_at(points, env))
    return worst


def create_surrogate_env(world, cfg: dict) -> SurrogateEnv:
    """Create a SurrogateEnv from the real world terrain.
    
    Args:
        world: MarsSim instance with terrain
        cfg: Configuration dict with uncertainty ranges
    
    Returns:
        SurrogateEnv ready for rollout_batch
    """
    # Create coarse terrain grid (16x16 or 24x24) from full DEM
    # This is the "lightweight point cloud" derived from the real DEM
    full_terrain = world.terrain
    target_size = cfg.get('surrogate_grid_size', 16)
    
    # Downsample terrain while preserving local highs/lows
    coarse_terrain = downsample_terrain(full_terrain, target_size)
    
    meters_per_cell = 2 * TERRAIN_RADIUS / (target_size - 1)
    
    # Default uncertainty ranges (calibrated from MuJoCo)
    traction_range = cfg.get('traction_range', (0.8, 1.2))
    loc_drift_range = cfg.get('loc_drift_range', (0.05, 0.15))
    draw_mult_range = cfg.get('draw_mult_range', (0.9, 1.1))
    
    return SurrogateEnv(
        terrain=coarse_terrain,
        meters_per_cell=meters_per_cell,
        dust_tau=world.dust_tau,
        traction_range=traction_range,
        loc_drift_range=loc_drift_range,
        draw_mult_range=draw_mult_range,
        terrain_full=full_terrain,
        mpc_full=2 * TERRAIN_RADIUS / (full_terrain.shape[0] - 1),
    )


def downsample_terrain(terrain: np.ndarray, target_size: int) -> np.ndarray:
    """Downsample terrain to target_size while preserving features.
    
    Uses max pooling to preserve local highs (important for hazard detection).
    
    Args:
        terrain: Full resolution terrain (n x n)
        target_size: Target grid size (e.g., 16 or 24)
    
    Returns:
        Downsampled terrain (target_size x target_size)
    """
    from scipy.ndimage import maximum_filter
    
    current_size = terrain.shape[0]
    if current_size == target_size:
        return terrain
    
    # Calculate pooling window size
    pool_size = current_size // target_size
    
    # Apply max pooling to preserve peaks
    pooled = maximum_filter(terrain, size=pool_size)
    
    # Subsample
    indices = np.linspace(0, current_size - 1, target_size, dtype=int)
    coarse = pooled[np.ix_(indices, indices)]
    
    return coarse
