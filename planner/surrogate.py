"""The surrogate simulator — the planner's cheap 'imagination'.

Runs a candidate action sequence n times under injected uncertainty to expose tail risk.
Fully vectorized over all n rollouts (one batched NumPy pass, NO python loop over rollouts),
no rendering, no contact solve. Target: sub-millisecond for n=20, horizon=100.

This is NOT MuJoCo. MuJoCo (world/scene.py) is the real world; this is a reduced model:
path integrated over the DEM slope map, energy = f(dist, slope, payload),
slip/hazard = f(slope, roughness) with probability of getting stuck rising past a threshold.
Uncertainty per rollout: traction, localization drift, battery-draw multiplier.

Authored by IBM Bob for the MARVIN mission planner.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common.types import ActionSeq, MissionState, RolloutBatch


@dataclass(frozen=True)
class SurrogateEnv:
    slope_deg: np.ndarray
    roughness: np.ndarray
    dust_tau: float
    traction_range: tuple[float, float]
    loc_drift_range: tuple[float, float]
    draw_mult_range: tuple[float, float]


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
        env: Environment parameters (slope map, uncertainty ranges)
        n: Number of rollouts to perform
        rng: Random number generator for uncertainty injection
    
    Returns:
        RolloutBatch with success, energy, and hazard arrays of shape (n,)
    """
    # Initialize arrays for all rollouts (vectorized)
    positions = np.tile(state.pose.xy, (n, 1))  # (n, 2)
    headings = np.full(n, state.pose.heading_rad)  # (n,)
    battery_wh = np.full(n, state.battery_pct * 10.0)  # rough: 1% = 10 Wh
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
    energy_used = np.zeros(n)
    peak_hazard = np.zeros(n)
    
    # Simulate each action in the sequence
    for action in seq:
        if action.kind.name == 'DRIVE':
            target_xy = action.params.get('xy')
            if target_xy is None:
                success[:] = False
                continue
            
            # Vectorized drive simulation
            target = np.array(target_xy)
            
            # Calculate distance for all rollouts
            delta = target - positions
            distances = np.linalg.norm(delta, axis=1)  # (n,)
            
            # Sample terrain along path (simplified: use midpoint)
            midpoints = (positions + target) / 2
            
            # Get slope and roughness at midpoints (with bounds checking)
            h, w = env.slope_deg.shape
            mid_i = np.clip(midpoints[:, 1].astype(int), 0, h - 1)
            mid_j = np.clip(midpoints[:, 0].astype(int), 0, w - 1)
            
            slopes = env.slope_deg[mid_i, mid_j]  # (n,)
            rough = env.roughness[mid_i, mid_j]  # (n,)
            
            # Hazard model: increases sharply with slope
            # Base hazard from slope (normalized to [0, 1])
            slope_hazard = np.clip(slopes / 30.0, 0, 1)  # 30° = max safe slope
            
            # Roughness adds to hazard
            roughness_hazard = np.clip(rough / 0.5, 0, 0.3)  # roughness contributes up to 0.3
            
            # Combined hazard with traction uncertainty
            hazard = slope_hazard + roughness_hazard
            hazard = hazard * (2.0 - traction_mult)  # poor traction increases hazard
            hazard = np.clip(hazard, 0, 1)
            
            # Update peak hazard
            peak_hazard = np.maximum(peak_hazard, hazard)
            
            # Failure probability increases with hazard
            failure_prob = hazard ** 2  # quadratic increase
            failures = rng.random(n) < failure_prob
            success[failures] = False
            
            # Energy model: base + slope penalty + distance
            # Base energy per meter
            base_energy_per_m = 5.0  # Wh/m
            
            # Slope penalty (uphill costs more)
            slope_factor = 1.0 + np.clip(slopes / 15.0, 0, 2)  # up to 3x for steep slopes
            
            # Roughness penalty
            rough_factor = 1.0 + rough * 0.5
            
            # Total energy with uncertainty
            energy = distances * base_energy_per_m * slope_factor * rough_factor * draw_mult
            energy_used += energy
            battery_wh -= energy
            
            # Update positions (with localization drift)
            drift = rng.normal(0, loc_drift_rate[:, np.newaxis], size=(n, 2))
            positions = target + drift
            
            # Update localization uncertainty
            loc_uncertainty += distances * 0.01  # grows with distance traveled
            
            # Check battery depletion
            depleted = battery_wh <= 0
            success[depleted] = False
            
        elif action.kind.name == 'SAMPLE':
            # Sampling action: fixed energy cost, no movement
            sample_energy = 20.0  # Wh per sample
            energy_used += sample_energy * draw_mult
            battery_wh -= sample_energy * draw_mult
            
            # Small failure probability (instrument issues)
            sample_failures = rng.random(n) < 0.05
            success[sample_failures] = False
            
        elif action.kind.name == 'SCAN':
            # Scanning action: small energy cost
            scan_energy = 5.0  # Wh per scan
            energy_used += scan_energy * draw_mult
            battery_wh -= scan_energy * draw_mult
            
        elif action.kind.name == 'OBSERVE':
            # Observation action: minimal energy
            obs_energy = 2.0  # Wh
            energy_used += obs_energy * draw_mult
            battery_wh -= obs_energy * draw_mult
            
        elif action.kind.name == 'HOLD':
            # Hold: no action, no energy
            pass
    
    return RolloutBatch(
        success=success,
        energy=energy_used,
        hazard=peak_hazard
    )


def create_surrogate_env(perception, cfg: dict) -> SurrogateEnv:
    """Create a SurrogateEnv from current perception and config.
    
    Args:
        perception: Current perception with slope, roughness, dust_tau
        cfg: Configuration dict with uncertainty ranges
    
    Returns:
        SurrogateEnv ready for rollout_batch
    """
    # Default uncertainty ranges (can be calibrated from MuJoCo)
    traction_range = cfg.get('traction_range', (0.7, 1.3))
    loc_drift_range = cfg.get('loc_drift_range', (0.1, 0.5))
    draw_mult_range = cfg.get('draw_mult_range', (0.9, 1.2))
    
    return SurrogateEnv(
        slope_deg=perception.slope_deg,
        roughness=perception.roughness,
        dust_tau=perception.dust_tau,
        traction_range=traction_range,
        loc_drift_range=loc_drift_range,
        draw_mult_range=draw_mult_range
    )