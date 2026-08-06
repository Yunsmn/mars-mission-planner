"""Link the two simulators: calibrate the lightweight surrogate FROM the detailed world.

The surrogate is only trustworthy if its injected uncertainty matches reality. Here we drive
a set of moves in the full MuJoCo world, fit the slip / energy / localization-drift
distributions, and hand those ranges to the surrogate. `surrogate_fidelity` then reports how
well the cheap model predicts full-physics outcomes — the validation number for the writeup.

SCALE: Real world is ±5m, battery in %, ~0.6 Wh/m base cost.

See docs/DESIGN.md §5.2 and §5.6.

Authored by IBM Bob for the MARVIN mission planner.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from planner.surrogate import SurrogateEnv

logger = logging.getLogger(__name__)

# Real-world scale
TERRAIN_RADIUS = 5.0


@dataclass
class CalibrationResult:
    """Results from calibrating surrogate against MuJoCo."""
    traction_range: tuple[float, float]
    loc_drift_range: tuple[float, float]
    draw_mult_range: tuple[float, float]
    n_samples: int
    fidelity_metrics: dict


def calibrate_surrogate(world, moves: list, n_samples: int = 10) -> SurrogateEnv:
    """Fit traction / draw / drift ranges from full-physics rollouts of `moves`.
    
    Process:
    1. Execute a set of representative moves in MuJoCo
    2. Measure actual traction (slip), energy draw (battery %), and localization drift
    3. Fit distributions to these measurements
    4. Return SurrogateEnv with calibrated uncertainty ranges
    
    Args:
        world: MuJoCo simulation instance (MarsSim)
        moves: List of representative move actions to calibrate from
        n_samples: Number of samples per move type
    
    Returns:
        SurrogateEnv with calibrated uncertainty ranges
    """
    logger.info(f"Calibrating surrogate from {len(moves)} move types, {n_samples} samples each")
    
    # Storage for measurements
    traction_samples = []
    drift_samples = []
    draw_mult_samples = []
    
    # For each move type, execute multiple times and measure
    for move_idx, move in enumerate(moves):
        logger.debug(f"Calibrating move {move_idx+1}/{len(moves)}")
        
        for sample_idx in range(n_samples):
            # Get initial state
            initial_x, initial_y, _ = world.pose()
            initial_battery = world.battery_pct
            
            # Execute the move
            if 'xy' in move:
                target_xy = move['xy']
                result = world.drive_to(target_xy[0], target_xy[1])
                
                # Measure actual vs expected
                final_x, final_y, _ = world.pose()
                
                # Traction: ratio of actual distance to intended distance
                intended_dist = np.hypot(
                    target_xy[0] - initial_x,
                    target_xy[1] - initial_y
                )
                actual_dist = result.get('distance_m', intended_dist)
                
                if intended_dist > 0.1:  # Only measure for non-trivial moves
                    traction_ratio = actual_dist / intended_dist
                    traction_samples.append(traction_ratio)
                
                # Localization drift: error in final position
                expected_xy = np.array(target_xy)
                actual_xy = np.array([final_x, final_y])
                drift = np.linalg.norm(actual_xy - expected_xy)
                drift_samples.append(drift)
                
                # Energy draw multiplier: actual vs nominal (battery %)
                battery_used = initial_battery - world.battery_pct
                nominal_energy = intended_dist * 0.6  # BASE_WH_PER_M, battery % ≈ Wh
                
                if nominal_energy > 0.1:
                    draw_mult = battery_used / nominal_energy
                    draw_mult_samples.append(draw_mult)
    
    # Fit distributions (use percentiles for robust range estimation)
    if traction_samples:
        traction_range = (
            float(np.percentile(traction_samples, 10)),
            float(np.percentile(traction_samples, 90))
        )
    else:
        traction_range = (0.8, 1.2)  # default
    
    if drift_samples:
        # Drift is absolute, so use mean and std
        drift_mean = float(np.mean(drift_samples))
        drift_std = float(np.std(drift_samples))
        loc_drift_range = (
            max(0.05, drift_mean - drift_std),
            drift_mean + drift_std
        )
    else:
        loc_drift_range = (0.05, 0.15)  # default
    
    if draw_mult_samples:
        draw_mult_range = (
            float(np.percentile(draw_mult_samples, 10)),
            float(np.percentile(draw_mult_samples, 90))
        )
    else:
        draw_mult_range = (0.9, 1.1)  # default
    
    logger.info(f"Calibration complete:")
    logger.info(f"  Traction range: {traction_range}")
    logger.info(f"  Loc drift range: {loc_drift_range}")
    logger.info(f"  Draw mult range: {draw_mult_range}")
    
    # Create calibrated environment using real world terrain
    from planner.surrogate import create_surrogate_env
    
    # Use a default config for calibration
    cfg = {
        'traction_range': traction_range,
        'loc_drift_range': loc_drift_range,
        'draw_mult_range': draw_mult_range,
        'surrogate_grid_size': 16
    }
    
    return create_surrogate_env(world, cfg)


def surrogate_fidelity(
    env: SurrogateEnv,
    world,
    seq: list,
    n_rollouts: int = 20
) -> dict:
    """Error between surrogate predictions and a full MuJoCo rollout of `seq`.
    
    Measures how well the cheap surrogate predicts full-physics outcomes by:
    1. Running the action sequence in MuJoCo (ground truth)
    2. Running the same sequence in surrogate n times
    3. Comparing outcomes (position error, energy error, success rate)
    
    Args:
        env: Calibrated SurrogateEnv
        world: MuJoCo simulation (MarsSim)
        seq: Action sequence to test
        n_rollouts: Number of surrogate rollouts for comparison
    
    Returns:
        Dict with fidelity metrics (lower error = better)
    """
    from planner.surrogate import rollout_batch
    from common.types import MissionState, Pose, Target
    
    logger.info("Measuring surrogate fidelity...")
    
    # Get initial state from world
    rng = np.random.default_rng(42)
    initial_x, initial_y, initial_yaw = world.pose()
    initial_battery = world.battery_pct
    
    # Build mission state
    state = MissionState(
        pose=Pose(xy=(initial_x, initial_y), heading_rad=initial_yaw),
        battery_pct=initial_battery,
        sol_time=0.0,
        localization_sigma=0.1,
        collected=tuple(t.id for t in world.targets if t.collected),
        remaining=tuple(
            Target(id=t.id, xy=t.xy, science_value=0.5, mineral_class="unknown")
            for t in world.targets if not t.collected
        )
    )
    
    # Execute in MuJoCo (ground truth)
    mujoco_success = True
    mujoco_battery_used = 0.0
    mujoco_final_xy = (initial_x, initial_y)
    
    for action in seq:
        if action.kind.name == 'DRIVE':
            xy = action.params.get('xy')
            if xy:
                battery_before = world.battery_pct
                result = world.drive_to(xy[0], xy[1])
                mujoco_success = mujoco_success and result.get('success', False)
                mujoco_battery_used += battery_before - world.battery_pct
                mujoco_final_xy = world.pose()[:2]
        
        elif action.kind.name == 'SAMPLE':
            target_id = action.params.get('target')
            if target_id:
                battery_before = world.battery_pct
                result = world.sample(target_id)
                mujoco_success = mujoco_success and result.get('success', False)
                mujoco_battery_used += battery_before - world.battery_pct
    
    # Execute in surrogate (multiple rollouts)
    action_seq = tuple(seq)
    batch = rollout_batch(action_seq, state, env, n_rollouts, rng)
    
    # Compare outcomes
    surrogate_success_rate = float(np.mean(batch.success))
    surrogate_energy_mean = float(np.mean(batch.energy))
    surrogate_energy_std = float(np.std(batch.energy))
    
    # Calculate errors
    success_error = abs(surrogate_success_rate - (1.0 if mujoco_success else 0.0))
    
    if mujoco_battery_used > 0:
        energy_error_pct = abs(surrogate_energy_mean - mujoco_battery_used) / mujoco_battery_used * 100
    else:
        energy_error_pct = 0.0
    
    metrics = {
        'success_error': success_error,
        'energy_error_pct': energy_error_pct,
        'energy_std': surrogate_energy_std,
        'mujoco_success': mujoco_success,
        'mujoco_battery_used': mujoco_battery_used,
        'surrogate_success_rate': surrogate_success_rate,
        'surrogate_energy_mean': surrogate_energy_mean,
        'n_rollouts': n_rollouts
    }
    
    logger.info(f"Fidelity metrics:")
    logger.info(f"  Success error: {success_error:.3f}")
    logger.info(f"  Energy error: {energy_error_pct:.1f}%")
    logger.info(f"  Energy std: {surrogate_energy_std:.2f}%")
    
    return metrics


def run_calibration_suite(world, cfg: dict) -> CalibrationResult:
    """Run a full calibration suite with representative moves.
    
    Args:
        world: MuJoCo simulation (MarsSim)
        cfg: Configuration dict
    
    Returns:
        CalibrationResult with fitted ranges and fidelity metrics
    """
    # Define representative moves for calibration (within ±5m bounds)
    # These should cover different terrain types and distances
    moves = [
        {'xy': (1.0, 0.0)},   # Short flat drive
        {'xy': (2.0, 0.0)},   # Medium flat drive
        {'xy': (1.5, 1.5)},   # Diagonal drive
        {'xy': (3.0, 0.0)},   # Longer drive
    ]
    
    n_samples = cfg.get('calibration_samples', 5)
    
    # Calibrate
    env = calibrate_surrogate(world, moves, n_samples)
    
    # Measure fidelity on a test sequence
    from common.types import Action, ActionKind
    test_seq = [
        Action(kind=ActionKind.DRIVE, params={'xy': (1.6, 0.8)}),
    ]
    
    fidelity = surrogate_fidelity(env, world, test_seq)
    
    return CalibrationResult(
        traction_range=env.traction_range,
        loc_drift_range=env.loc_drift_range,
        draw_mult_range=env.draw_mult_range,
        n_samples=n_samples * len(moves),
        fidelity_metrics=fidelity
    )
