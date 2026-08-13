#!/usr/bin/env python3
"""Quick integration test to verify the planning layer connects to MarsSim properly."""

import logging
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_basic_integration():
    """Test that the planning layer can interact with MarsSim."""
    from world.sim import MarsSim
    from common.types import Action, ActionKind, MissionState, Pose, Target, Perception
    from planner import surrogate, gating
    import numpy as np
    
    logger.info("Creating MarsSim...")
    world = MarsSim(seed=42)
    
    logger.info(f"World created with {len(world.targets)} targets")
    logger.info(f"Battery: {world.battery_pct:.1f}%")
    
    # Test pose
    x, y, yaw = world.pose()
    logger.info(f"Initial pose: ({x:.2f}, {y:.2f}, {yaw:.2f})")
    
    # Test that targets have the correct structure
    for t in world.targets:
        logger.info(f"Target {t.id}: xy={t.xy}, collected={t.collected}")
    
    # Test surrogate environment creation
    logger.info("\nCreating surrogate environment...")
    cfg = {
        'surrogate_grid_size': 16,
        'traction_range': (0.8, 1.2),
        'loc_drift_range': (0.05, 0.15),
        'draw_mult_range': (0.9, 1.1),
        'seed': 42
    }
    
    env = surrogate.create_surrogate_env(world, cfg)
    logger.info(f"Surrogate terrain shape: {env.terrain.shape}")
    logger.info(f"Meters per cell: {env.meters_per_cell:.3f}")
    
    # Test a simple rollout
    logger.info("\nTesting surrogate rollout...")
    state = MissionState(
        pose=Pose(xy=(x, y), heading_rad=yaw),
        battery_pct=world.battery_pct,
        sol_time=0.0,
        localization_sigma=0.1,
        collected=tuple(t.id for t in world.targets if t.collected),
        remaining=tuple(
            Target(id=t.id, xy=t.xy, science_value=0.5, mineral_class="unknown")
            for t in world.targets if not t.collected
        )
    )
    
    # Simple action sequence: drive to first target
    target = world.targets[0]
    seq = (Action(kind=ActionKind.DRIVE, params={'xy': target.xy}),)
    
    rng = np.random.default_rng(42)
    batch = surrogate.rollout_batch(seq, state, env, n=5, rng=rng)
    
    logger.info(f"Rollout results (n=5):")
    logger.info(f"  Success rate: {np.mean(batch.success):.2f}")
    logger.info(f"  Mean energy: {np.mean(batch.energy):.2f}%")
    logger.info(f"  Mean hazard: {np.mean(batch.hazard):.2f}")
    
    # Test actual drive
    logger.info(f"\nTesting actual drive to target {target.id} at {target.xy}...")
    battery_before = world.battery_pct
    result = world.drive_to(target.xy[0], target.xy[1])
    battery_after = world.battery_pct
    
    logger.info(f"Drive result: success={result['success']}, distance={result['distance_m']:.2f}m")
    logger.info(f"Battery: {battery_before:.1f}% -> {battery_after:.1f}% (used {battery_before - battery_after:.2f}%)")
    
    # Test sample
    if result['success']:
        logger.info(f"\nTesting sample collection...")
        battery_before = world.battery_pct
        sample_result = world.sample(target.id)
        battery_after = world.battery_pct
        
        logger.info(f"Sample result: success={sample_result['success']}")
        logger.info(f"Battery: {battery_before:.1f}% -> {battery_after:.1f}% (used {battery_before - battery_after:.2f}%)")
        logger.info(f"Target collected: {target.collected}")
    
    # Verify collected samples tracking
    collected_count = sum(1 for t in world.targets if t.collected)
    logger.info(f"\nTotal samples collected: {collected_count}")
    
    logger.info("\n✓ Integration test passed!")

if __name__ == '__main__':
    test_basic_integration()
