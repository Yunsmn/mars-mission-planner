#!/usr/bin/env python3
"""Test that demo.run works with the intelligent planning layer (using mock proposer)."""

import logging
import sys
import math

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_demo_with_mock_proposer():
    """Test the full mission loop with a mock proposer."""
    from world.sim import MarsSim
    from planner.loop import run_mission
    from common.types import Action, ActionKind
    import yaml
    
    logger.info("Loading config...")
    with open('config.yaml') as f:
        cfg = yaml.safe_load(f)
    
    # Flatten config structure
    flat_cfg = {
        'seed': cfg['planner']['seed'],
        'n_candidates': cfg['planner']['n_candidates'],
        'n_rollouts': cfg['planner']['n_rollouts'],
        'cvar_quantile': cfg['planner']['cvar_quantile'],
        'battery_reserve_pct': cfg['constraints']['battery_reserve_pct'],
        'risk_ceiling': cfg['constraints']['risk_ceiling'],
        'p_success_min': cfg['constraints']['p_success_min'],
        'surrogate_grid_size': 16,
        'energy_penalty_factor': 0.1
    }
    
    logger.info("Creating world...")
    world = MarsSim(seed=42)
    
    logger.info(f"World has {len(world.targets)} targets:")
    for t in world.targets:
        logger.info(f"  {t.id}: {t.xy}")
    
    # Create mock proposer that returns simple action sequences
    class MockProposer:
        def propose(self, state, perception, k):
            """Return simple action sequences - single actions only since loop executes first action."""
            sequences = []
            
            # Check if we're close to any target (within 0.6m = sample reach)
            for target in perception.visible_targets:
                if target.id not in state.collected:
                    dist = math.hypot(target.xy[0] - state.pose.xy[0], target.xy[1] - state.pose.xy[1])
                    
                    if dist < 0.6:
                        # Close enough to sample
                        sequences.append((
                            Action(kind=ActionKind.SAMPLE, params={'target': target.id}),
                        ))
                    else:
                        # Drive toward target
                        sequences.append((
                            Action(kind=ActionKind.DRIVE, params={'xy': target.xy}),
                        ))
                    break
            
            # Sequence 2: HOLD (safe fallback)
            sequences.append((
                Action(kind=ActionKind.HOLD, params={}),
            ))
            
            # Pad to k sequences
            while len(sequences) < k:
                sequences.append((Action(kind=ActionKind.HOLD, params={}),))
            
            return sequences[:k]
    
    model = MockProposer()
    
    logger.info("\nRunning mission with mock proposer...")
    objective = cfg['mission']['objective']
    
    try:
        log = run_mission(objective, world, model, flat_cfg, max_steps=15)
        
        logger.info(f"\n{'='*70}")
        logger.info("MISSION COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"Decisions made: {len(log.decisions)}")
        logger.info(f"Samples collected: {log.samples_collected}")
        logger.info(f"Battery used: {log.total_energy:.1f}%")
        logger.info(f"Success: {log.success}")
        logger.info(f"Final battery: {log.final_state.battery_pct:.1f}%")
        
        # Verify mission worked
        assert log.samples_collected >= 1, "Should have collected at least 1 sample"
        assert log.final_state.battery_pct > flat_cfg['battery_reserve_pct'], "Should respect battery reserve"
        
        logger.info("\n✓ Demo integration test PASSED!")
        
    except Exception as e:
        logger.error(f"Mission failed: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    test_demo_with_mock_proposer()