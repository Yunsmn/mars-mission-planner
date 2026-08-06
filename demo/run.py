"""MARVIN mission demo — intelligent onboard planning with propose-and-verify.

This demo showcases the full intelligence layer:
- Local model (Gemma 4) proposes action sequences
- Vectorized surrogate verifies under uncertainty (20x rollouts)
- Tail-risk gating ensures safety-first decisions
- Value of Information triggers observations when needed
- Natural language justifications explain every decision

The rover autonomously plans its mission to collect samples while respecting
battery reserves and risk constraints.

Run:  .venv/bin/python -m demo.run

Authored by IBM Bob for the MARVIN mission planner.
"""
from __future__ import annotations

import logging
import sys

import yaml

from model.propose import Proposer
from planner.loop import run_mission
from rover import capabilities as cap
from world import sensors
from world.sim import MarsSim

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

logger = logging.getLogger(__name__)


def main() -> None:
    """Run the MARVIN intelligent mission demo."""
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    
    logger.info("="*70)
    logger.info("MARVIN - Mars Autonomous Reasoning & Verification INtelligence")
    logger.info("="*70)
    logger.info(f"Mission: {cfg['mission']['objective']}")
    logger.info(f"Site: {cfg['mission']['site']}")
    
    # Initialize world (MuJoCo ground truth)
    seed = cfg['planner']['seed']
    sim = MarsSim(seed=seed)
    cap.bind(sim, seed=seed + 1)
    
    # Display world info
    K = sensors.camera_intrinsics(sim.model)
    logger.info("\n=== Onboard Camera Intrinsics ===")
    logger.info(f"  FOV: {K['fovy_deg']:.0f}° | Resolution: {K['width']}x{K['height']}")
    logger.info(f"  Focal: fx={K['fx']:.1f}, fy={K['fy']:.1f} | Center: cx={K['cx']:.1f}, cy={K['cy']:.1f}")
    
    logger.info("\n=== World Environment ===")
    logger.info(f"  Max slope: {sim.slope.max():.1f}°")
    logger.info(f"  Targets: {len(sim.targets)}")
    logger.info(f"  Start battery: {cap.battery():.1f}%")
    logger.info(f"  Start pose: {cap.get_pose()}")
    
    # Initialize proposer (Gemma 4 via Ollama)
    model_cfg = cfg['model']
    logger.info("\n=== Intelligence Layer ===")
    logger.info(f"  Model: {model_cfg['name']} @ {model_cfg['host']}")
    logger.info(f"  Temperature: {model_cfg['temperature']}")
    logger.info(f"  Candidates per decision: {cfg['planner']['n_candidates']}")
    logger.info(f"  Rollouts per candidate: {cfg['planner']['n_rollouts']}")
    
    try:
        proposer = Proposer(
            name=model_cfg['name'],
            host=model_cfg['host'],
            temperature=model_cfg['temperature']
        )
        
        # Run the intelligent mission
        logger.info("\n" + "="*70)
        logger.info("STARTING AUTONOMOUS MISSION")
        logger.info("="*70)
        
        mission_log = run_mission(
            objective=cfg['mission']['objective'],
            world=sim,
            model=proposer,
            cfg={
                **cfg['planner'],
                **cfg['constraints'],
                'energy_penalty_factor': 0.01
            },
            max_steps=30
        )
        
        # Display mission summary
        logger.info("\n" + "="*70)
        logger.info("MISSION SUMMARY")
        logger.info("="*70)
        logger.info(f"  Success: {mission_log.success}")
        logger.info(f"  Decisions made: {len(mission_log.decisions)}")
        logger.info(f"  Samples collected: {mission_log.samples_collected}/2")
        logger.info(f"  Total energy used: {mission_log.total_energy:.1f} Wh")
        logger.info(f"  Final battery: {mission_log.final_state.battery_pct:.1f}%")
        logger.info(f"  Final position: ({mission_log.final_state.pose.xy[0]:.1f}, "
                   f"{mission_log.final_state.pose.xy[1]:.1f})")
        
        # Display collected samples
        if mission_log.final_state.collected:
            logger.info(f"\n  Collected samples: {', '.join(mission_log.final_state.collected)}")
        
        # Display key decisions
        logger.info("\n=== Key Decisions ===")
        for i, decision in enumerate(mission_log.decisions[:5], 1):  # Show first 5
            logger.info(f"\nDecision {i}: {decision.action.kind.name}")
            logger.info(f"  {decision.rationale[:150]}...")
        
        if len(mission_log.decisions) > 5:
            logger.info(f"\n  ... and {len(mission_log.decisions) - 5} more decisions")
        
        logger.info("\n" + "="*70)
        logger.info("MISSION COMPLETE")
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f"\nMission failed with error: {e}")
        logger.error("Make sure Ollama is running: ollama serve")
        logger.error(f"And the model is available: ollama pull {model_cfg['name']}")
        
        # Fallback to scripted demo
        logger.info("\n" + "="*70)
        logger.info("FALLBACK: Running scripted demo (no intelligence layer)")
        logger.info("="*70)
        
        run_scripted_demo(sim)


def run_scripted_demo(sim: MarsSim) -> None:
    """Fallback scripted demo when Ollama is not available."""
    logger.info(f"\nStart battery: {cap.battery():.1f}%")
    logger.info(f"Start pose: {cap.get_pose()}")
    
    for t in sim.targets:
        p = cap.scan()
        logger.info(f"\nPerception: {len(p.visible_targets)} target(s) in view, dust_tau={p.dust_tau:.2f}")
        logger.info(f"-> Drive to {t.id} at {t.xy}")
        res = cap.drive_to(*t.xy)
        logger.info(f"   {res}")
        logger.info(f"   Sample: {cap.sample(t.id)}")
    
    logger.info(f"\nCollected: {[t.id for t in sim.targets if t.collected]}")
    logger.info(f"Battery remaining: {cap.battery():.1f}%")


if __name__ == "__main__":
    main()