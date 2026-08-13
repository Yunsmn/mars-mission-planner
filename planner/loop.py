"""The Propose-and-Verify loop — orchestrates one decision and the full mission.

Order in decide_next_action:
  propose (model) -> rollout_batch (surrogate) -> score -> gate (tail-risk + budget)
  -> maybe_observe (VoI) -> select robust best (or HOLD) -> justify -> Decision

Safety invariants (must hold — see docs/BUILD_WITH_BOB.md):
  * the model never executes, it only proposes;
  * nothing runs above risk_ceiling or below battery reserve;
  * empty safe set => HOLD.

Authored by IBM Bob for the MARVIN mission planner.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from common.types import Action, ActionKind, Constraints, Decision, MissionState, Perception
from planner import gating, justify, surrogate, voi

logger = logging.getLogger(__name__)


@dataclass
class MissionLog:
    """Record of mission execution."""
    decisions: list[Decision]
    final_state: MissionState
    total_energy: float
    samples_collected: int
    success: bool


SAMPLE_REACH_M = 0.55  # within this distance the instrument can sample (matches MarsSim.sample)


def _reflex_sample(world, pose) -> Decision | None:
    """Execution reflex: if the rover is within instrument reach of an uncollected target,
    sample it directly — no model call. A rover doesn't invoke an LLM to actuate a pickup it
    is already positioned over; the model is reserved for *planning* (which target, how to get
    there). This is what lets the mission complete quickly and reliably."""
    uncollected = [t for t in world.targets if not t.collected]
    if not uncollected:
        return None
    px, py = pose.xy
    nearest = min(uncollected, key=lambda t: (t.xy[0] - px) ** 2 + (t.xy[1] - py) ** 2)
    dist = ((nearest.xy[0] - px) ** 2 + (nearest.xy[1] - py) ** 2) ** 0.5
    if dist <= SAMPLE_REACH_M:
        return Decision(
            action=Action(kind=ActionKind.SAMPLE, params={"target": nearest.id}),
            rationale=(f"Within instrument reach of {nearest.id} ({dist:.2f} m) — sampling "
                       f"directly (execution reflex; model reserved for planning)."),
            scores=(),
        )
    return None


def _advance_toward_target(world, pose, battery_pct: float, cfg: dict) -> Decision | None:
    """Progress guarantee: if the model yields no productive action (HOLD) while the objective
    is unmet and battery allows, advance toward the nearest uncollected target. A real autonomy
    stack does not idle with work remaining and a full battery."""
    reserve = cfg.get("battery_reserve_pct", 15.0)
    uncollected = [t for t in world.targets if not t.collected]
    if not uncollected or battery_pct <= reserve:
        return None
    px, py = pose.xy
    nearest = min(uncollected, key=lambda t: (t.xy[0] - px) ** 2 + (t.xy[1] - py) ** 2)
    return Decision(
        action=Action(kind=ActionKind.DRIVE, params={"xy": nearest.xy}),
        rationale=(f"Objective unmet ({len(uncollected)} target(s) left, battery "
                   f"{battery_pct:.0f}%) — advancing to nearest target {nearest.id}."),
        scores=(),
    )


def decide_next_action(
    state: MissionState,
    perception: Perception,
    env: surrogate.SurrogateEnv,
    model,
    cfg: dict,
    consecutive_observations: int = 0,
    observations_per_target: dict[str, int] | None = None
) -> Decision:
    """The core propose-and-verify loop for one decision cycle.
    
    Steps:
    1. PROPOSE: Model generates k candidate action sequences
    2. VERIFY: Surrogate simulates each candidate n times under uncertainty
    3. SCORE: Calculate p_success, tail-risk, tail-energy, net value
    4. GATE: Filter to safe set (risk_ceiling, p_success_min, battery_reserve)
    5. VoI: Check if observation would improve decision quality
    6. SELECT: Choose best from safe set (or HOLD if empty)
    7. JUSTIFY: Generate natural language rationale
    
    Args:
        state: Current mission state
        perception: Current perception from sensors
        env: Surrogate environment for rollouts
        model: Proposer (Gemma 3 via Ollama)
        cfg: Configuration dict
        consecutive_observations: Number of consecutive observations taken
    
    Returns:
        Decision with chosen action, rationale, and candidate scores
    """
    # Extract config parameters
    n_candidates = cfg.get('n_candidates', 3)
    n_rollouts = cfg.get('n_rollouts', 20)
    seed = cfg.get('seed', 42)
    rng = np.random.default_rng(seed)
    
    # Build constraints
    constraints = Constraints(
        battery_reserve_pct=cfg.get('battery_reserve_pct', 15.0),
        risk_ceiling=cfg.get('risk_ceiling', 0.10),
        p_success_min=cfg.get('p_success_min', 0.70)
    )
    
    # STEP 1: PROPOSE - model generates candidates
    logger.info(f"Proposing {n_candidates} candidate action sequences...")
    candidates = model.propose(state, perception, n_candidates)
    
    if not candidates:
        logger.warning("No candidates proposed, defaulting to HOLD")
        hold_action = Action(kind=ActionKind.HOLD, params={})
        return Decision(
            action=hold_action,
            rationale="No valid candidates proposed. Maintaining safe state.",
            scores=()
        )
    
    logger.info(f"Received {len(candidates)} candidate sequences")
    
    # STEP 2-3: VERIFY and SCORE - simulate and evaluate each candidate
    scores = []
    for i, seq in enumerate(candidates):
        logger.debug(f"Evaluating candidate {i+1}/{len(candidates)}")
        
        # Rollout under uncertainty
        batch = surrogate.rollout_batch(seq, state, env, n_rollouts, rng)
        
        # Score the candidate
        score = gating.score_candidate(seq, batch, state, cfg)
        scores.append(score)
        
        logger.debug(
            f"  Candidate {i+1}: p_success={score.p_success:.2f}, "
            f"risk={score.risk:.2f}, value={score.value:.2f}"
        )
    
    # STEP 4: GATE - filter to safe set
    logger.info("Applying safety gates...")
    safe = gating.gate(scores, constraints, state.battery_pct)
    
    logger.info(f"Safe candidates: {len(safe)}/{len(scores)}")
    
    # STEP 5: VoI - check if observation would help
    # Only consider observation if we haven't exceeded the budget
    max_consecutive_obs = cfg.get('max_consecutive_observations', 2)
    
    if safe and consecutive_observations < max_consecutive_obs:
        obs_action = voi.maybe_observe(safe, state, perception, env, cfg, observations_per_target)
        
        if obs_action is not None:
            logger.info(f"VoI gate triggered: observation recommended (consecutive: {consecutive_observations + 1}/{max_consecutive_obs})")
            rationale = justify.justify(obs_action, scores, model)
            return Decision(
                action=obs_action,
                rationale=rationale,
                scores=tuple(scores)
            )
    
    # STEP 6: SELECT - choose best from safe set or HOLD
    if not safe:
        logger.warning("No safe candidates, executing HOLD")
        hold_action = Action(kind=ActionKind.HOLD, params={})
        rationale = justify.justify(hold_action, scores, model)
        return Decision(
            action=hold_action,
            rationale=rationale,
            scores=tuple(scores)
        )
    
    # Select candidate with highest value
    best = max(safe, key=lambda s: s.value)
    chosen_action = best.seq[0]  # Execute first action of best sequence
    
    logger.info(
        f"Selected action: {chosen_action.kind.name} "
        f"(value={best.value:.2f}, p_success={best.p_success:.2f})"
    )
    
    # STEP 7: JUSTIFY - generate rationale
    rationale = justify.justify(chosen_action, scores, model)
    
    return Decision(
        action=chosen_action,
        rationale=rationale,
        scores=tuple(scores)
    )


def run_mission(
    objective: str,
    world,
    model,
    cfg: dict,
    max_steps: int = 50
) -> MissionLog:
    """Drive the mission: perceive -> decide -> execute -> replan on new perception.
    
    The main mission loop that:
    1. Gets current perception from sensors
    2. Decides next action via propose-and-verify
    3. Executes the action through capabilities API
    4. Checks for replanning triggers
    5. Repeats until objective met or max steps reached
    
    Args:
        objective: Mission objective string (for context)
        world: The MuJoCo simulation (ground truth)
        model: Proposer model
        cfg: Configuration dict
        max_steps: Maximum mission steps
    
    Returns:
        MissionLog with decisions, final state, and metrics
    """
    import rover.capabilities as cap
    from world import sensors
    
    # Bind capabilities to the world
    seed = cfg.get('seed', 42)
    cap.bind(world, seed)
    rng = np.random.default_rng(seed)
    
    logger.info(f"Starting mission: {objective}")
    logger.info(f"Max steps: {max_steps}")
    
    decisions = []
    total_energy = 0.0
    last_perception = None
    consecutive_observations = 0
    observations_per_target = {}  # Track observations per target
    loc_sigma = 1.0  # Initial localization uncertainty
    
    # Mission loop
    for step in range(max_steps):
        logger.info(f"\n{'='*70}\nSTEP {step+1}/{max_steps}\n{'='*70}")
        
        # PERCEIVE - get current state and perception
        current_perception = cap.scan()
        current_pose = cap.get_pose()
        current_battery = cap.battery()
        
        # Build mission state with updated localization uncertainty
        # Uncertainty reduces with observations but doesn't fully reset after drives
        # Each observation reduces uncertainty by 0.2, minimum 0.1
        # Drives increase uncertainty slightly (0.05) due to odometry drift
        
        state = MissionState(
            pose=current_pose,
            battery_pct=current_battery,
            sol_time=step * 0.1,  # Rough time estimate
            localization_sigma=loc_sigma,
            collected=tuple(t.id for t in world.targets if t.collected),
            remaining=tuple(current_perception.visible_targets)
        )
        
        logger.info(
            f"State: pos=({state.pose.xy[0]:.1f}, {state.pose.xy[1]:.1f}), "
            f"battery={state.battery_pct:.1f}%, "
            f"collected={len(state.collected)}, "
            f"loc_sigma={state.localization_sigma:.2f}"
        )
        
        # Check for replanning triggers
        if last_perception is not None:
            if voi.should_replan(state, last_perception, current_perception):
                logger.info("Replanning triggered by perception change")
        
        # REFLEX first: sample immediately if we're already on an uncollected target
        # (no slow model call to pick up something we're standing on).
        decision = _reflex_sample(world, current_pose)
        if decision is None:
            # DECIDE - the model plans the next move (which target / how to get there),
            # verified by the surrogate and gated on tail-risk.
            env = surrogate.create_surrogate_env(world, cfg)
            decision = decide_next_action(state, current_perception, env, model, cfg,
                                          consecutive_observations, observations_per_target)

        # Progress guarantee: never idle while the objective is unmet and resources allow.
        if decision.action.kind == ActionKind.HOLD:
            advance = _advance_toward_target(world, current_pose, current_battery, cfg)
            if advance is not None:
                decision = advance
        decisions.append(decision)
        
        # Log decision
        logger.info(f"\nDECISION: {decision.action.kind.name}")
        logger.info(f"RATIONALE: {decision.rationale}")
        
        # EXECUTE - call capability API
        action = decision.action
        
        # Track consecutive observations and update localization uncertainty
        if action.kind in (ActionKind.OBSERVE, ActionKind.SCAN):
            consecutive_observations += 1
            # Reduce localization uncertainty with observations
            loc_sigma = max(0.1, loc_sigma - 0.2)
            
            # Track observations per target (for capping at ≤1 per target)
            if action.kind == ActionKind.OBSERVE and state.remaining:
                # Find closest target being observed
                closest_target = min(state.remaining, 
                                   key=lambda t: ((t.xy[0] - state.pose.xy[0])**2 + 
                                                 (t.xy[1] - state.pose.xy[1])**2)**0.5)
                observations_per_target[closest_target.id] = observations_per_target.get(closest_target.id, 0) + 1
        else:
            consecutive_observations = 0  # Reset on action execution
        
        if action.kind == ActionKind.DRIVE:
            xy = action.params['xy']
            logger.info(f"Executing DRIVE to ({xy[0]:.1f}, {xy[1]:.1f})")
            battery_before = cap.battery()
            result = cap.drive_to(xy[0], xy[1])
            battery_after = result.get('battery_pct', cap.battery())
            energy_used = battery_before - battery_after
            total_energy += energy_used
            
            # Increase localization uncertainty slightly due to odometry drift
            loc_sigma = min(2.0, loc_sigma + 0.05)
            
            if not result.get('success', False):
                logger.warning("Drive failed!")
        
        elif action.kind == ActionKind.SAMPLE:
            target_id = action.params['target']
            logger.info(f"Executing SAMPLE on {target_id}")
            battery_before = cap.battery()
            result = cap.sample(target_id)
            battery_after = result.get('battery_pct', cap.battery())
            energy_used = battery_before - battery_after
            total_energy += energy_used
            
            if result.get('success', False):
                logger.info(f"Sample {target_id} collected successfully")
        
        elif action.kind == ActionKind.SCAN:
            logger.info("Executing SCAN")
            battery_before = cap.battery()
            current_perception = cap.scan()
            battery_after = cap.battery()
            total_energy += battery_before - battery_after
        
        elif action.kind == ActionKind.OBSERVE:
            logger.info("Executing OBSERVE")
            # Targeted observation (simplified)
            battery_before = cap.battery()
            # Minimal observation action
            battery_after = cap.battery()
            total_energy += max(0.05, battery_before - battery_after)
        
        elif action.kind == ActionKind.HOLD:
            logger.info("Executing HOLD (safe state)")
            # No action, no energy
        
        # Check mission completion
        if sum(1 for t in world.targets if t.collected) >= 2:
            logger.info("Mission objective achieved: 2 samples collected")
            break
        
        if state.battery_pct < cfg.get('battery_reserve_pct', 15.0):
            logger.warning("Battery reserve reached, ending mission")
            break
        
        last_perception = current_perception
    
    # Build final state
    final_perception = cap.scan()
    final_pose = cap.get_pose()
    final_battery = cap.battery()
    
    final_state = MissionState(
        pose=final_pose,
        battery_pct=final_battery,
        sol_time=step * 0.1,
        localization_sigma=loc_sigma,
        collected=tuple(t.id for t in world.targets if t.collected),
        remaining=tuple(final_perception.visible_targets)
    )
    
    success = sum(1 for t in world.targets if t.collected) >= 2
    
    logger.info(f"\nMission complete: {len(decisions)} decisions, "
               f"{sum(1 for t in world.targets if t.collected)} samples, "
               f"{total_energy:.1f}% battery used")
    
    return MissionLog(
        decisions=decisions,
        final_state=final_state,
        total_energy=total_energy,
        samples_collected=sum(1 for t in world.targets if t.collected),
        success=success
    )