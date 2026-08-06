"""Value of Information — observe more only when it's worth the energy/time.

The 'insight-driven, not data-heavy' mechanism: don't collect data blindly; take an extra
observation only when its expected risk reduction beats its cost.

This implements the core VoI decision: when candidate actions are close in value or one is
borderline-risky, evaluate whether a cheap observation would reduce decision risk by more
than it costs in energy/time.

Authored by IBM Bob for the MARVIN mission planner.
"""
from __future__ import annotations

import numpy as np

from common.types import Action, ActionKind, CandidateScore, MissionState, Perception


def best_gap(safe: list[CandidateScore]) -> float:
    """Value gap between the top two safe candidates (small gap => ambiguous).
    
    Args:
        safe: List of safe (gated) candidate scores
    
    Returns:
        Value difference between best and second-best, or 0 if < 2 candidates
    """
    if len(safe) < 2:
        return 0.0
    
    # Sort by value descending
    sorted_safe = sorted(safe, key=lambda s: s.value, reverse=True)
    
    return sorted_safe[0].value - sorted_safe[1].value


def cheapest_observation(state: MissionState, perception: Perception) -> Action | None:
    """Propose the cheapest disambiguating observation.
    
    Observations that can help reduce uncertainty:
    1. SCAN - refresh full perception (moderate cost)
    2. OBSERVE - targeted observation (low cost)
    
    Args:
        state: Current mission state
        perception: Current perception
    
    Returns:
        An OBSERVE or SCAN action, or None if no useful observation available
    """
    # If localization uncertainty is high, a scan would help
    if state.localization_sigma > 2.0:
        return Action(kind=ActionKind.SCAN, params={})
    
    # If we have visible targets but uncertain terrain, observe
    if len(perception.visible_targets) > 0:
        return Action(kind=ActionKind.OBSERVE, params={})
    
    # Default: a simple scan
    return Action(kind=ActionKind.SCAN, params={})


def voi(obs: Action, state: MissionState, env) -> float:
    """Expected reduction in decision risk from taking `obs`.
    
    This is a simplified VoI estimate. A full implementation would:
    1. Simulate taking the observation
    2. Re-run the decision process with reduced uncertainty
    3. Calculate expected improvement in decision quality
    
    For now, we estimate based on current uncertainty levels.
    
    Args:
        obs: The observation action being considered
        state: Current mission state
        env: Environment (for uncertainty parameters)
    
    Returns:
        Estimated value of information (in same units as action value)
    """
    # Base VoI depends on current uncertainty
    base_voi = 0.0
    
    if obs.kind == ActionKind.SCAN:
        # SCAN reduces localization uncertainty significantly
        # Value scales with current uncertainty
        base_voi = state.localization_sigma * 0.1
        
        # Additional value if we have multiple visible targets (helps disambiguation)
        # This would come from perception, but we estimate here
        base_voi += 0.05
        
    elif obs.kind == ActionKind.OBSERVE:
        # OBSERVE provides targeted information
        # Lower value than full scan but still useful
        base_voi = state.localization_sigma * 0.05 + 0.02
    
    return base_voi


def cost(obs: Action) -> float:
    """Energy/time price of the observation.
    
    Args:
        obs: The observation action
    
    Returns:
        Cost in energy units (Wh) or equivalent value units
    """
    if obs.kind == ActionKind.SCAN:
        # SCAN is more expensive (full perception refresh)
        return 5.0 * 0.01  # 5 Wh * energy_penalty_factor
    
    elif obs.kind == ActionKind.OBSERVE:
        # OBSERVE is cheaper (targeted)
        return 2.0 * 0.01  # 2 Wh * energy_penalty_factor
    
    return 0.0


def maybe_observe(
    safe: list[CandidateScore],
    state: MissionState,
    perception: Perception,
    env,
    cfg: dict
) -> Action | None:
    """Return an OBSERVE action if voi(obs) > cost(obs) and the decision is ambiguous.
    
    This is the core VoI gate: only observe when:
    1. The decision is ambiguous (small gap between top candidates)
    2. The expected value of information exceeds the cost
    
    Args:
        safe: List of safe candidate scores
        state: Current mission state
        perception: Current perception
        env: Environment for VoI calculation
        cfg: Configuration with voi_gap_threshold
    
    Returns:
        An observation Action if VoI justifies it, None otherwise
    """
    # Check if decision is ambiguous
    gap = best_gap(safe)
    threshold = cfg.get('voi_gap_threshold', 0.15)
    
    if gap >= threshold:
        # Decision is clear, no need to observe
        return None
    
    # Decision is ambiguous, consider observing
    obs = cheapest_observation(state, perception)
    
    if obs is None:
        return None
    
    # Calculate VoI vs cost
    obs_voi = voi(obs, state, env)
    obs_cost = cost(obs)
    
    # Only observe if VoI exceeds cost
    if obs_voi > obs_cost:
        return obs
    
    return None


def should_replan(state: MissionState, last_perception: Perception, 
                  current_perception: Perception) -> bool:
    """Determine if replanning is needed based on perception changes.
    
    Triggers replanning when:
    - New targets become visible
    - Terrain hazards are discovered
    - Significant localization drift
    
    Args:
        state: Current mission state
        last_perception: Previous perception
        current_perception: Latest perception
    
    Returns:
        True if replanning is warranted
    """
    # Check for new visible targets
    last_target_ids = {t.id for t in last_perception.visible_targets}
    current_target_ids = {t.id for t in current_perception.visible_targets}
    
    if current_target_ids - last_target_ids:
        # New targets discovered
        return True
    
    # Check for significant terrain changes (simplified)
    # In a full implementation, would compare slope/roughness maps
    
    # Check for dust opacity changes (affects power budget)
    dust_change = abs(current_perception.dust_tau - last_perception.dust_tau)
    if dust_change > 0.2:
        return True
    
    # Check localization uncertainty
    if state.localization_sigma > 5.0:
        # High uncertainty, should replan with updated position estimate
        return True
    
    return False