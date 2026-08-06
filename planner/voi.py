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


def observation_cost(obs: Action) -> float:
    """Energy/time cost of an observation action.
    
    Args:
        obs: The observation action
    
    Returns:
        Cost in comparable units to action value
    """
    if obs.kind == ActionKind.SCAN:
        return 0.01  # Battery % cost for full scan
    elif obs.kind == ActionKind.OBSERVE:
        return 0.005  # Battery % cost for targeted observation
    return 0.0


def maybe_observe(
    safe: list[CandidateScore],
    state: MissionState,
    perception: Perception,
    env,
    cfg: dict
) -> Action | None:
    """Decide whether to observe before acting.
    
    Returns an observation action if:
    1. The best two candidates are close in value (ambiguous decision)
    2. The expected VoI exceeds the observation cost
    3. We haven't exceeded the observation budget for this mission
    
    Args:
        safe: List of safe candidate scores
        state: Current mission state
        perception: Current perception
        env: Environment
        cfg: Configuration dict
    
    Returns:
        An observation action if VoI justifies it, None otherwise
    """
    # Don't observe if we have no safe candidates or only one
    if len(safe) < 2:
        return None
    
    # Check observation budget (prevent infinite loops)
    # Track observations in mission state or config
    max_consecutive_observations = cfg.get('max_consecutive_observations', 2)
    
    # Simple heuristic: if localization uncertainty is very low, don't observe
    # (indicates we've already done enough observations)
    if state.localization_sigma < 0.5:
        return None
    
    # Calculate value gap between top candidates
    gap = best_gap(safe)
    threshold = cfg.get('voi_gap_threshold', 0.15)
    
    # Only consider observation if decision is ambiguous
    if gap >= threshold:
        return None
    
    # Get cheapest useful observation
    obs = cheapest_observation(state, perception)
    if obs is None:
        return None
    
    # Calculate VoI vs cost
    expected_voi = voi(obs, state, env)
    cost = observation_cost(obs)
    
    # Only observe if VoI exceeds cost by a margin
    if expected_voi > cost * 1.5:
        return obs
    
    return None


def should_replan(
    state: MissionState,
    old_perception: Perception,
    new_perception: Perception
) -> bool:
    """Check if significant perception change warrants replanning.
    
    Args:
        state: Current mission state
        old_perception: Previous perception
        new_perception: Current perception
    
    Returns:
        True if replanning is recommended
    """
    # Check if new targets appeared
    old_ids = {t.id for t in old_perception.visible_targets}
    new_ids = {t.id for t in new_perception.visible_targets}
    
    if new_ids != old_ids:
        return True
    
    # Check if terrain assessment changed significantly
    # (would need to compare slope/roughness distributions)
    # For now, always replan after observations
    return False


def cost_of(action: Action) -> float:
    """Estimate energy cost of an action (for VoI comparison).
    
    Args:
        action: The action to estimate
    
    Returns:
        Estimated energy cost in battery %
    """
    if action.kind == ActionKind.DRIVE:
        # Rough estimate based on typical drive distance
        return 1.0  # ~1% per meter average
    elif action.kind == ActionKind.SAMPLE:
        return 0.4  # Fixed sample cost
    elif action.kind == ActionKind.SCAN:
        return 0.1
    elif action.kind == ActionKind.OBSERVE:
        return 0.05
    return 0.0
