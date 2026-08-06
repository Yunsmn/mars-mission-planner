"""Tail-risk + budget gating. Safety-first: judge candidates on worst-case, not average.

Implements CVaR-style tail aggregation and filters candidates to the safe set based on:
- risk_ceiling: max acceptable tail (worst-10%) hazard
- p_success_min: minimum mean success probability
- battery_reserve_pct: never plan below this battery level

SCALE: Battery is in %, energy costs are in % (not Wh). Real world is ±5m.

Authored by IBM Bob for the MARVIN mission planner.
"""
from __future__ import annotations

import numpy as np

from common.types import ActionSeq, CandidateScore, Constraints, MissionState, RolloutBatch


def tail_worst(arr: np.ndarray, q: float) -> float:
    """CVaR-style worst (1-q) mean, e.g. q=0.90 -> mean of the worst 10%.
    
    Args:
        arr: Array of values (e.g., energy consumption or hazard scores)
        q: Quantile threshold (e.g., 0.90 for worst 10%)
    
    Returns:
        Mean of the worst (1-q) fraction of values
    """
    if len(arr) == 0:
        return 0.0
    
    # Sort in descending order to get worst values first
    sorted_arr = np.sort(arr)[::-1]
    
    # Take the worst (1-q) fraction
    n_worst = max(1, int(np.ceil(len(arr) * (1 - q))))
    worst_values = sorted_arr[:n_worst]
    
    return float(np.mean(worst_values))


def score_candidate(
    seq: ActionSeq,
    batch: RolloutBatch,
    state: MissionState,
    cfg: dict
) -> CandidateScore:
    """Score a candidate action sequence based on rollout outcomes.
    
    Args:
        seq: The action sequence being evaluated
        batch: Rollout results (success, energy, hazard arrays)
        state: Current mission state
        cfg: Configuration dict with cvar_quantile and energy penalty params
    
    Returns:
        CandidateScore with p_success, cvar_energy, risk, and net value
    """
    # Mean success probability across rollouts
    p_success = float(np.mean(batch.success))
    
    # Tail (worst-case) energy consumption (in battery %)
    q = cfg.get('cvar_quantile', 0.90)
    cvar_energy = tail_worst(batch.energy, q)
    
    # Tail (worst-case) hazard/risk
    risk = tail_worst(batch.hazard, q)
    
    # Calculate science value from the action sequence
    science_value = 0.0
    for action in seq:
        if action.kind.name == 'SAMPLE':
            # Find the target being sampled
            target_id = action.params.get('target')
            if target_id:
                for target in state.remaining:
                    if target.id == target_id:
                        science_value += target.science_value
                        break
    
    # Energy penalty (battery % to comparable scale with science value 0-1)
    # Since battery is in %, and typical moves cost 1-5%, scale appropriately
    energy_penalty_factor = cfg.get('energy_penalty_factor', 0.1)
    energy_penalty = cvar_energy * energy_penalty_factor
    
    # Net value = science gain - energy cost
    value = science_value - energy_penalty
    
    return CandidateScore(
        seq=seq,
        p_success=p_success,
        cvar_energy=cvar_energy,
        risk=risk,
        value=value
    )


def gate(
    scores: list[CandidateScore],
    c: Constraints,
    battery_pct: float
) -> list[CandidateScore]:
    """Keep only candidates within risk_ceiling, p_success_min, and battery reserve.
    
    Safety invariants (from docs/BUILD_WITH_BOB.md):
    - Nothing runs above risk_ceiling
    - Nothing runs below battery_reserve_pct
    - Must meet minimum success probability
    
    Args:
        scores: List of scored candidates
        c: Constraints (risk_ceiling, p_success_min, battery_reserve_pct)
        battery_pct: Current battery percentage
    
    Returns:
        Filtered list containing only safe candidates
    """
    safe = []
    
    for score in scores:
        # Check tail-risk constraint
        if score.risk > c.risk_ceiling:
            continue
        
        # Check success probability constraint
        if score.p_success < c.p_success_min:
            continue
        
        # Check battery reserve constraint
        # Energy is already in battery %, so direct comparison
        estimated_battery_after = battery_pct - score.cvar_energy
        
        if estimated_battery_after < c.battery_reserve_pct:
            continue
        
        # Passed all safety gates
        safe.append(score)
    
    return safe


def respects_battery_reserve(score: CandidateScore, battery_pct: float, reserve_pct: float) -> bool:
    """Check if executing this candidate would respect the battery reserve.
    
    Args:
        score: Candidate score with energy estimate (in battery %)
        battery_pct: Current battery percentage
        reserve_pct: Minimum battery reserve percentage
    
    Returns:
        True if battery would remain above reserve after action
    """
    estimated_battery_after = battery_pct - score.cvar_energy
    return estimated_battery_after >= reserve_pct
