"""Explainable decision log — natural-language rationale citing the real numbers.

The ground-team audit trail and the backbone of the demo video.
Generates human-readable justifications for why actions were chosen or rejected,
referencing actual success probabilities, tail risks, and battery constraints.

Authored by IBM Bob for the MARVIN mission planner.
"""
from __future__ import annotations

from common.types import Action, ActionKind, CandidateScore


def justify(action: Action, scores: list[CandidateScore], model=None) -> str:
    """Render a human-readable justification, e.g. why a target was skipped/chosen,
    referencing success %, tail risk, and battery reserve.
    
    Args:
        action: The chosen action
        scores: All candidate scores (chosen + rejected)
        model: Optional model for generating more sophisticated explanations
    
    Returns:
        Natural language rationale string
    """
    if not scores:
        return "No valid candidates available. Executing HOLD for safety."
    
    # Find the chosen candidate
    chosen = None
    for score in scores:
        if _actions_match(score.seq, action):
            chosen = score
            break
    
    if chosen is None:
        return f"Executing {action.kind.name} action."
    
    # Build justification based on action type
    if action.kind == ActionKind.HOLD:
        return _justify_hold(scores)
    
    elif action.kind == ActionKind.DRIVE:
        return _justify_drive(chosen, scores, action)
    
    elif action.kind == ActionKind.SAMPLE:
        return _justify_sample(chosen, scores, action)
    
    elif action.kind == ActionKind.SCAN:
        return _justify_scan(chosen, scores)
    
    elif action.kind == ActionKind.OBSERVE:
        return _justify_observe(chosen, scores)
    
    else:
        return f"Executing {action.kind.name} with {chosen.p_success*100:.0f}% success probability."


def _actions_match(seq: tuple, action: Action) -> bool:
    """Check if an action sequence starts with the given action."""
    if not seq:
        return False
    first_action = seq[0]
    return (first_action.kind == action.kind and 
            first_action.params == action.params)


def _justify_hold(scores: list[CandidateScore]) -> str:
    """Justify a HOLD decision (no safe alternatives)."""
    if not scores:
        return "HOLD: No valid action sequences proposed. Maintaining safe state."
    
    # Find why candidates were rejected
    max_risk = max(s.risk for s in scores)
    min_success = min(s.p_success for s in scores)
    
    reasons = []
    if max_risk > 0.10:  # risk_ceiling
        reasons.append(f"excessive tail-risk ({max_risk*100:.0f}%)")
    if min_success < 0.70:  # p_success_min
        reasons.append(f"low success probability ({min_success*100:.0f}%)")
    
    reason_str = " and ".join(reasons) if reasons else "safety constraints"
    
    return (f"HOLD: All {len(scores)} proposed sequences rejected due to {reason_str}. "
            f"Maintaining safe state until conditions improve.")


def _justify_drive(chosen: CandidateScore, scores: list[CandidateScore], action: Action) -> str:
    """Justify a DRIVE decision."""
    target_xy = action.params.get('xy', (0, 0))
    
    # Compare with alternatives
    rejected = [s for s in scores if s != chosen]
    
    rationale = (f"DRIVE to ({target_xy[0]:.1f}, {target_xy[1]:.1f}): "
                f"{chosen.p_success*100:.0f}% success probability, "
                f"worst-case risk {chosen.risk*100:.0f}%, "
                f"energy cost {chosen.cvar_energy:.1f} Wh (tail estimate).")
    
    # Mention rejected alternatives if any
    if rejected:
        worst_rejected = max(rejected, key=lambda s: s.risk)
        if worst_rejected.risk > chosen.risk * 1.5:
            rationale += (f" Rejected riskier alternative with {worst_rejected.risk*100:.0f}% "
                         f"tail-risk.")
    
    return rationale


def _justify_sample(chosen: CandidateScore, scores: list[CandidateScore], action: Action) -> str:
    """Justify a SAMPLE decision."""
    target_id = action.params.get('target', 'unknown')
    
    # Extract science value from the chosen sequence
    science_value = 0.0
    for act in chosen.seq:
        if act.kind == ActionKind.SAMPLE:
            # Science value is embedded in the score calculation
            science_value = chosen.value + (chosen.cvar_energy * 0.01)  # reverse penalty
            break
    
    rationale = (f"SAMPLE target {target_id}: science value {science_value:.2f}, "
                f"{chosen.p_success*100:.0f}% success probability, "
                f"energy cost {chosen.cvar_energy:.1f} Wh.")
    
    # Compare with other targets if rejected
    rejected = [s for s in scores if s != chosen]
    if rejected:
        best_rejected = max(rejected, key=lambda s: s.value)
        if best_rejected.value > 0:
            rationale += (f" Prioritized over alternative (value {best_rejected.value:.2f}) "
                         f"due to better risk/reward balance.")
    
    return rationale


def _justify_scan(chosen: CandidateScore, scores: list[CandidateScore]) -> str:
    """Justify a SCAN decision."""
    return (f"SCAN: Refreshing perception to reduce uncertainty. "
            f"Energy cost {chosen.cvar_energy:.1f} Wh. "
            f"Expected to improve decision quality for subsequent actions.")


def _justify_observe(chosen: CandidateScore, scores: list[CandidateScore]) -> str:
    """Justify an OBSERVE decision (Value of Information)."""
    # Find gap between top candidates
    if len(scores) >= 2:
        sorted_scores = sorted(scores, key=lambda s: s.value, reverse=True)
        gap = sorted_scores[0].value - sorted_scores[1].value
        
        return (f"OBSERVE: Decision ambiguous (top candidates within {gap:.2f} value). "
                f"Taking additional observation (cost {chosen.cvar_energy:.1f} Wh) "
                f"to reduce uncertainty before committing to action.")
    
    return (f"OBSERVE: Taking additional observation to reduce decision uncertainty. "
            f"Energy cost {chosen.cvar_energy:.1f} Wh.")


def format_decision_log(decision, step: int) -> str:
    """Format a complete decision log entry for display/recording.
    
    Args:
        decision: Decision object with action, rationale, and scores
        step: Mission step number
    
    Returns:
        Formatted log entry string
    """
    header = f"\n{'='*70}\nSTEP {step}: {decision.action.kind.name}\n{'='*70}"
    
    body = f"\nRATIONALE:\n{decision.rationale}\n"
    
    # Add candidate comparison table
    if decision.scores:
        body += "\nCANDIDATE COMPARISON:\n"
        body += f"{'Rank':<6} {'Success%':<10} {'Risk%':<10} {'Energy(Wh)':<12} {'Value':<8}\n"
        body += "-" * 70 + "\n"
        
        sorted_scores = sorted(decision.scores, key=lambda s: s.value, reverse=True)
        for i, score in enumerate(sorted_scores[:5], 1):  # Top 5
            body += (f"{i:<6} {score.p_success*100:<10.1f} {score.risk*100:<10.1f} "
                    f"{score.cvar_energy:<12.1f} {score.value:<8.2f}\n")
    
    return header + body