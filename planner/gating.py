"""Tail-risk + budget gating. Safety-first: judge candidates on worst-case, not average.

TODO(bob): implement (task B7).
"""
from __future__ import annotations

from common.types import CandidateScore, Constraints, MissionState, RolloutBatch


def tail_worst(arr: "np.ndarray", q: float) -> float:  # noqa: F821
    """CVaR-style worst (1-q) mean, e.g. q=0.90 -> mean of the worst 10%."""
    raise NotImplementedError("TODO(bob): tail (CVaR) aggregate")


def score_candidate(seq, batch: RolloutBatch, state: MissionState, cfg: dict) -> CandidateScore:
    raise NotImplementedError("TODO(bob): p_success, cvar_energy, tail risk, net value")


def gate(scores: list[CandidateScore], c: Constraints, battery_pct: float
         ) -> list[CandidateScore]:
    """Keep only candidates within risk_ceiling, p_success_min, and battery reserve."""
    raise NotImplementedError("TODO(bob): filter to the safe set")
