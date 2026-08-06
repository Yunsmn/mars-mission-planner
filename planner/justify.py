"""Explainable decision log — natural-language rationale citing the real numbers.

The ground-team audit trail and the backbone of the demo video.

TODO(bob): implement (task B10).
"""
from __future__ import annotations

from common.types import Action, CandidateScore


def justify(action: Action, scores: list[CandidateScore], model) -> str:
    """Render a human-readable justification, e.g. why a target was skipped/chosen,
    referencing success %, tail risk, and battery reserve."""
    raise NotImplementedError("TODO(bob): produce NL rationale from the scored candidates")
