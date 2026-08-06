"""Value of Information — observe more only when it's worth the energy/time.

The 'insight-driven, not data-heavy' mechanism: don't collect data blindly; take an extra
observation only when its expected risk reduction beats its cost.

TODO(bob): implement (task B8).
"""
from __future__ import annotations

from common.types import Action, CandidateScore, MissionState, Perception


def best_gap(safe: list[CandidateScore]) -> float:
    """Value gap between the top two safe candidates (small gap => ambiguous)."""
    raise NotImplementedError("TODO(bob): compute decision ambiguity")


def cheapest_observation(state: MissionState, perception: Perception) -> Action | None:
    raise NotImplementedError("TODO(bob): propose the cheapest disambiguating observation")


def voi(obs: Action, state: MissionState, env) -> float:
    """Expected reduction in decision risk from taking `obs`."""
    raise NotImplementedError("TODO(bob): estimate value of information")


def cost(obs: Action) -> float:
    raise NotImplementedError("TODO(bob): energy/time price of the observation")


def maybe_observe(safe, state, perception, env, cfg) -> Action | None:
    """Return an OBSERVE action if voi(obs) > cost(obs) and the decision is ambiguous."""
    raise NotImplementedError("TODO(bob): VoI decision gate")
