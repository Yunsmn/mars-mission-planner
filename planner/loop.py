"""The Propose-and-Verify loop — orchestrates one decision and the full mission.

Order in decide_next_action:
  propose (model) -> rollout_batch (surrogate) -> score -> gate (tail-risk + budget)
  -> maybe_observe (VoI) -> select robust best (or HOLD) -> justify -> Decision

Safety invariants (must hold — see docs/BUILD_WITH_BOB.md):
  * the model never executes, it only proposes;
  * nothing runs above risk_ceiling or below battery reserve;
  * empty safe set => HOLD.

TODO(bob): implement (task B11).
"""
from __future__ import annotations

from common.types import Decision, MissionState, Perception


def decide_next_action(state: MissionState, perception: Perception, env, model, cfg) -> Decision:
    raise NotImplementedError("TODO(bob): full propose->verify->gate->voi->select->justify")


def run_mission(objective: str, world, model, cfg) -> "MissionLog":  # noqa: F821
    """Drive the mission: perceive -> decide -> execute -> replan on new perception."""
    raise NotImplementedError("TODO(bob): mission loop with replanning")
