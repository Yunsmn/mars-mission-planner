"""The surrogate simulator — the planner's cheap 'imagination'.

Runs a candidate action sequence n times under injected uncertainty to expose tail risk.
MUST be vectorized over all n rollouts (one batched NumPy pass, NO python loop over rollouts),
no rendering, no contact solve. Target: sub-millisecond for n=20, horizon=100.

This is NOT MuJoCo. MuJoCo (world/scene.py) is the real world; this is a reduced model:
path integrated over the DEM slope map, energy = f(dist, slope, payload),
slip/hazard = f(slope, roughness) with probability of getting stuck rising past a threshold.
Uncertainty per rollout: traction, localization drift, battery-draw multiplier.

TODO(bob): implement (task B6). This is the core novelty — keep it fast and honest.
"""
from __future__ import annotations

from dataclasses import dataclass

from common.types import ActionSeq, MissionState, RolloutBatch


@dataclass(frozen=True)
class SurrogateEnv:
    slope_deg: "np.ndarray"        # noqa: F821
    roughness: "np.ndarray"        # noqa: F821
    dust_tau: float
    traction_range: tuple[float, float]
    loc_drift_range: tuple[float, float]
    draw_mult_range: tuple[float, float]


def rollout_batch(seq: ActionSeq, state: MissionState, env: SurrogateEnv,
                  n: int, rng: "np.random.Generator") -> RolloutBatch:  # noqa: F821
    """Simulate `seq` from `state` n times under uncertainty. Returns per-rollout arrays."""
    raise NotImplementedError(
        "TODO(bob): vectorized rollouts (success/energy/hazard), all n at once"
    )
