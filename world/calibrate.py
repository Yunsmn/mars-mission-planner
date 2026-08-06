"""Link the two simulators: calibrate the lightweight surrogate FROM the detailed world.

The surrogate is only trustworthy if its injected uncertainty matches reality. Here we drive
a set of moves in the full MuJoCo world, fit the slip / energy / localization-drift
distributions, and hand those ranges to the surrogate. `surrogate_fidelity` then reports how
well the cheap model predicts full-physics outcomes — the validation number for the writeup.

See docs/DESIGN.md §5.2 and §5.6.

TODO(bob): implement (task B6b).
"""
from __future__ import annotations

from planner.surrogate import SurrogateEnv


def calibrate_surrogate(model, data, moves) -> SurrogateEnv:
    """Fit traction / draw / drift ranges from full-physics rollouts of `moves`."""
    raise NotImplementedError("TODO(bob): fit surrogate uncertainty ranges from MuJoCo")


def surrogate_fidelity(env: SurrogateEnv, model, data, seq) -> float:
    """Error between surrogate predictions and a full MuJoCo rollout of `seq` (lower = better)."""
    raise NotImplementedError("TODO(bob): measure surrogate-vs-world prediction error")
