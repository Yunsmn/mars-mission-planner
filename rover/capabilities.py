"""Onboard capability API — the rover's verbs, called IN-PROCESS by the planner.

NO server, NO MCP, NO network: on Mars there is nothing to connect to. The planner binds a
MarsSim (the ground-truth world) once via `bind()`, then calls these plain functions.
`sample()` is an abstract instrument action (pickup) — there is no arm simulation.

Implemented by Claude against MarsSim; the LLM planner that decides *which* verbs to call is
authored by Bob.
"""
from __future__ import annotations

import numpy as np

from common.types import Perception, Pose
from world import sensors

_SIM = None
_RNG = np.random.default_rng(0)


def bind(sim, seed: int = 0) -> None:
    """Attach the world the capabilities act on (call once at startup)."""
    global _SIM, _RNG
    _SIM = sim
    _RNG = np.random.default_rng(seed)


def _sim():
    if _SIM is None:
        raise RuntimeError("capabilities not bound — call rover.capabilities.bind(sim) first")
    return _SIM


def get_pose() -> Pose:
    """Noisy onboard pose estimate (odometry/IMU)."""
    return sensors.read_pose(_sim(), _RNG)


def battery() -> float:
    """Percent battery remaining."""
    return float(_sim().battery_pct)


def assess_slope(region: tuple[float, float]) -> float:
    """Terrain slope (degrees) at a world location."""
    return _sim().slope_at(region[0], region[1])


def scan() -> Perception:
    """Refresh perception from onboard sensors (latest observation drives decisions)."""
    return sensors.observe(_sim(), _RNG)


def drive_to(x: float, y: float) -> dict:
    """Navigate to (x, y). Returns {success, distance_m, steps, battery_pct}."""
    return _sim().drive_to(x, y)


def sample(target_id: str) -> dict:
    """Abstract instrument pickup at a reached target. Returns {success, ...}."""
    return _sim().sample(target_id)
