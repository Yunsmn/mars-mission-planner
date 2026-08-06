"""Shared, immutable data types — the contracts every layer speaks in.

Declarative scaffolding (matches docs/INTERFACES.md). Behavior lives in the other modules,
authored with IBM Bob. Annotations are strings (PEP 563) so this imports without numpy.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

Vec2 = "tuple[float, float]"


@dataclass(frozen=True)
class Pose:
    xy: tuple[float, float]
    heading_rad: float


@dataclass(frozen=True)
class Target:
    id: str
    xy: tuple[float, float]
    science_value: float          # data-derived from CRISM mineralogy (see data/science_value.py)
    mineral_class: str            # e.g. "carbonate", "phyllosilicate", "olivine", "mafic"


@dataclass(frozen=True)
class Constraints:
    battery_reserve_pct: float
    risk_ceiling: float           # max acceptable tail (worst-10%) hazard
    p_success_min: float


@dataclass(frozen=True)
class MissionState:
    pose: Pose
    battery_pct: float
    sol_time: float               # fractional sol (Mars day)
    localization_sigma: float     # positional uncertainty (m)
    collected: tuple[str, ...]    # target ids sampled so far
    remaining: tuple[Target, ...]


@dataclass(frozen=True)
class Perception:
    slope_deg: "np.ndarray"       # local slope map (degrees)
    roughness: "np.ndarray"       # local roughness proxy
    visible_targets: tuple[Target, ...]
    dust_tau: float               # current atmospheric optical depth


class ActionKind(Enum):
    DRIVE = auto()
    SCAN = auto()
    SAMPLE = auto()
    OBSERVE = auto()              # a Value-of-Information observation
    HOLD = auto()                 # safe default


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    params: dict                  # e.g. {"xy": (x, y)} or {"target": "t3"}


ActionSeq = "tuple[Action, ...]"


@dataclass(frozen=True)
class RolloutBatch:
    success: "np.ndarray"         # shape (n,), bool
    energy: "np.ndarray"          # shape (n,), Wh used
    hazard: "np.ndarray"          # shape (n,), peak hazard in [0, 1]


@dataclass(frozen=True)
class CandidateScore:
    seq: "tuple[Action, ...]"
    p_success: float
    cvar_energy: float            # worst-case (tail) energy
    risk: float                   # worst-case (tail) hazard
    value: float                  # science value net of energy penalty


@dataclass(frozen=True)
class Decision:
    action: Action
    rationale: str                # natural-language justification (planner/justify.py)
    scores: "tuple[CandidateScore, ...]"
