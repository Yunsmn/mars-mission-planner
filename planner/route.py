"""Route planning — the AI decides the path based on the lightsim.

Flow: generate candidate routes to a goal → the fast surrogate ("lightsim") rolls each out under
uncertainty and predicts its entrapment (tail) risk → IBM Granite *decides* which route to take
from those assessments → a safety gate guarantees the chosen route is never a predicted trap.

This is how MARVIN avoids the class of soft-soil dune that stuck Opportunity (Purgatory, 2005) and
stranded Spirit (Troy, 2009): the model reasons over the lightsim's predictions instead of driving
blindly along the shortest orbital path.
"""
from __future__ import annotations

import math

import numpy as np

from common.types import Action, ActionKind, MissionState, Pose
from planner import gating, surrogate


def candidate_routes(start, goal, side_offset: float = 1.9) -> dict:
    """The routes the rover could take to the goal: the direct line and detours to either side."""
    sx, sy = start
    gx, gy = goal
    mx = sx + (gx - sx) * 0.25
    return {
        "direct": [tuple(goal)],
        "detour north": [(round(mx, 2), round(sy + side_offset, 2)),
                         (round(gx * 0.9, 2), round(sy + side_offset, 2)), tuple(goal)],
        "detour south": [(round(mx, 2), round(sy - side_offset, 2)),
                         (round(gx * 0.9, 2), round(sy - side_offset, 2)), tuple(goal)],
    }


def _length(start, wps) -> float:
    pos, d = start, 0.0
    for wp in wps:
        d += math.dist(pos, wp)
        pos = wp
    return d


def assess_routes(start, env, cfg: dict, routes: dict) -> list[dict]:
    """Lightsim each route: peak tail (worst-case) entrapment risk over its drive segments + length."""
    rng = np.random.default_rng(0)
    q = cfg.get("cvar_quantile", 0.9)
    n = cfg.get("n_rollouts", 20)
    ceiling = cfg.get("risk_ceiling", 0.10)
    out = []
    for name, wps in routes.items():
        pos, worst = start, 0.0
        for wp in wps:
            st = MissionState(Pose(pos, 0.0), 100.0, 0.0, 1.0, (), ())
            b = surrogate.rollout_batch((Action(ActionKind.DRIVE, {"xy": wp}),), st, env, n, rng)
            worst = max(worst, float(gating.tail_worst(b.hazard, q)))
            pos = wp
        out.append({"name": name, "waypoints": wps, "tail_risk": round(worst, 3),
                    "length_m": round(_length(start, wps), 2), "safe": worst <= ceiling})
    return out


def plan_route(start, goal, env, model, cfg: dict) -> dict:
    """Generate + lightsim-assess candidate routes, let Granite decide, and gate for safety.

    Returns {chosen, waypoints, rationale, decided_by, assessed}.
    """
    routes = candidate_routes(start, goal)
    assessed = assess_routes(start, env, cfg, routes)
    by_name = {r["name"]: r for r in assessed}
    safe = [r for r in assessed if r["safe"]]

    chosen_name, rationale, decided_by = None, "", "gate"
    if model is not None and hasattr(model, "choose_route"):
        try:
            chosen_name, why = model.choose_route(assessed, goal, cfg)
            if chosen_name in by_name and by_name[chosen_name]["safe"]:
                rationale = f"IBM Granite chose the {chosen_name}: {why}"
                decided_by = "granite"
            else:
                chosen_name = None            # model picked an unsafe/invalid route -> gate takes over
        except Exception:
            chosen_name = None

    if chosen_name is None:                    # safety fallback: safest, then shortest
        best = min(safe, key=lambda r: (r["tail_risk"], r["length_m"])) if safe else \
            min(assessed, key=lambda r: r["tail_risk"])
        chosen_name = best["name"]
        rationale = (f"Safety gate selected the {chosen_name} "
                     f"({best['tail_risk'] * 100:.0f}% entrapment risk) — the model's pick was unsafe or absent.")

    r = by_name[chosen_name]
    return {"chosen": chosen_name, "waypoints": r["waypoints"], "rationale": rationale,
            "decided_by": decided_by, "assessed": assessed}
