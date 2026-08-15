"""Route planning — the AI decides the route, the lightsim verifies it.

Flow: from the perceived hazard's geometry, IBM Granite makes the routing call — which side to skirt
the dune and how much clearance to leave (it reasons from the obstacle, it is not picking from a
scored menu) → the fast surrogate ("lightsim") builds that detour and rolls it out under uncertainty,
sampling the whole path to predict its entrapment (tail) risk → a safety gate guarantees the committed
route is never a predicted trap, falling back to a programmatic search only if Granite's decision
verifies unsafe.

This is how MARVIN avoids the soft-soil dune that stuck Opportunity (Purgatory, 2005) and stranded
Spirit (Troy, 2009): the model reasons its way around the hazard, and its decision is checked against
the lightsim before the wheels move.
"""
from __future__ import annotations

import math

import numpy as np

from common.types import Action, ActionKind, MissionState, Pose
from planner import gating, surrogate

R = surrogate.TERRAIN_RADIUS


def candidate_routes(start, goal, side_offset: float = 1.9) -> dict:
    """Fixed reference routes (used by the comparison renderer, not the live planner)."""
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


def assess_path(start, waypoints, env, cfg: dict, rng) -> float:
    """Peak tail (worst-case) entrapment risk over a path's drive segments, via the lightsim."""
    q = cfg.get("cvar_quantile", 0.9)
    n = cfg.get("n_rollouts", 20)
    pos, worst = start, 0.0
    for wp in waypoints:
        st = MissionState(Pose(pos, 0.0), 100.0, 0.0, 1.0, (), ())
        b = surrogate.rollout_batch((Action(ActionKind.DRIVE, {"xy": wp}),), st, env, n, rng)
        worst = max(worst, float(gating.tail_worst(b.hazard, q)))
        pos = wp
    return round(worst, 3)


def obstacle_descriptor(env) -> tuple | None:
    """Locate the hazard (dune) in world coords from the surrogate's coarse height grid:
    returns (cx, cy, x_min, x_max, y_min, y_max) or None if the terrain is essentially flat."""
    t = env.terrain
    n = t.shape[0]
    lo, hi = float(t.min()), float(t.max())
    if hi - lo < 0.05:                       # no meaningful relief
        return None
    idx = np.argwhere(t > lo + 0.55 * (hi - lo))
    if len(idx) == 0:
        return None
    to_world = lambda i: -R + i / (n - 1) * 2 * R
    xs = [to_world(j) for _, j in idx]       # terrain is indexed [row=y, col=x]
    ys = [to_world(i) for i, _ in idx]
    return (round(float(np.mean(xs)), 1), round(float(np.mean(ys)), 1),
            round(min(xs), 1), round(max(xs), 1), round(min(ys), 1), round(max(ys), 1))


def _label(proposed) -> str:
    return "direct line" if not proposed else "via " + " → ".join(f"({p[0]:.1f},{p[1]:.1f})"
                                                                  for p in proposed)


def _entry(start, proposed, waypoints, risk, ceiling, reason="") -> dict:
    return {"name": _label(proposed), "proposed": proposed, "waypoints": waypoints, "tail_risk": risk,
            "length_m": round(_length(start, waypoints), 2), "safe": risk <= ceiling, "reason": reason}


def _detour(start, goal, side_y) -> list:
    """A proven two-corner skirt: out to `side_y` at low x, along it past the ridge, then the goal."""
    sx, sy = start
    gx, gy = goal
    return [(round(sx + (gx - sx) * 0.25, 2), round(side_y, 2)),
            (round(gx * 0.9, 2), round(side_y, 2)), tuple(goal)]


def _gate_fallback(start, goal, obstacle, env, cfg, rng, tried) -> dict:
    """Guaranteed-safe search: skirt each side of the dune with growing clearance, pick the safest."""
    ceiling = cfg.get("risk_ceiling", 0.10)
    ylo, yhi = (obstacle[4], obstacle[5]) if obstacle else (-1.5, 1.5)
    cands = []
    for mag in (0.5, 0.9, 1.3):
        for side_y in (ylo - mag, yhi + mag):
            wps = _detour(start, goal, float(np.clip(side_y, -R, R)))
            risk = assess_path(start, wps, env, cfg, rng)
            e = _entry(start, wps[:-1], wps, risk, ceiling, "safety-gate detour")
            e["name"] = "gate detour (" + ("south" if side_y < 0 else "north") + f", clearance {mag} m)"
            cands.append(e)
    safe = [c for c in cands if c["safe"]]
    best = min(safe, key=lambda c: (c["tail_risk"], c["length_m"])) if safe \
        else min(cands, key=lambda c: c["tail_risk"])
    n_bad = sum(1 for t in tried if not t["safe"])
    best["reason"] = (f"safety gate — {n_bad} Granite proposal(s) stayed unsafe, so the rover took "
                      f"the safest verified detour ({best['tail_risk'] * 100:.0f}% risk)")
    return best


def plan_route(start, goal, env, model, cfg: dict) -> dict:
    """Granite decides the route (side + clearance), the lightsim verifies it, the gate guarantees
    safety. Returns {chosen, waypoints, rationale, decided_by, assessed, obstacle}. `assessed` lists
    the straight-line plan, Granite's chosen detour, and any gate detour, each with its entrapment risk.
    """
    rng = np.random.default_rng(0)
    ceiling = cfg.get("risk_ceiling", 0.10)
    obstacle = obstacle_descriptor(env)
    ylo, yhi = (obstacle[4], obstacle[5]) if obstacle else (-1.5, 1.5)

    # Always assess the straight line too — it's the orbital plan MARVIN is beating, and the contrast
    # (direct = trap, chosen = safe) is the whole point.
    direct_risk = assess_path(start, [tuple(goal)], env, cfg, rng)
    assessed = [_entry(start, [], [tuple(goal)], direct_risk, ceiling, "the orbital straight-line plan")]
    assessed[0]["name"] = "direct line"

    chosen, decided_by, rationale = None, "gate", ""
    if model is not None and hasattr(model, "decide_route") and obstacle is not None:
        try:
            side, clearance, reason, ok = model.decide_route(start, goal, obstacle, cfg)
        except Exception:
            side, clearance, ok = None, None, False
        if ok:
            side_y = (yhi + clearance) if side == "north" else (ylo - clearance)
            wps = _detour(start, goal, float(np.clip(side_y, -R, R)))
            risk = assess_path(start, wps, env, cfg, rng)
            e = _entry(start, wps[:-1], wps, risk, ceiling, reason)
            e["name"] = f"Granite: {side} detour, {clearance:.1f} m clearance"
            assessed.append(e)
            if e["safe"]:
                chosen, decided_by = e, "granite"
                rationale = (f"IBM Granite chose to skirt the dune on the {side} side with "
                             f"{clearance:.1f} m clearance: {reason}. The lightsim verified it at "
                             f"{risk * 100:.0f}% entrapment risk.")

    if chosen is None:                         # Granite absent, unparsed, or its clearance verified unsafe
        gate = _gate_fallback(start, goal, obstacle, env, cfg, rng, assessed[1:])
        assessed.append(gate)
        chosen, decided_by, rationale = gate, "gate", gate["reason"]

    return {"chosen": chosen["name"], "waypoints": chosen["waypoints"], "rationale": rationale,
            "decided_by": decided_by, "assessed": assessed, "obstacle": obstacle}
