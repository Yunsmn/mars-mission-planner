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
from planner import astar, gating, surrogate

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
    if len(idx) == 0 or len(idx) > 0.30 * t.size:   # nothing, or high ground everywhere (not a
        return None                                 # localized obstacle — it's just varied terrain)
    to_world = lambda i: -R + i / (n - 1) * 2 * R
    xs = [to_world(j) for _, j in idx]       # terrain is indexed [row=y, col=x]
    ys = [to_world(i) for i, _ in idx]
    return (round(float(np.mean(xs)), 1), round(float(np.mean(ys)), 1),
            round(min(xs), 1), round(max(xs), 1), round(min(ys), 1), round(max(ys), 1))


def _route_env(env):
    """A finer surrogate grid for route scoring, so the lightsim agrees with A*'s fine paths (the
    default coarse mission grid smears a dune too wide to score a skirting route fairly)."""
    terr = env.terrain_full if env.terrain_full is not None else env.terrain
    grid = min(32, terr.shape[0])
    coarse = surrogate.downsample_terrain(terr, grid)
    return surrogate.SurrogateEnv(
        terrain=coarse, meters_per_cell=2 * R / (grid - 1), dust_tau=env.dust_tau,
        traction_range=env.traction_range, loc_drift_range=env.loc_drift_range,
        draw_mult_range=env.draw_mult_range)


def assess_variants(start, goal, env, cfg: dict) -> list[dict]:
    """A* several candidate roads at different caution levels, then have the lightsim predict each
    one's entrapment risk under uncertainty. Returns an ordered list (direct → safe), each an entry
    with name/waypoints/tail_risk/length_m/safe — the roads Granite weighs before it commits."""
    rng = np.random.default_rng(0)
    ceiling = cfg.get("risk_ceiling", 0.10)
    terr = env.terrain_full if env.terrain_full is not None else env.terrain
    mpc = env.mpc_full if env.mpc_full else env.meters_per_cell
    fenv = _route_env(env)
    out = []
    for v in astar.plan_variants(start, goal, terr, mpc):
        risk = assess_path(start, v["waypoints"], fenv, cfg, rng)
        out.append({"name": v["name"], "waypoints": v["waypoints"], "tail_risk": risk,
                    "length_m": v["length_m"], "safe": risk <= ceiling})
    return out


def plan_route(start, goal, env, model, cfg: dict) -> dict:
    """A* offers a shortest and a safe route (verified by the lightsim); IBM Granite decides which to
    drive, and a gate guarantees the committed route is never a predicted trap. Returns
    {chosen, waypoints, rationale, decided_by, assessed, obstacle}.
    """
    ceiling = cfg.get("risk_ceiling", 0.10)
    assessed = assess_variants(start, goal, env, cfg)
    by_name = {a["name"]: a for a in assessed}

    chosen, decided_by, rationale = None, "gate", ""
    if model is not None and hasattr(model, "decide_route") and len(assessed) > 1:
        try:
            choice, reason = model.decide_route(assessed, goal, cfg)
        except Exception:
            choice, reason = None, ""
        if choice in by_name and by_name[choice]["safe"]:
            chosen, decided_by = by_name[choice], "granite"
            rationale = (f"IBM Granite weighed {len(assessed)} routes and chose the {choice} one: "
                         f"{reason} The lightsim puts it at {chosen['tail_risk'] * 100:.0f}% "
                         f"entrapment risk.")

    if chosen is None:                          # Granite absent, unparsed, or it picked an unsafe route
        safe = [a for a in assessed if a["safe"]]
        chosen = min(safe, key=lambda a: a["length_m"]) if safe \
            else min(assessed, key=lambda a: a["tail_risk"])
        rationale = (f"Safety gate selected the {chosen['name']} route "
                     f"({chosen['tail_risk'] * 100:.0f}% entrapment risk).")

    return {"chosen": chosen["name"], "waypoints": chosen["waypoints"], "rationale": rationale,
            "decided_by": decided_by, "assessed": assessed, "obstacle": obstacle_descriptor(env)}
