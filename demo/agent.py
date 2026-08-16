"""The MARVIN conversational agent — shared by the headless console (demo.cli) and the live 3D
window (demo.live). Judges talk to Granite in natural language: ask about the terrain and science
around the rover, or tell it to collect a sample. Granite answers and, when asked to act, drives the
lightsim-verified safe route (via the A* navigator) and samples.

One Granite call per turn: the live readout below already contains each sample's A* route risks, so
MARVIN can answer *and* decide in a single call.
"""
from __future__ import annotations

import math
import re

from planner import route, surrogate
from rover import capabilities as cap


def describe_environment(sim, env) -> str:
    """The space-data readout MARVIN reasons over: terrain relief, slope, dust, and the dune hazard."""
    slope_max = float(sim.slope.max())
    relief = float(sim.terrain.max() - sim.terrain.min())
    obst = route.obstacle_descriptor(env)
    lines = [
        f"  terrain: local relief {relief:.2f} m across the patch, slope up to {slope_max:.0f} deg",
        f"  dust opacity tau: {sim.dust_tau:.2f} (higher dust = less solar power)",
    ]
    if obst:
        ox, oy, x0, x1, y0, y1 = obst
        lines.append(f"  HAZARD: a soft-sand dune ridge along x~{ox:.1f}, spanning y {y0:.1f} to "
                     f"{y1:.1f} — driving onto it can strand the rover")
    else:
        lines.append("  no major hazard detected on the route ahead")
    return "\n".join(lines)


def _sample_lines(sim, env, cfg) -> str:
    x, y, _ = sim.pose()
    uncollected = [t for t in sim.targets if not t.collected]
    if not uncollected:
        return "  (all samples cached)"
    out = []
    for t in uncollected:
        dist = math.hypot(t.xy[0] - x, t.xy[1] - y)
        paths = route.assess_paths((x, y), t.xy, env, cfg)
        risks = " / ".join(f"{p['name']} {p['tail_risk'] * 100:.0f}% risk" for p in paths)
        out.append(f"  - {t.id} at ({t.xy[0]:+.1f}, {t.xy[1]:+.1f}): {t.mineral_class}, "
                   f"science {t.science_value:.2f}, {dist:.1f} m away. Routes (A* + lightsim): {risks}")
    return "\n".join(out)


def build_context(sim, env, cfg) -> str:
    x, y, yaw = sim.pose()
    return (f"ROVER STATE: position ({x:+.1f}, {y:+.1f}), heading {math.degrees(yaw):+.0f} deg, "
            f"battery {sim.battery_pct:.0f}%.\n"
            f"ENVIRONMENT (orbital DEM + onboard sensors):\n{describe_environment(sim, env)}\n"
            f"SAMPLES IN RANGE:\n{_sample_lines(sim, env, cfg)}")


def _named_target(sim, text: str):
    """Any uncollected target whose id or mineral appears anywhere in the action text."""
    for t in sim.targets:
        if not t.collected and (t.id.lower() in text or t.mineral_class.lower() in text):
            return t
    return None


def _nearest_uncollected(sim, near=None):
    unc = [t for t in sim.targets if not t.collected]
    if not unc:
        return None
    ref = near if near else sim.pose()[:2]
    return min(unc, key=lambda t: math.hypot(t.xy[0] - ref[0], t.xy[1] - ref[1]))


def _drive_safe(sim, env, cfg, goal, sample_id=None) -> str:
    """Drive the lightsim-verified route to `goal` (A* safe path), reporting the shortest-vs-safe
    contrast, then sample if asked. Uses cap.drive_to so the live viewer animates."""
    assessed = route.assess_paths(sim.pose()[:2], goal, env, cfg)
    by = {a["name"]: a for a in assessed}
    safe = by.get("safe") if by.get("safe", {}).get("safe") else \
        min(assessed, key=lambda a: a["tail_risk"])
    short = by.get("shortest", {})
    lines = [f"  navigator: shortest {short.get('length_m', '?')} m at "
             f"{short.get('tail_risk', 0) * 100:.0f}% risk; taking the safe route "
             f"{safe['length_m']} m at {safe['tail_risk'] * 100:.0f}% risk."]
    for wp in safe["waypoints"][:-1]:
        cap.drive_to(*wp)
    sim.drive_to_reach(*safe["waypoints"][-1])
    x, y, _ = sim.pose()
    if sample_id:
        if math.hypot(goal[0] - x, goal[1] - y) < 0.6:
            res = cap.sample(sample_id)
            lines.append(f"  reached it and reached out the arm — {'sample cached' if res.get('success') else res.get('reason')}.")
        else:
            lines.append(f"  stopped {math.hypot(goal[0]-x, goal[1]-y):.1f} m away.")
    else:
        lines.append(f"  arrived at ({x:+.1f}, {y:+.1f}).")
    return "\n".join(lines)


def execute(sim, env, cfg, action: str) -> str | None:
    """Carry out MARVIN's action string; returns a result line to print, or None for 'none'.
    Forgiving: finds a named target or a coordinate pair anywhere in the action."""
    a = (action or "").lower().strip()
    if not a or a.startswith("none"):
        return None
    nums = re.findall(r"-?\d+\.?\d*", a)
    coords = (float(nums[0]), float(nums[1])) if len(nums) >= 2 else None
    target = _named_target(sim, a)
    wants_collect = "collect" in a or "sample" in a or "pick" in a

    if wants_collect:
        if target:
            return _drive_safe(sim, env, cfg, target.xy, sample_id=target.id)
        t = _nearest_uncollected(sim, near=coords)
        if t is None:
            return "  (nothing left to collect)"
        return _drive_safe(sim, env, cfg, t.xy, sample_id=t.id)
    if coords:                                       # goto <x> <y>
        return _drive_safe(sim, env, cfg, coords)
    if target:                                       # "go to the outcrop" without coords
        return _drive_safe(sim, env, cfg, target.xy, sample_id=target.id)
    return None


def turn(sim, env, cfg, model, user_msg: str):
    """One conversational turn: returns (say, result) where result may be None."""
    say, action = model.converse(build_context(sim, env, cfg), user_msg)
    result = execute(sim, env, cfg, action)
    return say, result
