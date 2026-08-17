"""MARVIN's intent API — the Mars segment's interface to the link.

Receives a HOUSTON briefing, hands its objectives to the EXISTING onboard planner (perceive ->
A* candidates -> lightsim rollout -> Granite decides -> gate approves -> capability API drives), and
produces the downlink stream the console renders. Nothing here commands an actuator directly: motion
goes through the same `route.plan_route` + capability API the headless demos use.

The briefing's `route_advisory` is written from orbital data and is non-binding. When MARVIN's chosen
route differs from it, that deviation is downlinked prominently — Earth planned from orbit; the rover,
seeing the soft sand its own sensors found, knew better.
"""
from __future__ import annotations

import math

from demo import agent, scene
from planner import route, surrogate
from rover import capabilities as cap
from world.sim import MarsSim


def _coarse(grid, size=28):
    """Downsample a height grid to a compact list-of-lists for the ops-map panel (JSON-friendly)."""
    from planner.surrogate import downsample_terrain
    g = downsample_terrain(grid, size) if grid.shape[0] > size else grid
    lo, hi = float(g.min()), float(g.max())
    span = max(hi - lo, 1e-6)
    return [[round(float((v - lo) / span), 3) for v in row] for row in g]


class Rover:
    """One rover, one shared world — the onboard segment behind the deep-space link."""

    def __init__(self, cfg: dict, model=None):
        self.cfg = cfg
        self.sim = MarsSim(seed=42, terrain=scene.terrain(), targets=scene.TARGETS)
        cap.bind(self.sim, 1)
        self.env = surrogate.create_surrogate_env(self.sim, cfg)
        self.model = model

    def reset(self):
        self.sim = MarsSim(seed=42, terrain=scene.terrain(), targets=scene.TARGETS)
        cap.bind(self.sim, 1)
        self.env = surrogate.create_surrogate_env(self.sim, cfg=self.cfg)

    def _match_objective(self, objectives):
        """Map the briefing's primary sample objective onto a scene target (by mineral/name)."""
        for o in sorted(objectives, key=lambda o: o.get("priority", 9)):
            if o.get("type") != "sample":
                continue
            tok = str(o.get("target", "")).lower()
            for t in self.sim.targets:
                if not t.collected and (t.mineral_class.lower() in tok or t.id.lower() in tok
                                        or "carbonate" in tok):
                    return t, o
        unc = [t for t in self.sim.targets if not t.collected]
        return (unc[0], objectives[0]) if unc else (None, None)

    def run_briefing(self, brief: dict):
        """Execute the briefing; yield (kind, payload) downlink events in order."""
        yield "log", "Uplink received. Perception online — building the local cost map from onboard sensing."
        target, obj = self._match_objective(brief.get("objectives", []))
        if target is None:
            yield "log", "All briefed samples already cached. Standing by."
            return
        goal = target.xy
        yield "log", f"Primary objective: sample {target.id} ({target.mineral_class}) at " \
                     f"({goal[0]:+.1f}, {goal[1]:+.1f}). A* candidates + lightsim rollout, 20x each."

        plan = route.plan_route(self.sim.pose()[:2], goal, self.env, self.model, self.cfg)
        assessed = plan["assessed"]

        yield "panel", {"type": "ops_map", "grid": _coarse(self.env.terrain_full),
                        "extent": 5.0, "pose": list(self.sim.pose()),
                        "targets": [{"id": t.id, "x": t.xy[0], "y": t.xy[1], "done": t.collected}
                                    for t in self.sim.targets],
                        "routes": [{"name": a["name"], "waypoints": a["waypoints"],
                                    "risk": a["tail_risk"], "chosen": a["name"] == plan["chosen"]}
                                   for a in assessed]}
        yield "panel", {"type": "route_comparison",
                        "rows": [{"name": a["name"], "length_m": a["length_m"],
                                  "risk_pct": round(a["tail_risk"] * 100), "safe": a["safe"]}
                                 for a in assessed]}

        # Deviation report — advisory (orbital, direct) vs what MARVIN's own sensing chose.
        by = {a["name"]: a for a in assessed}
        direct = by.get("direct")
        chosen = by.get(plan["chosen"], assessed[0])
        deviated = plan["chosen"] != "direct" and direct is not None
        if deviated:
            yield "deviation", {
                "advisory": brief.get("route_advisory", {}).get("note", "direct approach"),
                "advisory_route": "direct", "advisory_risk_pct": round(direct["tail_risk"] * 100),
                "chosen_route": plan["chosen"], "chosen_risk_pct": round(chosen["tail_risk"] * 100),
                "chosen_length_m": chosen["length_m"], "decided_by": plan["decided_by"],
                "justification": plan["rationale"]}
        else:
            yield "log", f"Route holds with the advisory. {plan['rationale']}"

        # Execute on the true physics; telemetry ticks as it drives.
        for wp in chosen["waypoints"][:-1]:
            cap.drive_to(*wp)
            x, y, _ = self.sim.pose()
            yield "telemetry", {"pose": [round(x, 2), round(y, 2)], "battery_pct": round(self.sim.battery_pct, 1)}
        self.sim.drive_to_reach(*chosen["waypoints"][-1])
        x, y, _ = self.sim.pose()
        yield "telemetry", {"pose": [round(x, 2), round(y, 2)], "battery_pct": round(self.sim.battery_pct, 1)}
        yield "panel", {"type": "power", "battery_pct": round(self.sim.battery_pct, 1),
                        "reserve_pct": self.cfg.get("battery_reserve_pct", 15)}

        if math.hypot(goal[0] - x, goal[1] - y) < 0.6:
            res = cap.sample(target.id)
            ok = res.get("success")
            yield "log", "Arm deployed. Sample cached." if ok else f"Sample attempt failed: {res.get('reason')}"
            yield "summary", {"objective": obj.get("id", "obj-1"), "sampled": bool(ok),
                              "route": plan["chosen"], "battery_pct": round(self.sim.battery_pct, 1),
                              "note": "Earth planned from orbit; the rover routed on what it sensed."}
        else:
            yield "log", f"Stopped {math.hypot(goal[0]-x, goal[1]-y):.1f} m short — reassessing."

    def answer(self, question: str):
        """Rover-state question (relayed from HOUSTON): answered in MARVIN's voice via the existing
        conversation path. Returns a text string to downlink."""
        say, result = agent.turn(self.sim, self.env, self.cfg, self.model, question)
        return say + ("\n" + result if result else "")
