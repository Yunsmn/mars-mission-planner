"""MARVIN's intent API — the Mars segment's interface to the link.

Handles three things the ground can send over the link:
  - run_briefing(brief)  : a HOUSTON science briefing -> the full mission (perceive -> A* -> lightsim
                           -> Granite decides -> gate -> drive -> sample), with the deviation report.
  - run_command(text)    : any free command ("drive to 2,1", "collect the outcrop") -> MARVIN's own
                           conversation decides the action and carries it out on the real physics.
  - answer(question)     : a rover-state question -> MARVIN describes what its sensors see, no motion.

Nothing here commands an actuator directly: motion goes through route.plan_route + the capability API
(the same gate the headless demos use). Numbers reported to the console come from the live sim state.
"""
from __future__ import annotations

import base64
import io
import math
import os
import re
import time

os.environ.setdefault("MUJOCO_GL", "glfw")   # headless deploys should set this to egl

from demo import agent, scene
from planner import route, surrogate
from rover import capabilities as cap
from world.sim import MarsSim

RENDER_W, RENDER_H, JPEG_Q = 480, 340, 52    # streamed satellite/chase frames
FRAME_EVERY = 8                              # render one frame per N drive macro-steps
FRAME_PACE_S = 0.04                          # pace the drive so it's watchable, not a blur


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
        self.model = model
        self.cam_mode = "satellite"          # "satellite" (overhead) | "chase" (behind the rover)
        self.renderer = None
        self.reset()

    def reset(self):
        self.sim = MarsSim(seed=42, terrain=scene.terrain(), targets=scene.TARGETS, render_mesh=True)
        cap.bind(self.sim, 1)
        self.env = surrogate.create_surrogate_env(self.sim, self.cfg)
        self._init_renderer()

    def set_camera(self, mode: str):
        self.cam_mode = "chase" if "chase" in (mode or "").lower() else "satellite"

    def _init_renderer(self):
        if self.renderer is not None:
            try:
                self.renderer.close()
            except Exception:
                pass
            self.renderer = None
        try:                                  # no GL (e.g. a headless deploy without egl) -> no stream
            import mujoco
            self.renderer = mujoco.Renderer(self.sim.model, RENDER_H, RENDER_W)
        except Exception:
            self.renderer = None

    def _camera(self):
        import mujoco
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        x, y, yaw = self.sim.pose()
        cz = float(self.sim.data.xpos[self.sim._chassis][2])
        if self.cam_mode == "chase":
            cam.lookat[:] = [x, y, cz]
            cam.distance, cam.elevation, cam.azimuth = 2.4, -20.0, math.degrees(yaw) + 180.0
        else:                                 # satellite: high, near-overhead, tracking the rover
            cam.lookat[:] = [x, y, 0.1]
            cam.distance, cam.elevation, cam.azimuth = 7.5, -84.0, 90.0
        return cam

    def _frame(self):
        if self.renderer is None:
            return None
        try:
            self.renderer.update_scene(self.sim.data, self._camera())
            buf = io.BytesIO()
            from PIL import Image
            Image.fromarray(self.renderer.render()).save(buf, "JPEG", quality=JPEG_Q)
            return base64.b64encode(buf.getvalue()).decode()
        except Exception:
            return None

    def state(self) -> dict:
        """The live numbers the console and MARVIN's answers must agree on."""
        x, y, yaw = self.sim.pose()
        return {"pose": [round(x, 2), round(y, 2), round(yaw, 3)],
                "battery_pct": round(self.sim.battery_pct, 1),
                "cached": [t.id for t in self.sim.targets if t.collected]}

    # ---- shared drive: plan a route to a goal, verify, drive, (optionally) sample ---------------
    def _drive_to_goal(self, goal, sample_id=None, advisory_note=None):
        plan = route.plan_route(self.sim.pose()[:2], goal, self.env, self.model, self.cfg)
        assessed = plan["assessed"]
        by = {a["name"]: a for a in assessed}
        chosen = by.get(plan["chosen"], assessed[0])

        yield "panel", {"type": "ops_map", "grid": _coarse(self.env.terrain_full), "extent": 5.0,
                        "pose": list(self.sim.pose()),
                        "targets": [{"id": t.id, "x": t.xy[0], "y": t.xy[1], "done": t.collected}
                                    for t in self.sim.targets],
                        "routes": [{"name": a["name"], "waypoints": a["waypoints"],
                                    "risk": a["tail_risk"], "chosen": a["name"] == plan["chosen"]}
                                   for a in assessed]}
        yield "panel", {"type": "route_comparison",
                        "rows": [{"name": a["name"], "length_m": a["length_m"],
                                  "risk_pct": round(a["tail_risk"] * 100), "safe": a["safe"]}
                                 for a in assessed]}

        direct = by.get("direct")
        if advisory_note and plan["chosen"] != "direct" and direct is not None:
            yield "deviation", {"advisory": advisory_note, "advisory_route": "direct",
                                "advisory_risk_pct": round(direct["tail_risk"] * 100),
                                "chosen_route": plan["chosen"],
                                "chosen_risk_pct": round(chosen["tail_risk"] * 100),
                                "chosen_length_m": chosen["length_m"], "decided_by": plan["decided_by"],
                                "justification": plan["rationale"]}
        else:
            yield "log", plan["rationale"]

        wps = chosen["waypoints"]
        f0 = self._frame()
        if f0:
            yield "frame", f0
        for i, wp in enumerate(wps):
            if i == len(wps) - 1:                 # stop ~0.35 m short of the final target, facing it
                px, py, _ = self.sim.pose()
                d = math.hypot(wp[0] - px, wp[1] - py)
                gx, gy = ((wp[0] - (wp[0] - px) / d * 0.35, wp[1] - (wp[1] - py) / d * 0.35)
                          if d > 0.35 else (px, py))
            else:
                gx, gy = wp
            for _ in self.sim.drive_iter(gx, gy):
                fr = self._frame()
                if fr:
                    yield "frame", fr
                yield "telemetry", self.state()
                time.sleep(FRAME_PACE_S)

        x, y, _ = self.sim.pose()
        sampled = False
        if sample_id and math.hypot(goal[0] - x, goal[1] - y) < 0.6:
            res = cap.sample(sample_id)
            sampled = bool(res.get("success"))
            yield "log", "Arm deployed. Sample cached." if sampled else f"Sample attempt: {res.get('reason')}"
        elif not sample_id:
            yield "log", f"Arrived at ({x:+.1f}, {y:+.1f})."
        # power + summary AFTER the sample, so every number the console shows agrees with state()
        st = self.state()
        yield "panel", {"type": "power", "battery_pct": st["battery_pct"],
                        "reserve_pct": self.cfg.get("battery_reserve_pct", 15)}
        yield "summary", {"route": plan["chosen"], "sampled": sampled, "reached": True,
                          "battery_pct": st["battery_pct"], "pose": st["pose"],
                          "note": "Earth planned from orbit; the rover routed on what it sensed."}

    # ---- a HOUSTON briefing -> the full science mission -----------------------------------------
    def _match_objective(self, objectives):
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
        yield "log", "Uplink received. Perception online — building the local cost map from onboard sensing."
        target, obj = self._match_objective(brief.get("objectives", []))
        if target is None:
            yield "log", "All briefed samples already cached. Standing by."
            return
        goal = target.xy
        yield "log", (f"Primary objective: sample {target.id} ({target.mineral_class}) at "
                      f"({goal[0]:+.1f}, {goal[1]:+.1f}). A* candidates + lightsim rollout, 20x each.")
        yield from self._drive_to_goal(goal, sample_id=target.id,
                                       advisory_note=brief.get("route_advisory", {}).get("note"))

    # ---- a free command -> MARVIN's own conversation decides + acts -----------------------------
    def run_command(self, text: str):
        say, action = self.model.converse(agent.build_context(self.sim, self.env, self.cfg), text)
        yield "say", say
        a = (action or "").lower().strip()
        if not a or a.startswith("none"):
            return                                # it was talk, not an order
        nums = re.findall(r"-?\d+\.?\d*", a)
        coords = (float(nums[0]), float(nums[1])) if len(nums) >= 2 else None
        target = agent._named_target(self.sim, a)
        if any(w in a for w in ("collect", "sample", "pick")):
            t = target or agent._nearest_uncollected(self.sim, near=coords)
            if t is None:
                yield "log", "No uncollected sample in range."
            else:
                yield from self._drive_to_goal(t.xy, sample_id=t.id)
        elif coords:
            yield "log", f"Plotting a route to ({coords[0]:+.1f}, {coords[1]:+.1f})."
            yield from self._drive_to_goal(coords, sample_id=None)
        elif target:
            yield from self._drive_to_goal(target.xy, sample_id=target.id)
        # else: no drivable target — the spoken reply already stands
