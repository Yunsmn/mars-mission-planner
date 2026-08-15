"""Render the Purgatory Dune mission as a real MuJoCo video, with the NASA Perseverance mesh.

One pass produces web/vendor/purgatory.mp4 (a tracking chase-cam of the rover taking IBM Granite's
lightsim-verified route around the dune and sampling the outcrop) and data/derived/purgatory_record.json
(the route assessment + measured stats), so the showcase video and telemetry come from the same run.

Run:  .venv/bin/python -m web.render_purgatory
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile

os.environ.setdefault("MUJOCO_GL", "glfw")

import mujoco
import yaml
from PIL import Image

from demo.purgatory import GOAL, REAL_STUCK_SOLS, TARGETS, dune_terrain
from model.propose import Proposer
from planner import route, surrogate
from world.sim import MarsSim

W, H, FPS = 720, 480, 30
SHOOT_EVERY = 3       # capture one frame per N physics macro-steps while driving


def _cam():
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation, cam.azimuth = 1.7, -32.0, 205.0
    return cam


def run():
    cfg0 = yaml.safe_load(open("config.yaml"))
    cfg = {**cfg0["planner"], **cfg0["constraints"], "energy_penalty_factor": 0.01}
    terrain = dune_terrain()

    # 1) IBM Granite decides the route from the lightsim's entrapment-risk assessment.
    plan_sim = MarsSim(seed=42, terrain=terrain, targets=TARGETS)
    env = surrogate.create_surrogate_env(plan_sim, cfg)
    m = cfg0["model"]
    model = Proposer(m["name"], m["host"], m["temperature"])
    plan = route.plan_route((0.0, 0.0), GOAL, env, model, cfg)

    # 2) Render the chosen route with the photoreal rover on the ground-truth physics.
    sim = MarsSim(seed=42, terrain=terrain, targets=TARGETS, render_mesh=True)
    r = mujoco.Renderer(sim.model, H, W)
    cam = _cam()
    frames = []

    def shoot(hold: int = 1):
        x, y, _ = sim.pose()
        cz = float(sim.data.xpos[sim._chassis][2])
        cam.lookat[:] = [x, y, cz]
        cam.azimuth += 0.35          # slow orbit while tracking the rover
        r.update_scene(sim.data, cam)
        f = r.render().copy()
        frames.extend([f] * hold)

    ctr = [0]
    orig_step = sim.step
    def step(vL, vR, n=1):
        orig_step(vL, vR, n)
        ctr[0] += 1
        if ctr[0] % SHOOT_EVERY == 0:
            shoot()
    sim.step = step

    for _ in range(28):              # opening hold on the rover at the start line
        shoot()
    for wp in plan["waypoints"][:-1]:
        sim.drive_to(*wp)
    sim.drive_to_reach(*plan["waypoints"][-1])   # pull up beside the outcrop, not onto it
    x, y, _ = sim.pose()
    reached = math.hypot(x - GOAL[0], y - GOAL[1]) < 0.6
    sampled = bool(reached and sim.sample("outcrop").get("success"))
    for _ in range(34):              # closing hold on the sampled outcrop
        shoot()

    traj = [[round(float(px), 3), round(float(py), 3)] for (px, py) in sim.trajectory]
    dist = sum(math.dist(traj[i - 1], traj[i]) for i in range(1, len(traj)))
    record = {
        "event": "Opportunity — Purgatory Dune (Meridiani Planum, sol 446, 26 Apr 2005)",
        "real_stuck_sols": REAL_STUCK_SOLS,
        "assessed_routes": plan["assessed"], "decided_by": plan["decided_by"],
        "chosen_route": plan["chosen"], "rationale": plan["rationale"],
        "reached": reached, "sampled": sampled,
        "distance_m": round(dist, 2), "battery_used_pct": round(100.0 - sim.battery_pct, 1),
    }
    os.makedirs("data/derived", exist_ok=True)
    json.dump(record, open("data/derived/purgatory_record.json", "w"), indent=2)

    tmp = tempfile.mkdtemp()
    for i, f in enumerate(frames):
        Image.fromarray(f).save(f"{tmp}/f{i:05d}.png")
    os.makedirs("web/vendor", exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", f"{tmp}/f%05d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", "web/vendor/purgatory.mp4"], check=True)
    print(f"route={plan['chosen']} by {plan['decided_by']} | reached={reached} sampled={sampled} | "
          f"{len(frames)} frames | mp4 {os.path.getsize('web/vendor/purgatory.mp4') // 1024} KB")


if __name__ == "__main__":
    run()
