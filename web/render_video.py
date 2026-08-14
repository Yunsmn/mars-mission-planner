"""Render AND record ONE real IBM Granite mission on the real Jezero terrain in a single pass:
writes the showcase video (web/vendor/mission.mp4) and the measured record
(data/derived/mission_record.json) from the same run, so the page's video and telemetry always
agree. Uses MuJoCo offscreen rendering (glfw) — a genuine render of the simulation.

Run:  .venv/bin/python -m web.render_video
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

from model.propose import Proposer
from planner.loop import run_mission
from world.sim import TERRAIN_RADIUS, MarsSim

W, H, FPS = 640, 360, 30


def _fmt(a) -> dict:
    p = a.params or {}
    to = p.get("target") or (f"({p['xy'][0]:.1f}, {p['xy'][1]:.1f})" if p.get("xy") else "")
    return {"act": a.kind.name, "to": to}


def run():
    cfg = yaml.safe_load(open("config.yaml"))
    sim = MarsSim(seed=42)

    r = mujoco.Renderer(sim.model, H, W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, 0.1]
    cam.distance, cam.elevation, cam.azimuth = 5.2, -40.0, 115.0
    frames = []

    def shoot():
        cam.azimuth += 0.4
        r.update_scene(sim.data, cam)
        frames.append(r.render().copy())

    # capture frames while the rover actually drives, and hold on each cached sample
    ctr = [0]
    orig_step = sim.step
    def step(vL, vR, n=1):
        orig_step(vL, vR, n)
        ctr[0] += 1
        if ctr[0] % 4 == 0:
            shoot()
    sim.step = step

    collect_at = {}
    orig_sample = sim.sample
    def sample(tid, **kw):
        collect_at[tid] = len(sim.trajectory)
        res = orig_sample(tid, **kw)
        for _ in range(8):
            shoot()
        return res
    sim.sample = sample

    for _ in range(15):
        shoot()

    model = Proposer(cfg["model"]["name"], cfg["model"]["host"], cfg["model"]["temperature"])
    plan_cfg = {**cfg["planner"], **cfg["constraints"], "energy_penalty_factor": 0.01}
    log = run_mission(cfg["mission"]["objective"], sim, model, plan_cfg, max_steps=30)

    traj = [[round(float(x), 3), round(float(y), 3)] for (x, y) in sim.trajectory]
    dist = sum(math.dist(traj[i - 1], traj[i]) for i in range(1, len(traj)))
    data = {
        "grid": int(sim.terrain.shape[0]), "extent": float(TERRAIN_RADIUS),
        "terrain": [[round(float(v), 4) for v in row] for row in sim.terrain],
        "trajectory": traj,
        "targets": [{"id": t.id, "x": round(t.xy[0], 3), "y": round(t.xy[1], 3),
                     "collectAt": collect_at.get(t.id, len(traj))} for t in sim.targets],
        "decisions": [_fmt(d.action) for d in log.decisions],
        "stats": {"decisions": len(log.decisions), "samples": int(log.samples_collected),
                  "distance_m": round(dist, 2), "battery_used_pct": round(100.0 - sim.battery_pct, 1),
                  "sim_seconds": round(sim.sol_time, 1)},
    }
    os.makedirs("data/derived", exist_ok=True)
    json.dump(data, open("data/derived/mission_record.json", "w"))

    tmp = tempfile.mkdtemp()
    for i, f in enumerate(frames):
        Image.fromarray(f).save(f"{tmp}/f{i:05d}.png")
    os.makedirs("web/vendor", exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", f"{tmp}/f%05d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", "web/vendor/mission.mp4"], check=True)
    print(f"stats: {data['stats']} | frames: {len(frames)} | "
          f"mp4 {os.path.getsize('web/vendor/mission.mp4') // 1024} KB")


if __name__ == "__main__":
    run()
