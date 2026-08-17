"""Live MuJoCo window + conversational MARVIN — the real 3D simulation open, and you talk to the
rover's onboard IBM Granite intelligence in plain language while you watch it act.

    live> what's the terrain like ahead?
    live> collect the carbonate sample      # MARVIN drives the safe route around the dune, on screen

The window is interactive (orbit with the mouse, scroll to zoom) and re-centres on the rover as it
drives. Needs a display (use demo.cli over SSH) and Ollama running.
Run:  .venv/bin/python -m demo.live
"""
from __future__ import annotations

import math
import time

import mujoco
import mujoco.viewer
import yaml

from demo import agent
from demo import scene
from planner import surrogate
from rover import capabilities as cap
from world.sim import MarsSim

STEP_PACE_S = 0.008

INTRO = """============================================================
 MARVIN — live physics window. Talk to the rover's Granite AI.
============================================================
Just type. Try:
  hi
  what's the terrain like around you?
  collect the carbonate sample     (watch it route around the steep mound)
Utility: 'status' · 'reset' · 'quit'"""


def main():
    cfg0 = yaml.safe_load(open("config.yaml"))
    cfg = {**cfg0["planner"], **cfg0["constraints"], "energy_penalty_factor": 0.01}
    sim = MarsSim(seed=42, terrain=scene.terrain(), targets=scene.TARGETS, render_mesh=True)
    cap.bind(sim, 1)
    env = surrogate.create_surrogate_env(sim, cfg)
    model = {"m": None}

    def get_model():
        if model["m"] is None:
            from model.propose import Proposer
            mc = cfg0["model"]
            print(f"  (waking MARVIN — {mc['name']} via Ollama ...)")
            model["m"] = Proposer(mc["name"], mc["host"], mc["temperature"])
        return model["m"]

    viewer = mujoco.viewer.launch_passive(sim.model, sim.data)
    cz = float(sim.data.xpos[sim._chassis][2])
    with viewer.lock():
        viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 1.6, -20.0, 205.0
        x, y, _ = sim.pose()
        viewer.cam.lookat[:] = [x, y, cz + 0.05]

    orig_step = sim.step
    def step(vL, vR, n=1):
        orig_step(vL, vR, n)
        if viewer.is_running():
            px, py, _ = sim.pose()
            with viewer.lock():
                viewer.cam.lookat[0], viewer.cam.lookat[1] = px, py
            viewer.sync()
            time.sleep(STEP_PACE_S)
    sim.step = step

    def status():
        x, y, yaw = sim.pose()
        cached = sum(t.collected for t in sim.targets)
        print(f"  pose=({x:+.2f}, {y:+.2f})  battery={sim.battery_pct:.0f}%  "
              f"cached={cached}/{len(sim.targets)}")

    print(INTRO)
    status()

    while viewer.is_running():
        try:
            line = input("\nlive> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            viewer.sync()
            continue
        low = line.lower()
        if low in ("quit", "exit", "q"):
            break
        if low in ("status", "map"):
            status()
            continue
        if low == "reset":
            sim.reset()
            viewer.sync()
            print("  (scenario reset)")
            status()
            continue
        if low in ("help", "?"):
            print(INTRO)
            continue

        print("  MARVIN is thinking ...")
        say, result = agent.turn(sim, env, cfg, get_model(), line)
        print(f"\nMARVIN: {say}")
        if result:
            print(result)
            status()
        viewer.sync()

    viewer.close()
    print("bye.")


if __name__ == "__main__":
    main()
