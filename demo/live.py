"""Live MuJoCo viewer + control console — the real 3D simulation window open, the rover moving as you
type. This is the "MuJoCo on one side, a TUI on the other" experience: drive the rover yourself, or
let IBM Granite route it around the Purgatory dune, and watch it happen in the physics window.

The window is interactive — orbit with the mouse, scroll to zoom; it re-centres on the rover as it
drives. Needs a display (won't work over a plain SSH session — use `demo.cli` there) and Ollama
running for the `route` command.

Run:  .venv/bin/python -m demo.live
"""
from __future__ import annotations

import math
import time

import mujoco
import mujoco.viewer
import yaml

from demo.purgatory import GOAL, TARGETS, dune_terrain
from planner import route, surrogate
from rover import capabilities as cap
from world.sim import MarsSim

STEP_PACE_S = 0.008        # small wall-clock pause per macro-step so the drive is watchable

HELP = """commands:
  route [x y]     lightsim + IBM Granite pick a SAFE route around the dune, then drive it
  drive <x> <y>   TELEOP: drive to a coordinate — try `drive 2.6 0` to auger into the dune
  sample [<id>]   grab the nearest reachable target (or a named one)
  status          pose, battery, targets
  reset           restart the scenario (keeps the window)
  help / quit"""


def main():
    cfg = yaml.safe_load(open("config.yaml"))
    plan_cfg = {**cfg["planner"], **cfg["constraints"], "energy_penalty_factor": 0.01}
    sim = MarsSim(seed=42, terrain=dune_terrain(), targets=TARGETS, render_mesh=True)
    cap.bind(sim, 1)
    model = {"m": None}

    def get_model():
        if model["m"] is None:
            from model.propose import Proposer
            mc = cfg["model"]
            print(f"  (loading {mc['name']} via Ollama at {mc['host']} ...)")
            model["m"] = Proposer(mc["name"], mc["host"], mc["temperature"])
        return model["m"]

    viewer = mujoco.viewer.launch_passive(sim.model, sim.data)
    with viewer.lock():
        viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 3.4, -22.0, 200.0
        x, y, _ = sim.pose()
        viewer.cam.lookat[:] = [x, y, 0.15]

    # animate the physics window as the rover steps, and keep it centred on the rover
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
        print(f"  pose=({x:+.2f}, {y:+.2f})  heading={math.degrees(yaw):+.0f}deg  "
              f"battery={sim.battery_pct:.1f}%")
        for t in sim.targets:
            d = math.hypot(t.xy[0] - x, t.xy[1] - y)
            print(f"    [{'x' if t.collected else ' '}] {t.id:10s} ({t.xy[0]:+.1f},{t.xy[1]:+.1f})  "
                  f"{d:4.2f} m away  [{t.mineral_class}]")

    print("=" * 60)
    print(" MARVIN — live physics window. Drive the rover, or type `route`.")
    print("=" * 60)
    print(HELP)
    print("\n  The dune (soft-soil trap) sits between the rover and the outcrop at "
          f"({GOAL[0]}, {GOAL[1]}).")
    status()

    while viewer.is_running():
        try:
            line = input("\nlive> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            viewer.sync()
            continue
        cmd, *args = line.split()

        if cmd in ("quit", "exit", "q"):
            break
        elif cmd == "help":
            print(HELP)
        elif cmd == "status":
            status()
        elif cmd == "drive":
            if len(args) != 2:
                print("  usage: drive <x> <y>")
                continue
            try:
                tx, ty = float(args[0]), float(args[1])
            except ValueError:
                print("  x and y must be numbers")
                continue
            print(f"  driving to ({tx:+.1f}, {ty:+.1f}) ...")
            r = cap.drive_to(tx, ty)
            x, y, _ = sim.pose()
            reached = math.hypot(x - tx, y - ty) < 0.3
            print(f"  {'arrived' if reached else 'STALLED (dune?)'} at ({x:+.2f}, {y:+.2f})  "
                  f"battery={sim.battery_pct:.1f}%")
        elif cmd == "sample":
            unc = [t for t in sim.targets if not t.collected]
            if not unc:
                print("  nothing left to sample")
                continue
            x, y, _ = sim.pose()
            tid = args[0] if args else min(unc, key=lambda t: math.hypot(t.xy[0] - x, t.xy[1] - y)).id
            print("  ", cap.sample(tid))
        elif cmd == "route":
            gx, gy = (float(args[0]), float(args[1])) if len(args) == 2 else GOAL
            print("  lightsim rolling out routes; Granite deciding ...")
            plan = route.plan_route(sim.pose()[:2], (gx, gy), surrogate.create_surrogate_env(sim, plan_cfg),
                                    get_model(), plan_cfg)
            for r in plan["assessed"]:
                print(f"    {r['name']:14s} entrapment risk {r['tail_risk'] * 100:3.0f}%  "
                      f"{'SAFE' if r['safe'] else 'WOULD GET STUCK'}")
            print(f"  DECISION ({plan['decided_by'].upper()}): {plan['rationale']}")
            print(f"  driving the {plan['chosen']} — watch the window ...")
            for wp in plan["waypoints"]:
                cap.drive_to(*wp)
            unc = [t for t in sim.targets if not t.collected]
            x, y, _ = sim.pose()
            if unc and math.hypot(unc[0].xy[0] - x, unc[0].xy[1] - y) < 0.6:
                print("  ", cap.sample(unc[0].id))
            status()
        elif cmd == "reset":
            sim.reset()
            viewer.sync()
            print("  scenario reset")
        else:
            print("  unknown command — type 'help'")
        viewer.sync()

    viewer.close()
    print("bye.")


if __name__ == "__main__":
    main()
