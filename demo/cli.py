"""Interactive MARVIN console — drive the rover yourself, or hand IBM Granite an objective and
watch it plan. Headless (no viewer), so it runs anywhere.

Teleop:   drive <x> <y> · sample [<id>] · scan · status · targets · reset
Autonomy: ask   -> Granite proposes the next action (with rationale), does NOT execute
          step  -> Granite decides AND executes one action
          run   -> Granite runs the whole mission autonomously

Needs Ollama running with the model in config.yaml (granite4.1:3b) for the autonomy commands.

Run:  .venv/bin/python -m demo.cli
"""
from __future__ import annotations

import logging
import math

import numpy as np
import yaml

from common.types import MissionState, Pose
from planner import loop, route, surrogate
from rover import capabilities as cap
from world.sim import TERRAIN_RADIUS, MarsSim

logging.disable(logging.INFO)   # keep the console clean; we print what matters ourselves


def _ascii_map(sim, width=41, height=17):
    """A compact top-down text map (cameras off): @ = rover, * = target, x = cached."""
    grid = [[" "] * width for _ in range(height)]

    def cell(x, y):
        c = int((x + TERRAIN_RADIUS) / (2 * TERRAIN_RADIUS) * (width - 1))
        r = int((TERRAIN_RADIUS - y) / (2 * TERRAIN_RADIUS) * (height - 1))
        return max(0, min(height - 1, r)), max(0, min(width - 1, c))

    # shade high ground / dunes (obstacles) as '^'
    n = sim.terrain.shape[0]
    lo, hi = float(sim.terrain.min()), float(sim.terrain.max())
    thresh = lo + 0.8 * (hi - lo)
    for rr in range(height):
        for cc in range(width):
            wx = -TERRAIN_RADIUS + cc / (width - 1) * 2 * TERRAIN_RADIUS
            wy = TERRAIN_RADIUS - rr / (height - 1) * 2 * TERRAIN_RADIUS
            i = min(n - 1, max(0, int((wy + TERRAIN_RADIUS) / (2 * TERRAIN_RADIUS) * (n - 1))))
            j = min(n - 1, max(0, int((wx + TERRAIN_RADIUS) / (2 * TERRAIN_RADIUS) * (n - 1))))
            if sim.terrain[i, j] > thresh:
                grid[rr][cc] = "^"

    for t in sim.targets:
        r, c = cell(*t.xy)
        grid[r][c] = "x" if t.collected else "*"
    x, y, _ = sim.pose()
    r, c = cell(x, y)
    grid[r][c] = "@"
    top = "+" + "-" * width + "+"
    print(top)
    for row in grid:
        print("|" + "".join(row) + "|")
    print(top)


def status(sim):
    x, y, yaw = sim.pose()
    coll = [t.id for t in sim.targets if t.collected]
    _ascii_map(sim)
    print(f"pose=({x:+.2f}, {y:+.2f})  heading={math.degrees(yaw):+.0f}deg  "
          f"battery={sim.battery_pct:.1f}%  cached={len(coll)}/2")
    for t in sim.targets:
        d = math.hypot(t.xy[0] - x, t.xy[1] - y)
        mark = "x" if t.collected else " "
        print(f"  [{mark}] {t.id:16s} ({t.xy[0]:+.1f},{t.xy[1]:+.1f})  {d:4.2f} m  "
              f"science={t.science_value:.2f}  [{t.mineral_class}]")


def _state(sim, perception):
    x, y, yaw = sim.pose()
    return MissionState(pose=Pose((x, y), yaw), battery_pct=sim.battery_pct,
                        sol_time=sim.sol_time, localization_sigma=1.0,
                        collected=tuple(t.id for t in sim.targets if t.collected),
                        remaining=tuple(perception.visible_targets))


HELP = """commands:
  status              pose, battery, cached samples, target list + text map
  targets             list targets and their science values
  scan                refresh perception
  drive <x> <y>       TELEOP: drive the rover to a coordinate (x,y in [-5,5])
  sample [<id>]       TELEOP: sample the nearest reachable target (or a named one)
  ask                 ask IBM Granite for the next action (shows plan + rationale, no execute)
  step                Granite decides AND executes one action
  run                 Granite runs the full mission autonomously
  dune                load the Purgatory dune scenario (a soft-soil trap between rover and goal)
  route               lightsim + Granite pick a SAFE route around the dune, then drive it
  reset               reset the mission
  help / quit"""


def main():
    cfg = yaml.safe_load(open("config.yaml"))
    plan_cfg = {**cfg["planner"], **cfg["constraints"], "energy_penalty_factor": 0.01}
    state = {"sim": MarsSim(seed=42)}
    cap.bind(state["sim"], 1)
    model = {"m": None}

    def get_model():
        if model["m"] is None:
            from model.propose import Proposer
            mc = cfg["model"]
            model["m"] = Proposer(mc["name"], mc["host"], mc["temperature"])
            print(f"  (using {mc['name']} via Ollama at {mc['host']})")
        return model["m"]

    def granite_decision():
        sim = state["sim"]
        p = cap.scan()
        env = surrogate.create_surrogate_env(sim, plan_cfg)
        return loop.decide_next_action(_state(sim, p), p, env, get_model(), plan_cfg, 0, {})

    print("=" * 60)
    print(" MARVIN console — drive the rover, or let IBM Granite plan.")
    print("=" * 60)
    print(HELP)
    status(state["sim"])

    while True:
        try:
            line = input("\nmarvin> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        cmd, *args = line.split()
        sim = state["sim"]

        if cmd in ("quit", "exit", "q"):
            break
        elif cmd == "help":
            print(HELP)
        elif cmd == "status":
            status(sim)
        elif cmd == "targets":
            for t in sim.targets:
                print(f"  {t.id:16s} ({t.xy[0]:+.1f},{t.xy[1]:+.1f})  "
                      f"science={t.science_value:.2f}  [{t.mineral_class}]  "
                      f"{'cached' if t.collected else 'pending'}")
        elif cmd == "scan":
            p = cap.scan()
            print(f"  {len(p.visible_targets)} target(s) in view · dust tau={p.dust_tau:.2f} · "
                  f"local slope up to {float(p.slope_deg.max()):.1f}deg")
        elif cmd == "drive":
            if len(args) != 2:
                print("  usage: drive <x> <y>")
                continue
            try:
                x, y = float(args[0]), float(args[1])
            except ValueError:
                print("  x and y must be numbers")
                continue
            print(f"  driving to ({x:+.1f}, {y:+.1f}) ...")
            print("  ", cap.drive_to(x, y))
        elif cmd == "sample":
            if args:
                print("  ", cap.sample(args[0]))
            else:
                x, y, _ = sim.pose()
                unc = [t for t in sim.targets if not t.collected]
                if not unc:
                    print("  nothing left to sample")
                    continue
                nearest = min(unc, key=lambda t: math.hypot(t.xy[0] - x, t.xy[1] - y))
                print(f"  sampling nearest ({nearest.id}) ...")
                print("  ", cap.sample(nearest.id))
        elif cmd in ("ask", "plan"):
            print("  Granite is thinking ...")
            d = granite_decision()
            print(f"  PROPOSED: {d.action.kind.name} {d.action.params}")
            print(f"  WHY: {d.rationale}")
        elif cmd == "step":
            print("  Granite is thinking ...")
            d = granite_decision()
            print(f"  DECISION: {d.action.kind.name} {d.action.params}")
            print(f"  WHY: {d.rationale}")
            a = d.action
            if a.kind.name == "DRIVE":
                cap.drive_to(a.params["xy"][0], a.params["xy"][1])
            elif a.kind.name == "SAMPLE":
                print("  ", cap.sample(a.params["target"]))
            elif a.kind.name in ("SCAN", "OBSERVE"):
                cap.scan()
            status(sim)
        elif cmd == "run":
            obj = " ".join(args) or cfg["mission"]["objective"]
            print(f"  running mission: {obj}")
            log = loop.run_mission(obj, sim, get_model(), plan_cfg, max_steps=30)
            for i, d in enumerate(log.decisions, 1):
                print(f"   {i:02d}. {d.action.kind.name:8s} {d.rationale[:90]}")
            print(f"  => {log.samples_collected}/2 samples, {len(log.decisions)} decisions, "
                  f"{'SUCCESS' if log.success else 'incomplete'}")
            status(sim)
        elif cmd == "dune":
            from demo.purgatory import TARGETS, dune_terrain
            state["sim"] = MarsSim(seed=42, terrain=dune_terrain(), targets=TARGETS)
            cap.bind(state["sim"], 1)
            print("  Purgatory dune scenario loaded — a dune (^) blocks the direct path to the")
            print("  outcrop (like the ripple that stuck Opportunity for 38 sols in 2005). Try 'route'.")
            status(state["sim"])
        elif cmd == "route":
            import math as _m
            from demo.purgatory import GOAL
            gx, gy = (float(args[0]), float(args[1])) if len(args) == 2 else GOAL
            env = surrogate.create_surrogate_env(sim, plan_cfg)
            print("  lightsim rolling out routes; Granite deciding ...")
            plan = route.plan_route(sim.pose()[:2], (gx, gy), env, get_model(), plan_cfg)
            for r in plan["assessed"]:
                print(f"    {r['name']:14s} entrapment risk {r['tail_risk'] * 100:3.0f}%  "
                      f"{'SAFE' if r['safe'] else 'WOULD GET STUCK'}")
            print(f"  DECISION ({plan['decided_by'].upper()}): {plan['rationale']}")
            print(f"  driving the {plan['chosen']} ...")
            for wp in plan["waypoints"][:-1]:
                cap.drive_to(*wp)
            sim.drive_to_reach(*plan["waypoints"][-1])      # pull up beside the outcrop, not onto it
            unc = [t for t in sim.targets if not t.collected]
            if unc:
                x, y, _ = sim.pose()
                if _m.hypot(unc[0].xy[0] - x, unc[0].xy[1] - y) < 0.6:
                    print("  reaching out the arm ...")
                    print("  ", cap.sample(unc[0].id))
            status(sim)
        elif cmd == "reset":
            state["sim"] = MarsSim(seed=42)
            cap.bind(state["sim"], 1)
            print("  mission reset")
        else:
            print("  unknown command — type 'help'")

    print("bye.")


if __name__ == "__main__":
    main()
