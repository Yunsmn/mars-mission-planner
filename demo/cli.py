"""MARVIN console — talk to the rover's onboard IBM Granite intelligence in plain language.

Ask it about the terrain and science around it, ask what it is, or tell it to collect a sample —
it reasons over the space data, picks the lightsim-verified safe route (A* navigator), drives, and
samples. Headless (draws an ASCII map), so it runs anywhere, including over SSH.

    you> hi
    you> what's the terrain like around you?
    you> collect the carbonate sample

Needs Ollama running with the model in config.yaml (granite4.1:3b).
Run:  .venv/bin/python -m demo.cli
"""
from __future__ import annotations

import logging
import math

import yaml

from demo import agent
from demo import scene
from planner import surrogate
from rover import capabilities as cap
from world.sim import TERRAIN_RADIUS, MarsSim

logging.disable(logging.INFO)


def _ascii_map(sim, width=41, height=17):
    """Top-down text map: @ = rover, * = sample, x = cached, ^ = high ground / dune."""
    grid = [[" "] * width for _ in range(height)]
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

    def cell(x, y):
        c = int((x + TERRAIN_RADIUS) / (2 * TERRAIN_RADIUS) * (width - 1))
        r = int((TERRAIN_RADIUS - y) / (2 * TERRAIN_RADIUS) * (height - 1))
        return max(0, min(height - 1, r)), max(0, min(width - 1, c))

    for t in sim.targets:
        r, c = cell(*t.xy)
        grid[r][c] = "x" if t.collected else "*"
    x, y, _ = sim.pose()
    r, c = cell(x, y)
    grid[r][c] = "@"
    bar = "+" + "-" * width + "+"
    print(bar)
    for row in grid:
        print("|" + "".join(row) + "|")
    print(bar)


def status(sim):
    x, y, yaw = sim.pose()
    _ascii_map(sim)
    cached = sum(t.collected for t in sim.targets)
    print(f"pose=({x:+.2f}, {y:+.2f})  heading={math.degrees(yaw):+.0f}deg  "
          f"battery={sim.battery_pct:.1f}%  cached={cached}/{len(sim.targets)}")


INTRO = """============================================================
 MARVIN console — talk to the rover's onboard Granite AI.
============================================================
Just type. Try:
  hi
  what's the terrain like around you?
  what samples can you see?
  collect the carbonate sample
Utility: 'map' (show the map) · 'reset' · 'quit'
The terrain is varied (rolling hills/hollows); '^' marks high/steep ground — a soft-sand mound sits
between the rover and the outcrop."""


def main():
    cfg0 = yaml.safe_load(open("config.yaml"))
    cfg = {**cfg0["planner"], **cfg0["constraints"], "energy_penalty_factor": 0.01}
    state = {"sim": MarsSim(seed=42, terrain=scene.terrain(), targets=scene.TARGETS)}
    cap.bind(state["sim"], 1)
    env = {"e": surrogate.create_surrogate_env(state["sim"], cfg)}
    model = {"m": None}

    def get_model():
        if model["m"] is None:
            from model.propose import Proposer
            mc = cfg0["model"]
            print(f"  (waking MARVIN — {mc['name']} via Ollama ...)")
            model["m"] = Proposer(mc["name"], mc["host"], mc["temperature"])
        return model["m"]

    print(INTRO)
    status(state["sim"])

    while True:
        try:
            line = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        low = line.lower()
        if low in ("quit", "exit", "q"):
            break
        if low in ("map", "status"):
            status(state["sim"])
            continue
        if low == "reset":
            state["sim"] = MarsSim(seed=42, terrain=scene.terrain(), targets=scene.TARGETS)
            cap.bind(state["sim"], 1)
            env["e"] = surrogate.create_surrogate_env(state["sim"], cfg)
            print("  (scenario reset)")
            status(state["sim"])
            continue
        if low in ("help", "?"):
            print(INTRO)
            continue

        print("  MARVIN is thinking ...")
        say, result = agent.turn(state["sim"], env["e"], cfg, get_model(), line)
        print(f"\nMARVIN: {say}")
        if result:
            print(result)
            status(state["sim"])

    print("bye.")


if __name__ == "__main__":
    main()
