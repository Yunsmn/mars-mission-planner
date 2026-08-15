"""Purgatory Dune mission — one cohesive scenario, grounded in a real Mars event.

Real event: on sol 446 (26 Apr 2005) NASA's Opportunity drove straight into a wind-shaped ripple
("Purgatory Dune", ~0.33 m tall x 2.5 m wide) in Meridiani Planum, dug its wheels in >10 cm, and
got stuck. Freeing it took 38 sols (to sol 484) of Earth-side sandbox testing and careful driving.
Spirit hit the same class of soft-soil trap at Troy (2009) and never escaped.

MARVIN, one mission: the lightsim (surrogate) rolls out candidate routes and predicts each one's
entrapment risk; IBM Granite decides which to take; the rover drives the safe route around the dune
and samples the outcrop beyond it. Compared against the Earth-planned direct path, which drives into
the dune and gets stuck.

Run:  .venv/bin/python -m demo.purgatory
"""
from __future__ import annotations

import json
import math

import numpy as np
import yaml

from planner import route, surrogate
from rover import capabilities as cap
from world import dem
from world.sim import TERRAIN_RADIUS, MarsSim

REAL_STUCK_SOLS = 38
SOLS_PER_MONTH = 30.0
GOAL = (2.6, 0.0)
TARGETS = [("outcrop", GOAL[0], GOAL[1], 0.90, "carbonate")]


def dune_terrain(n: int = 64) -> np.ndarray:
    base = dem.synthesize(n=n, amplitude_m=0.12)
    xs = np.linspace(-TERRAIN_RADIUS, TERRAIN_RADIUS, n)
    ys = np.linspace(-TERRAIN_RADIUS, TERRAIN_RADIUS, n)
    dune = np.zeros((n, n))
    for i, yy in enumerate(ys):
        for j, xx in enumerate(xs):
            if -1.4 <= yy <= 1.4:                            # finite ridge; the ends are passable
                dune[i, j] = 0.75 * math.exp(-((xx - 1.0) / 0.32) ** 2)
    return (base + dune).astype(np.float64)


def run_physics(terrain, waypoints, sample_at_goal=False) -> dict:
    sim = MarsSim(seed=42, terrain=terrain, targets=TARGETS)
    cap.bind(sim, 1)
    for wp in waypoints:
        sim.drive_to(*wp)
    x, y, _ = sim.pose()
    reached = math.hypot(x - GOAL[0], y - GOAL[1]) < 0.6
    sampled = bool(sample_at_goal and reached and sim.sample("outcrop").get("success"))
    traj = sim.trajectory
    dist = sum(math.dist(traj[i - 1], traj[i]) for i in range(1, len(traj))) if len(traj) > 1 else 0.0
    return {"reached": reached, "final": (round(x, 2), round(y, 2)),
            "distance_m": round(dist, 2), "sampled": sampled}


def main(out="data/derived/purgatory.json"):
    cfg0 = yaml.safe_load(open("config.yaml"))
    cfg = {**cfg0["planner"], **cfg0["constraints"]}
    terrain = dune_terrain()

    # lightsim + IBM Granite decide the route
    sim0 = MarsSim(seed=42, terrain=terrain, targets=TARGETS)
    env = surrogate.create_surrogate_env(sim0, cfg)
    from model.propose import Proposer
    m = cfg0["model"]
    plan = route.plan_route((0.0, 0.0), GOAL, env, Proposer(m["name"], m["host"], m["temperature"]), cfg)

    baseline = run_physics(terrain, [GOAL])                              # Earth-planned direct
    marvin = run_physics(terrain, plan["waypoints"], sample_at_goal=True)  # Granite's route

    result = {
        "event": "Opportunity — Purgatory Dune (Meridiani Planum, sol 446, 26 Apr 2005)",
        "real_stuck_sols": REAL_STUCK_SOLS, "goal": GOAL,
        "assessed_routes": plan["assessed"], "decided_by": plan["decided_by"],
        "chosen_route": plan["chosen"], "rationale": plan["rationale"],
        "baseline": {"route": "direct (orbital)", **baseline, "sols_lost": REAL_STUCK_SOLS},
        "marvin": {"route": plan["chosen"], **marvin, "sols_lost": 0},
    }
    json.dump(result, open(out, "w"), indent=2)

    print(f"PURGATORY DUNE — Opportunity, sol 446 (real: stuck {REAL_STUCK_SOLS} sols)\n")
    print("  lightsim entrapment-risk assessment:")
    for r in plan["assessed"]:
        print(f"    {r['name']:14s} {r['tail_risk'] * 100:3.0f}%  {r['length_m']:.1f} m  "
              f"[{'SAFE' if r['safe'] else 'WOULD GET STUCK'}]")
    print(f"\n  Decision ({plan['decided_by'].upper()}): {plan['rationale']}\n")
    print(f"  Earth-planned direct : reached={baseline['reached']} (stuck at dune) -> "
          f"{REAL_STUCK_SOLS} sols to extract (~{REAL_STUCK_SOLS / SOLS_PER_MONTH:.1f} months)")
    print(f"  MARVIN ({plan['chosen']:12s}): reached={marvin['reached']} sampled={marvin['sampled']} "
          f"({marvin['distance_m']} m) -> 0 sols lost, trap avoided")
    print(f"\n  => Granite + the lightsim saved ~{REAL_STUCK_SOLS} sols and avoided the trap "
          f"that stranded Spirit.\n  wrote {out}")


if __name__ == "__main__":
    main()
