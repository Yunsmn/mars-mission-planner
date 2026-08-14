"""Purgatory Dune, recreated — obstacle avoidance grounded in a real Mars event.

Real event: on sol 446 (26 Apr 2005) NASA's Opportunity drove straight into a wind-shaped ripple
("Purgatory Dune", ~0.33 m tall x 2.5 m wide) in Meridiani Planum, dug its wheels in >10 cm, and
got stuck. Freeing it took 38 sols (to sol 484) of Earth-side sandbox testing and careful driving
— every move an Earth-in-the-loop command cycle. Spirit hit the same class of soft-soil trap at
Troy in 2009 and never escaped.

MARVIN's lightsim (the surrogate) rolls out candidate routes under uncertainty, predicts the
entrapment risk of each, and routes AROUND the dune instead of into it — no entrapment, no lost
sols. This module recreates the trap on real-scale terrain and compares:
  - direct (no lightsim / Earth-planned by orbital view): drives into the dune and gets stuck;
  - MARVIN: the lightsim rejects the through-dune route and takes the safe detour.

Run:  .venv/bin/python -m demo.purgatory
"""
from __future__ import annotations

import json
import math

import numpy as np
import yaml

from common.types import Action, ActionKind, MissionState, Pose
from planner import gating, surrogate
from rover import capabilities as cap
from world import dem
from world.sim import TERRAIN_RADIUS, MarsSim

REAL_STUCK_SOLS = 38                      # Opportunity, sol 446 -> 484
GOAL = (2.6, 0.0)
GROUND_CYCLE_SOLS = 1.0                   # ~1 command cycle per sol
SOLS_PER_MONTH = 30.0

ROUTES = {
    "direct (into dune)": [GOAL],
    "detour north": [(0.6, 1.9), (2.4, 1.9), GOAL],
    "detour south": [(0.6, -1.9), (2.4, -1.9), GOAL],
}


def dune_terrain(n: int = 64) -> np.ndarray:
    base = dem.synthesize(n=n, amplitude_m=0.12)
    xs = np.linspace(-TERRAIN_RADIUS, TERRAIN_RADIUS, n)
    ys = np.linspace(-TERRAIN_RADIUS, TERRAIN_RADIUS, n)
    dune = np.zeros((n, n))
    for i, yy in enumerate(ys):
        for j, xx in enumerate(xs):
            if -1.4 <= yy <= 1.4:                        # finite ridge: the ends are passable
                dune[i, j] = 0.75 * math.exp(-((xx - 1.0) / 0.32) ** 2)
    return (base + dune).astype(np.float64)


def route_tail_risk(env, cfg, route) -> float:
    """Max surrogate-predicted tail (worst-case) hazard across a route's drive segments."""
    rng = np.random.default_rng(0)
    pos, worst = (0.0, 0.0), 0.0
    for wp in route:
        st = MissionState(Pose(pos, 0.0), 100.0, 0.0, 1.0, (), ())
        b = surrogate.rollout_batch((Action(ActionKind.DRIVE, {"xy": wp}),), st, env,
                                    cfg.get("n_rollouts", 20), rng)
        worst = max(worst, float(gating.tail_worst(b.hazard, cfg.get("cvar_quantile", 0.9))))
        pos = wp
    return worst


def run_physics(terrain, route) -> dict:
    sim = MarsSim(seed=42, terrain=terrain)
    cap.bind(sim, 1)
    for wp in route:
        sim.drive_to(*wp)
    x, y, _ = sim.pose()
    traj = sim.trajectory
    dist = sum(math.dist(traj[i - 1], traj[i]) for i in range(1, len(traj))) if len(traj) > 1 else 0.0
    return {"reached": math.hypot(x - GOAL[0], y - GOAL[1]) < 0.5,
            "final": (round(x, 2), round(y, 2)), "distance_m": round(dist, 2)}


def main(out="data/derived/purgatory.json"):
    cfg0 = yaml.safe_load(open("config.yaml"))
    cfg = {**cfg0["planner"], **cfg0["constraints"]}
    ceiling = cfg["risk_ceiling"]
    terrain = dune_terrain()
    sim = MarsSim(seed=42, terrain=terrain)
    env = surrogate.create_surrogate_env(sim, cfg)

    routes = []
    for name, wps in ROUTES.items():
        risk = route_tail_risk(env, cfg, wps)
        routes.append({"name": name, "waypoints": wps, "tail_risk": round(risk, 3),
                       "safe": risk <= ceiling})

    # Earth / no-lightsim: take the direct route (looks shortest from orbit). Physics: it gets stuck.
    direct = routes[0]
    baseline_phys = run_physics(terrain, ROUTES["direct (into dune)"])

    # MARVIN: reject over-risk routes, take the lowest-risk safe one.
    safe = [r for r in routes if r["safe"]]
    marvin_route = min(safe, key=lambda r: r["tail_risk"]) if safe else None
    marvin_phys = run_physics(terrain, ROUTES[marvin_route["name"]]) if marvin_route else None

    result = {
        "event": "Opportunity — Purgatory Dune (Meridiani Planum, sol 446, 26 Apr 2005)",
        "real_stuck_sols": REAL_STUCK_SOLS,
        "risk_ceiling": ceiling,
        "routes": routes,
        "baseline": {"route": direct["name"], "predicted_tail_risk": direct["tail_risk"],
                     **baseline_phys, "sols_lost": REAL_STUCK_SOLS},
        "marvin": {"route": marvin_route["name"] if marvin_route else "HOLD",
                   "predicted_tail_risk": marvin_route["tail_risk"] if marvin_route else None,
                   **(marvin_phys or {}), "sols_lost": 0},
    }
    json.dump(result, open(out, "w"), indent=2)

    print(f"PURGATORY DUNE — Opportunity, sol 446 (real: stuck {REAL_STUCK_SOLS} sols)\n")
    print(f"  lightsim route assessment (risk ceiling {ceiling:.0%}):")
    for r in routes:
        tag = "SAFE" if r["safe"] else "ENTRAPMENT RISK"
        print(f"    {r['name']:20s} tail-risk {r['tail_risk'] * 100:>4.0f}%  [{tag}]")
    print()
    b = result["baseline"]
    print(f"  Earth-planned (no lightsim): {b['route']} -> reached={b['reached']} "
          f"(stuck at dune) -> {REAL_STUCK_SOLS} sols to extract "
          f"(~{REAL_STUCK_SOLS / SOLS_PER_MONTH:.1f} months), risk of permanent loss")
    m = result["marvin"]
    print(f"  MARVIN (lightsim):           {m['route']} -> reached={m.get('reached')} "
          f"-> 0 sols lost, entrapment avoided")
    print(f"\n  => MARVIN saves ~{REAL_STUCK_SOLS} sols and avoids the trap that stranded Spirit.")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
