"""Ablation — does the architecture earn its parts? Three policies, same targets, real terrain:

  1. No planner        — go to the NEAREST target (no reasoning, no verification).
  2. Planner, no sim   — go to the HIGHEST-science target (reasons about value, but never
                         checks whether the approach is safe).
  3. Planner + fast sim — pick the highest-science target whose surrogate-predicted tail-risk
                         is within the safety ceiling (propose AND verify).

The fast surrogate is the measurement instrument: for each candidate it rolls the drive out 20x
under uncertainty and reports worst-case (tail) hazard + success probability. The point: without
the verify step, a planner chasing science value walks into a high-risk approach; with it, the
gate rejects that and takes the best SAFE target — same objective, bounded risk.

Run:  .venv/bin/python -m demo.ablation
"""
from __future__ import annotations

import json
import math

import numpy as np
import yaml

from common.types import Action, ActionKind, MissionState, Pose
from planner import gating, surrogate
from world.sim import TERRAIN_RADIUS, MarsSim


def _risk_success(env, state, xy, cfg, rng):
    seq = (Action(kind=ActionKind.DRIVE, params={"xy": xy}),)
    b = surrogate.rollout_batch(seq, state, env, cfg.get("n_rollouts", 20), rng)
    risk = float(gating.tail_worst(b.hazard, cfg.get("cvar_quantile", 0.9)))
    return risk, float(np.mean(b.success))


def build_scene(sim, env, state, cfg, rng):
    """Pick real positions on the terrain: a high-tail-risk approach (assigned high science) and
    two low-risk basin targets — so value and safety actually pull in different directions."""
    cands = []
    for ang in range(0, 360, 15):
        for rad in (2.0, 2.8, 3.4):
            x = round(rad * math.cos(math.radians(ang)), 2)
            y = round(rad * math.sin(math.radians(ang)), 2)
            if abs(x) <= TERRAIN_RADIUS - 0.6 and abs(y) <= TERRAIN_RADIUS - 0.6:
                r, p = _risk_success(env, state, (x, y), cfg, rng)
                cands.append({"xy": (x, y), "risk": r, "p": p})
    cands.sort(key=lambda c: c["risk"])
    safe1, safe2 = cands[0], cands[len(cands) // 6]      # low-risk basin targets
    risky = cands[-1]                                    # steepest approach
    return [
        {"id": "basin_A", "science": 0.50, **safe1},
        {"id": "basin_B", "science": 0.60, **safe2},
        {"id": "rim_C",   "science": 0.90, **risky},     # tempting but risky
    ]


def main(out="data/derived/ablation.json"):
    cfg0 = yaml.safe_load(open("config.yaml"))
    cfg = {**cfg0["planner"], **cfg0["constraints"]}
    ceiling = cfg["risk_ceiling"]
    sim = MarsSim(seed=42)
    env = surrogate.create_surrogate_env(sim, cfg)
    rng = np.random.default_rng(0)
    state = MissionState(pose=Pose((0.0, 0.0), 0.0), battery_pct=100.0, sol_time=0.0,
                         localization_sigma=1.0, collected=(), remaining=())

    tg = build_scene(sim, env, state, cfg, rng)
    by_id = {t["id"]: t for t in tg}

    def dist(t):
        return math.hypot(t["xy"][0], t["xy"][1])

    nearest = min(tg, key=dist)
    highest = max(tg, key=lambda t: t["science"])
    safe = [t for t in tg if t["risk"] <= ceiling]
    best_safe = max(safe, key=lambda t: t["science"]) if safe else None

    configs = [
        ("No planner", "nearest target", nearest),
        ("Planner, no fast-sim", "highest science, unchecked", highest),
        ("Planner + fast-sim", "highest science within risk ceiling", best_safe),
    ]
    rows = []
    for name, rule, chosen in configs:
        if chosen is None:
            rows.append({"config": name, "rule": rule, "chosen": "HOLD", "risk": None,
                         "p_success": None, "science": 0.0, "safe": True})
            continue
        rows.append({
            "config": name, "rule": rule, "chosen": chosen["id"],
            "risk": round(chosen["risk"], 3), "p_success": round(chosen["p"], 3),
            "science": chosen["science"], "safe": chosen["risk"] <= ceiling,
        })

    result = {"risk_ceiling": ceiling, "targets": [
        {"id": t["id"], "risk": round(t["risk"], 3), "p_success": round(t["p"], 3),
         "science": t["science"]} for t in tg], "rows": rows}
    json.dump(result, open(out, "w"), indent=2)

    print(f"risk ceiling = {ceiling:.0%}")
    print(f"targets: " + " | ".join(
        f"{t['id']}(sci {t['science']}, risk {t['risk']:.0%})" for t in tg))
    print()
    for r in rows:
        tag = "SAFE" if r["safe"] else "OVER-RISK"
        rk = f"{r['risk']:.0%}" if r["risk"] is not None else "—"
        print(f"  {r['config']:22s} -> {r['chosen']:8s}  tail-risk {rk:>5}  "
              f"science {r['science']:.2f}  [{tag}]")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
