"""Run a real IBM Granite mission on the real Jezero terrain and record it for the showcase.

Everything the showcase reports is measured from this run — decisions, traverse distance,
battery used, sim time. Deterministic-ish (seed fixed); recorded once to JSON and committed so
the page reflects a genuine mission, not hand-picked numbers.

Run:  .venv/bin/python -m web.record_mission
"""
from __future__ import annotations

import json
import math

import yaml

from planner.loop import run_mission
from model.propose import Proposer
from world.sim import TERRAIN_RADIUS, MarsSim


def _fmt_action(a) -> dict:
    p = a.params or {}
    if "target" in p:
        to = p["target"]
    elif "xy" in p:
        to = f"({p['xy'][0]:.1f}, {p['xy'][1]:.1f})"
    else:
        to = ""
    return {"act": a.kind.name, "to": to}


def record(out: str = "data/derived/mission_record.json") -> dict:
    cfg = yaml.safe_load(open("config.yaml"))
    sim = MarsSim(seed=42)                       # real Jezero MOLA terrain

    collect_at: dict[str, int] = {}
    orig_sample = sim.sample
    def sample_recording(tid, **kw):
        collect_at[tid] = len(sim.trajectory)
        return orig_sample(tid, **kw)
    sim.sample = sample_recording               # capture when each target is cached

    model = Proposer(cfg["model"]["name"], cfg["model"]["host"], cfg["model"]["temperature"])
    plan_cfg = {**cfg["planner"], **cfg["constraints"], "energy_penalty_factor": 0.01}
    log = run_mission(cfg["mission"]["objective"], sim, model, plan_cfg, max_steps=30)

    traj = [[round(float(x), 3), round(float(y), 3)] for (x, y) in sim.trajectory]
    dist = sum(math.dist(traj[i - 1], traj[i]) for i in range(1, len(traj)))

    data = {
        "grid": int(sim.terrain.shape[0]),
        "extent": float(TERRAIN_RADIUS),
        "terrain": [[round(float(v), 4) for v in row] for row in sim.terrain],
        "trajectory": traj,
        "targets": [{"id": t.id, "x": round(t.xy[0], 3), "y": round(t.xy[1], 3),
                     "collectAt": collect_at.get(t.id, len(traj))} for t in sim.targets],
        "decisions": [_fmt_action(d.action) for d in log.decisions],
        "stats": {
            "decisions": len(log.decisions),
            "samples": int(log.samples_collected),
            "distance_m": round(dist, 2),
            "battery_used_pct": round(100.0 - sim.battery_pct, 1),
            "sim_seconds": round(sim.sol_time, 1),
        },
    }
    json.dump(data, open(out, "w"))
    print("recorded:", data["stats"])
    print("decisions:", " -> ".join(d["act"] for d in data["decisions"]))
    return data


if __name__ == "__main__":
    record()
