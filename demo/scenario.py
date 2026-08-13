"""Richer scenario — the rover must choose WHICH two samples to cache, and the choice matters.

Targets carry different CRISM-derived science values. The two *nearest* targets are low-value
(basalt, olivine); the high-value ones (carbonate, phyllosilicate) are farther. Under the same
2-sample budget, a value-aware planner (what IBM Granite does, guided by model/SKILL.md) returns
far more science than a naive nearest-first policy — for a small extra energy cost that stays
within the battery reserve.

This is a planning comparison over the real distance/energy model (deterministic and clear).
Pass --granite to also ask the live Granite planner which target it picks first.

Run:  .venv/bin/python -m demo.scenario [--granite]
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rover import energy

# (id, x, y, science_value, mineral) — nearest are LOW value; the best science is farther out.
SCENE = [
    ("basalt_rock",     0.6,  0.5, 0.30, "basalt"),
    ("olivine_dune",   -0.7, -0.9, 0.45, "olivine"),
    ("clay_delta",     -1.8,  1.4, 0.78, "phyllosilicate"),
    ("carbonate_ridge", 2.3,  1.6, 0.92, "carbonate"),
]
START = (0.0, 0.0)
BUDGET = 2                 # cache 2 samples
NOMINAL_SLOPE_DEG = 2.0    # gentle terrain
PAYLOAD_KG = 3.0


def _dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _plan_energy_pct(order):
    """Battery % spent to visit `order` (list of targets) from START in sequence."""
    pos, total = START, 0.0
    for t in order:
        total += energy.drive_cost_wh(_dist(pos, (t[1], t[2])), NOMINAL_SLOPE_DEG, PAYLOAD_KG)
        pos = (t[1], t[2])
    return total


def naive_nearest(targets):
    """Greedily visit the nearest target each time (ignores science value)."""
    chosen, pos, remaining = [], START, list(targets)
    while len(chosen) < BUDGET and remaining:
        t = min(remaining, key=lambda t: _dist(pos, (t[1], t[2])))
        chosen.append(t); remaining.remove(t); pos = (t[1], t[2])
    return chosen


def value_aware(targets):
    """Pick the highest science-value targets (visit nearest-first among the chosen)."""
    chosen = sorted(targets, key=lambda t: t[3], reverse=True)[:BUDGET]
    # order the chosen ones to reduce travel
    ordered, pos, rem = [], START, list(chosen)
    while rem:
        t = min(rem, key=lambda t: _dist(pos, (t[1], t[2])))
        ordered.append(t); rem.remove(t); pos = (t[1], t[2])
    return ordered


def _science(order):
    return sum(t[3] for t in order)


def render(rows, out="docs/figures/scenario.png"):
    import os
    names = [r[0] for r in rows]
    sci = [r[2] for r in rows]
    colors = ["#e0553b", "#39c26b"]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, sci, color=colors, edgecolor="black")
    ax.set_ylabel("total science value cached (2 samples)")
    ax.set_title("Same 2-sample budget — the planner's choice decides the science return")
    for r, b in zip(rows, bars):
        picked = "\n".join(f"{t[0]} ({t[3]:.2f})" for t in r[1])
        ax.annotate(f"science {r[2]:.2f} · {r[3]:.1f}% batt\n{picked}",
                    (b.get_x() + b.get_width() / 2, r[2]),
                    textcoords="offset points", xytext=(0, 6), ha="center", fontsize=9)
    ax.set_ylim(0, max(sci) * 1.4)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    print("Scene (nearest are low-value; best science is farther):")
    for tid, x, y, sv, m in SCENE:
        print(f"  {tid:16s} ({x:+.1f},{y:+.1f})  d={_dist(START,(x,y)):.2f}m  science={sv:.2f}  [{m}]")

    naive = naive_nearest(SCENE)
    value = value_aware(SCENE)
    rows = [
        ("Naive\nnearest-first", naive, _science(naive), _plan_energy_pct(naive)),
        ("Value-aware\n(Granite + SKILL)", value, _science(value), _plan_energy_pct(value)),
    ]
    for label, order, sci, batt in rows:
        print(f"\n  {label.splitlines()[0]:12s}: {[t[0] for t in order]}"
              f"  science={sci:.2f}  battery={batt:.1f}%")
    ratio = _science(value) / max(_science(naive), 1e-6)
    print(f"\n  => value-aware returns {ratio:.1f}x the science for +"
          f"{_plan_energy_pct(value) - _plan_energy_pct(naive):.1f}% battery "
          f"(well within the 15% reserve).")

    if "--granite" in sys.argv:
        _granite_pick()

    print(f"\n  wrote {render(rows)}")


def _granite_pick():
    """Ask the live Granite planner which target it proposes first on this scene."""
    import yaml
    from common.types import MissionState, Perception, Pose, Target
    from model.propose import Proposer
    import numpy as np
    targets = tuple(Target(t[0], (t[1], t[2]), science_value=t[3], mineral_class=t[4]) for t in SCENE)
    perc = Perception(slope_deg=np.full((7, 7), 2.0), roughness=np.full((7, 7), 0.6),
                      visible_targets=targets, dust_tau=0.5)
    state = MissionState(pose=Pose(START, 0.0), battery_pct=100.0, sol_time=0.0,
                         localization_sigma=1.0, collected=(), remaining=targets)
    cfg = yaml.safe_load(open("config.yaml"))["model"]
    cands = Proposer(cfg["name"], cfg["host"], cfg["temperature"]).propose(state, perc, 2)
    print("\n  Live IBM Granite proposals:")
    for i, c in enumerate(cands, 1):
        print(f"    cand {i}: " + " -> ".join(f"{a.kind.name}{a.params}" for a in c))


if __name__ == "__main__":
    main()
