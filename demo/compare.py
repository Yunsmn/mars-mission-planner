"""Why onboard autonomy matters — a three-way mission comparison.

Runs the same sampling campaign three ways on the identical Mars terrain and reports the
numbers that make the case:

  1. Adaptive onboard (MARVIN / Granite)  — perceives, decides, and acts between comms
     windows. Uses onboard perception to place the instrument, so it actually collects.
  2. Deterministic fixed plan (orbital only) — a plan computed once from orbital imagery and
     executed without adaptation. Orbital localization error is ~1 m, larger than the
     instrument's reach, so a fixed plan drives to the wrong spot and MISSES the samples.
  3. Earth-in-the-loop replanning — the same decisions, but each one waits a full ground
     command cycle (~1 sol). A real science campaign of many decisions becomes MONTHS.

Deterministic and fast, no LLM required — this is analysis, not a live model run.

Run:  .venv/bin/python -m demo.compare [output.png]
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rover import capabilities as cap
from world.sim import MarsSim

# Real Mars-mission cadence: rovers receive roughly one command cycle per sol.
GROUND_CYCLE_SOLS = 1.0
SOLS_PER_MONTH = 30.0
# Orbital-image localization error (m). HiRISE-scale ~0.25-1 m — larger than the
# instrument reach (0.55 m), so an orbital-only plan cannot reliably place the instrument.
ORBITAL_LOCALIZATION_ERROR_M = 0.85
CAMPAIGN_DECISIONS = 120   # a realistic multi-target science campaign, for extrapolation


def _fresh_sim(seed: int = 42) -> MarsSim:
    sim = MarsSim(seed=seed)
    cap.bind(sim, seed=seed + 1)
    return sim


def run_adaptive(seed: int = 42) -> dict:
    """Onboard: use perception to drive to each target and sample it (nearest-first)."""
    sim = _fresh_sim(seed)
    decisions = 0
    for _ in range(20):
        uncollected = [t for t in sim.targets if not t.collected]
        if sum(t.collected for t in sim.targets) >= 2 or not uncollected:
            break
        x, y, _ = sim.pose()
        nearest = min(uncollected, key=lambda t: (t.xy[0] - x) ** 2 + (t.xy[1] - y) ** 2)
        sim.drive_to(*nearest.xy)          # onboard perception places us on the target
        decisions += 1
        sim.sample(nearest.id)             # within reach -> collected
        decisions += 1
    samples = sum(t.collected for t in sim.targets)
    return {"name": "Adaptive onboard\n(MARVIN / Granite)", "samples": samples,
            "success": samples >= 2, "decisions": decisions,
            "battery": sim.battery_pct, "sols": 1.0}


def run_deterministic(seed: int = 42) -> dict:
    """Fixed plan from orbital data only: drive to orbital estimates (offset), no correction."""
    sim = _fresh_sim(seed)
    for t in sim.targets:                  # fixed a->b->c order, decided pre-mission
        ox, oy = t.xy[0] + ORBITAL_LOCALIZATION_ERROR_M, t.xy[1]
        sim.drive_to(ox, oy)               # arrives at the orbital estimate, not the target
        sim.sample(t.id)                   # true target is ~0.85 m away > 0.55 m reach -> miss
    samples = sum(t.collected for t in sim.targets)
    return {"name": "Deterministic fixed plan\n(orbital only)", "samples": samples,
            "success": samples >= 2, "decisions": len(sim.targets) * 2,
            "battery": sim.battery_pct, "sols": 1.0}


def earth_in_the_loop(adaptive: dict) -> dict:
    """Same decisions as adaptive, but each waits a ground command cycle (~1 sol)."""
    sols = adaptive["decisions"] * GROUND_CYCLE_SOLS
    return {"name": "Earth-in-the-loop\nreplanning", "samples": adaptive["samples"],
            "success": adaptive["success"], "decisions": adaptive["decisions"],
            "battery": adaptive["battery"], "sols": sols}


def render(adaptive: dict, deterministic: dict, earth: dict, out_path: str) -> str:
    import os

    arms = [adaptive, deterministic, earth]
    names = [a["name"] for a in arms]
    colors = ["#39c26b", "#e0553b", "#e0a03b"]

    fig, (ax_s, ax_t) = plt.subplots(1, 2, figsize=(13, 5.5))

    # samples cached
    ax_s.bar(names, [a["samples"] for a in arms], color=colors, edgecolor="black")
    ax_s.axhline(2, ls="--", color="gray", lw=1)
    ax_s.text(2.35, 2.05, "objective: 2", color="gray", fontsize=9)
    ax_s.set_ylabel("samples cached")
    ax_s.set_title("Did the mission succeed?")
    for i, a in enumerate(arms):
        ax_s.annotate("✓ success" if a["success"] else "✗ failed",
                      (i, a["samples"]), textcoords="offset points", xytext=(0, 6),
                      ha="center", fontsize=10,
                      color="#2a8f4f" if a["success"] else "#c0392b", weight="bold")

    # time to complete (sols, log scale)
    ax_t.bar(names, [max(a["sols"], 0.04) for a in arms], color=colors, edgecolor="black")
    ax_t.set_yscale("log")
    ax_t.set_ylabel("time to complete (sols, log)")
    ax_t.set_title("How long did it take?")
    for i, a in enumerate(arms):
        ax_t.annotate(f"{a['sols']:.0f} sol" + ("s" if a["sols"] != 1 else ""),
                      (i, max(a["sols"], 0.04)), textcoords="offset points",
                      xytext=(0, 6), ha="center", fontsize=10)

    fig.suptitle("Onboard autonomy vs. fixed plans vs. Earth-in-the-loop", fontsize=14, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "demo/out/comparison.png"
    adaptive = run_adaptive()
    deterministic = run_deterministic()
    earth = earth_in_the_loop(adaptive)

    print("\n=== Mission comparison (same terrain, same targets) ===")
    for a in (adaptive, deterministic, earth):
        tag = "SUCCESS" if a["success"] else "FAILED"
        print(f"  {a['name'].replace(chr(10),' '):38s}  samples={a['samples']}  "
              f"{tag:7s}  time={a['sols']:.0f} sol(s)  decisions={a['decisions']}")

    # headline extrapolation
    campaign_sols = CAMPAIGN_DECISIONS * GROUND_CYCLE_SOLS
    print(f"\n  Headline: a {CAMPAIGN_DECISIONS}-decision science campaign would take "
          f"~{campaign_sols:.0f} sols (~{campaign_sols / SOLS_PER_MONTH:.1f} months) of "
          f"Earth-in-the-loop planning — MARVIN runs it between comms windows.")
    print(f"  Fixed orbital-only plans miss the samples entirely "
          f"(localization error {ORBITAL_LOCALIZATION_ERROR_M} m > {0.55} m instrument reach).")

    path = render(adaptive, deterministic, earth, out)
    print(f"\n  wrote {path}")


if __name__ == "__main__":
    main()
