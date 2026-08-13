"""Top-down mission visualization for the demo/video.

Renders the terrain, the rover's driven path, and the sample targets (collected vs pending),
plus a battery/science panel. Runs a fast scripted-but-real mission (no LLM) so it always
produces an artifact; the intelligent planner (Bob's layer) can feed its own trajectory here.

Headless (Agg backend) — no display or GL needed, so it's reliable in any environment.

Run:  .venv/bin/python -m demo.visualize [output.png]
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rover import capabilities as cap
from world.sim import TERRAIN_RADIUS, MarsSim


def run_and_render(out_path: str = "demo/out/mission_map.png", seed: int = 42) -> str:
    """Drive a scripted real mission, then render terrain + path + targets to `out_path`."""
    import os

    sim = MarsSim(seed=seed)
    cap.bind(sim, seed=seed + 1)

    battery_events = [("start", cap.battery())]
    for t in sim.targets:
        cap.drive_to(*t.xy)
        battery_events.append((f"drove→{t.id}", cap.battery()))
        cap.sample(t.id)
        battery_events.append((f"sampled {t.id}", cap.battery()))

    path = np.array(sim.trajectory) if sim.trajectory else np.empty((0, 2))

    fig, (ax_map, ax_bat) = plt.subplots(
        1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [2, 1]}
    )

    # --- terrain map -------------------------------------------------------------
    extent = [-TERRAIN_RADIUS, TERRAIN_RADIUS, -TERRAIN_RADIUS, TERRAIN_RADIUS]
    im = ax_map.imshow(sim.terrain, extent=extent, origin="lower",
                       cmap="copper", alpha=0.95)
    fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04, label="elevation (m)")

    if len(path):
        ax_map.plot(path[:, 0], path[:, 1], "-", color="#00e5ff", lw=2.0,
                    label="rover path", zorder=3)
        ax_map.plot(path[0, 0], path[0, 1], "o", color="white",
                    markeredgecolor="black", ms=10, label="start", zorder=4)

    for t in sim.targets:
        collected = t.collected
        ax_map.scatter(*t.xy, s=180, marker="*" if collected else "s",
                       color="#39ff14" if collected else "#ffae42",
                       edgecolor="black", linewidth=1.2, zorder=5)
        ax_map.annotate(t.id, t.xy, textcoords="offset points", xytext=(8, 8),
                        fontsize=9, color="white",
                        path_effects=None)

    ax_map.set_title("MARVIN — mission on Mars terrain\n"
                     f"max slope {sim.slope.max():.1f}°  ·  "
                     f"{sum(t.collected for t in sim.targets)}/{len(sim.targets)} samples cached")
    ax_map.set_xlabel("x (m)"); ax_map.set_ylabel("y (m)")
    ax_map.set_xlim(-TERRAIN_RADIUS, TERRAIN_RADIUS)
    ax_map.set_ylim(-TERRAIN_RADIUS, TERRAIN_RADIUS)
    ax_map.legend(loc="upper right", framealpha=0.85)

    # --- battery timeline --------------------------------------------------------
    labels = [e[0] for e in battery_events]
    vals = [e[1] for e in battery_events]
    ax_bat.plot(range(len(vals)), vals, "-o", color="#ff5252")
    ax_bat.set_xticks(range(len(labels)))
    ax_bat.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax_bat.set_ylabel("battery (%)")
    ax_bat.set_ylim(min(vals) - 2, 101)
    ax_bat.set_title("Battery vs. mission progress")
    ax_bat.grid(True, alpha=0.3)
    for i, v in enumerate(vals):
        ax_bat.annotate(f"{v:.1f}", (i, v), textcoords="offset points",
                        xytext=(0, 6), fontsize=8, ha="center")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "demo/out/mission_map.png"
    path = run_and_render(out)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
