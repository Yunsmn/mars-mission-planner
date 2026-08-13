"""Animated mission playback — watch the rover drive the terrain and cache samples.

Produces:
  docs/figures/mission.gif         — motion (open it to watch)
  docs/figures/mission_frames.png  — a 4-frame montage (left -> right) for a quick look

Headless (Agg). Uses a scripted-real mission (real physics, no LLM) so it always renders.

Run:  .venv/bin/python -m demo.animate
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from rover import capabilities as cap
from world.sim import TERRAIN_RADIUS, MarsSim

_EXT = [-TERRAIN_RADIUS, TERRAIN_RADIUS, -TERRAIN_RADIUS, TERRAIN_RADIUS]


def run_mission(seed: int = 42):
    sim = MarsSim(seed=seed)
    cap.bind(sim, seed + 1)
    segments = []                      # (start_idx, end_idx, target_id) per leg
    for t in sim.targets:
        start = len(sim.trajectory)
        sim.drive_to(*t.xy)
        segments.append((start, len(sim.trajectory), t.id))
        sim.sample(t.id)
    return sim, np.array(sim.trajectory), segments


def _collected_at(idx: int, segments) -> set:
    return {tid for (_s, e, tid) in segments if e <= idx}


def _terrain(ax, sim):
    ax.imshow(sim.terrain, extent=_EXT, origin="lower", cmap="copper", alpha=0.95)
    ax.set_xlim(-TERRAIN_RADIUS, TERRAIN_RADIUS)
    ax.set_ylim(-TERRAIN_RADIUS, TERRAIN_RADIUS)


def _targets(ax, sim, collected):
    for t in sim.targets:
        c = t.id in collected
        ax.scatter(*t.xy, s=150, marker="*" if c else "s",
                   color="#39ff14" if c else "#ffae42", edgecolor="black", zorder=5)
        ax.annotate(t.id, t.xy, textcoords="offset points", xytext=(7, 7),
                    fontsize=8, color="white")


def _frame(ax, sim, traj, segments, j):
    ax.clear()
    _terrain(ax, sim)
    col = _collected_at(j, segments)
    ax.plot(traj[:j + 1, 0], traj[:j + 1, 1], "-", color="#00e5ff", lw=2, zorder=3)
    ax.plot(traj[j, 0], traj[j, 1], "o", color="white", mec="black", ms=11, zorder=6)
    _targets(ax, sim, col)
    return col


def make_gif(sim, traj, segments, out="docs/figures/mission.gif", n_frames=80):
    idxs = np.linspace(0, len(traj) - 1, min(n_frames, len(traj))).astype(int)
    fig, ax = plt.subplots(figsize=(7, 7))

    def draw(fi):
        col = _frame(ax, sim, traj, segments, idxs[fi])
        ax.set_title(f"MARVIN mission — samples cached {len(col)}/{len(sim.targets)}")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

    anim = FuncAnimation(fig, draw, frames=len(idxs), interval=80)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=12))
    plt.close(fig)
    return out


def make_montage(sim, traj, segments, out="docs/figures/mission_frames.png"):
    picks = [0, len(traj) // 3, 2 * len(traj) // 3, len(traj) - 1]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6))
    for ax, j in zip(axes, picks):
        col = _frame(ax, sim, traj, segments, j)
        ax.set_title(f"cached {len(col)}/{len(sim.targets)}")
        ax.set_xlabel("x (m)")
    fig.suptitle("MARVIN mission playback  (left → right)", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    sim, traj, segments = run_mission()
    g = make_gif(sim, traj, segments)
    m = make_montage(sim, traj, segments)
    print(f"wrote {g} and {m}  (trajectory points: {len(traj)})")


if __name__ == "__main__":
    main()
