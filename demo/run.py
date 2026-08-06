"""Basic simulation demo — the working world Claude built, before Bob's planning layer.

Runs headless (no offscreen render, which is unreliable in this environment): it builds the
world, prints the camera intrinsics, drives the rover to each sample via the capability API,
grabs it, and prints telemetry. Deterministic.

Run:  .venv/bin/python -m demo.run
Later, Bob's planner replaces the scripted loop with propose-and-verify decisions.
"""
from __future__ import annotations

from rover import capabilities as cap
from world import sensors
from world.sim import MarsSim


def main() -> None:
    sim = MarsSim(seed=42)
    cap.bind(sim, seed=1)

    K = sensors.camera_intrinsics(sim.model)
    print("=== onboard camera intrinsics ===")
    print(f"  fovy={K['fovy_deg']:.0f}deg  resolution={K['width']}x{K['height']}")
    print(f"  fx={K['fx']:.1f}  fy={K['fy']:.1f}  cx={K['cx']:.1f}  cy={K['cy']:.1f}")

    print(f"\n=== world ===  max slope {sim.slope.max():.1f}deg  targets={len(sim.targets)}")
    print(f"start battery {cap.battery():.1f}%  pose {cap.get_pose()}")

    for t in sim.targets:
        p = cap.scan()
        print(f"\nperception: {len(p.visible_targets)} target(s) in view, dust_tau={p.dust_tau}")
        print(f"-> drive to {t.id} at {t.xy}")
        res = cap.drive_to(*t.xy)
        print(f"   {res}")
        print(f"   grab: {cap.sample(t.id)}")

    print(f"\ncollected: {[t.id for t in sim.targets if t.collected]}")
    print(f"battery left: {cap.battery():.1f}%")


if __name__ == "__main__":
    main()
