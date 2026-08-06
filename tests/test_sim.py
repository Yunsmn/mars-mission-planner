"""Behavior tests for the working simulation (Claude-built world layer)."""
import math

import numpy as np

from rover import capabilities as cap
from world import sensors
from world.sim import MarsSim


def test_terrain_is_drivable():
    sim = MarsSim(seed=42)
    assert sim.slope.max() < 15.0            # gentle enough for a basic rover
    assert np.all(np.isfinite(sim.data.qpos))


def test_drive_reaches_target_and_uses_battery():
    sim = MarsSim(seed=42)
    cap.bind(sim, seed=1)
    tx, ty = sim.targets[0].xy
    b0 = cap.battery()
    res = cap.drive_to(tx, ty)
    x, y, _ = sim.pose()
    assert res["success"]
    assert math.hypot(tx - x, ty - y) < 0.3
    assert cap.battery() < b0                # driving drained battery
    assert np.all(np.isfinite(sim.data.qpos))


def test_sample_pickup_marks_collected():
    sim = MarsSim(seed=42)
    cap.bind(sim)
    t = sim.targets[0]
    cap.drive_to(*t.xy)
    assert cap.sample(t.id)["success"]
    assert t.collected


def test_camera_intrinsics_consistent():
    sim = MarsSim(seed=42)
    K = sensors.camera_intrinsics(sim.model)
    assert K["cx"] == K["width"] / 2
    assert K["cy"] == K["height"] / 2
    assert K["fx"] > 0 and K["fy"] > 0
