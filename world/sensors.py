"""Onboard sensing over the ground-truth world: camera intrinsics + noisy pose + perception.

Perception is what the planner SEES — deliberately noisy, never the true state. That noise is
what makes replan-on-new-perception and Value-of-Information meaningful. See docs/SENSORS.md.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from common.types import Perception, Pose, Target

# Default onboard camera resolution (intrinsics scale with this).
CAM_WIDTH = 640
CAM_HEIGHT = 480


def camera_intrinsics(model, cam_name: str = "rover_cam",
                      width: int = CAM_WIDTH, height: int = CAM_HEIGHT) -> dict:
    """Pinhole intrinsics for a MuJoCo camera, derived from its vertical FOV.

    fy = (H/2) / tan(fovy/2);  fx = fy (square pixels);  cx = W/2, cy = H/2.
    Returns fx, fy, cx, cy, the 3x3 K matrix, plus fovy and resolution.
    """
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    fovy_deg = float(model.cam_fovy[cam_id])
    fy = (height / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
    fx = fy
    cx, cy = width / 2.0, height / 2.0
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "K": K,
            "fovy_deg": fovy_deg, "width": width, "height": height}


def read_pose(sim, rng: np.random.Generator,
              pos_sigma: float = 0.05, yaw_sigma_deg: float = 2.0) -> Pose:
    """Estimated pose from odometry/IMU with drift + noise (NOT the true pose)."""
    x, y, yaw = sim.pose()
    return Pose(
        xy=(x + rng.normal(0, pos_sigma), y + rng.normal(0, pos_sigma)),
        heading_rad=yaw + rng.normal(0, math.radians(yaw_sigma_deg)),
    )


def observe(sim, rng: np.random.Generator, view_range: float = 3.0) -> Perception:
    """Build a Perception: nearby uncollected targets (with detection noise) + local terrain."""
    x, y, _ = sim.pose()
    visible = []
    
    # Try to load science value map if available
    science_map = None
    try:
        from data.science_value import load_science_map, get_science_value, classify_mineral
        import os
        science_path = "data/derived/jezero_science_value.npy"
        if os.path.exists(science_path):
            science_map = load_science_map(science_path)
    except Exception:
        pass  # Fall back to default values
    
    for t in sim.targets:
        if t.collected:
            continue
        if math.hypot(t.xy[0] - x, t.xy[1] - y) <= view_range:
            # detection noise on the reported location
            vx = t.xy[0] + rng.normal(0, 0.08)
            vy = t.xy[1] + rng.normal(0, 0.08)
            
            # Get science value from CRISM-derived map if available
            if science_map is not None:
                from data.science_value import get_science_value, classify_mineral
                science_value = get_science_value(t.xy, science_map)
                mineral_class = classify_mineral(science_value)
            else:
                science_value = 0.5
                mineral_class = "unknown"
            
            visible.append(Target(id=t.id, xy=(vx, vy),
                                  science_value=science_value, 
                                  mineral_class=mineral_class))
    
    local = _local_window(sim.slope, sim, x, y)
    return Perception(slope_deg=local, roughness=local * 0.3,
                      visible_targets=tuple(visible), dust_tau=sim.dust_tau)


def _local_window(grid: np.ndarray, sim, x: float, y: float, half: int = 3) -> np.ndarray:
    n = grid.shape[0]
    from world.sim import TERRAIN_RADIUS
    j = int(np.clip((x + TERRAIN_RADIUS) / (2 * TERRAIN_RADIUS) * (n - 1), 0, n - 1))
    i = int(np.clip((y + TERRAIN_RADIUS) / (2 * TERRAIN_RADIUS) * (n - 1), 0, n - 1))
    i0, i1 = max(0, i - half), min(n, i + half + 1)
    j0, j1 = max(0, j - half), min(n, j + half + 1)
    return grid[i0:i1, j0:j1].copy()
