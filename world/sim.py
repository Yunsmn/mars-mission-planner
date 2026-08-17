"""MarsSim — the high-fidelity ground-truth world (see docs/DESIGN.md §5.6).

A skid-steer 4-wheel rover on a MuJoCo heightfield under Mars gravity, with wheel-terrain
contact (slip emerges from physics), a forward camera, and pickup-able sample targets.
This is the REAL world the planner acts on; the fast surrogate (planner/surrogate.py) is a
separate cheap approximation of it.

Built by Claude as the working simulation; the LLM planning layer on top is authored by Bob.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from world import dem as demlib

MARS_GRAVITY = 3.72          # m/s^2
TERRAIN_RADIUS = 5.0         # world spans [-5, 5] m in x and y
WHEEL_RADIUS = 0.06

# Optional photoreal skin: NASA's public-domain Perseverance model as a *visual-only* mesh on the
# chassis (no collisions; physics stays the fast skid-steer box). Used for rendering/video only.
import os as _os
ROVER_MESH_PATH = _os.path.join(_os.path.dirname(__file__), "assets", "perseverance.obj")
ROVER_MESH_SCALE = 0.257     # ~0.8 m long, matched to the chassis footprint
ROVER_MESH_YAW = 90          # mesh length axis is Y; +90deg aligns its front with chassis +X
ROVER_MESH_Z = -0.10         # drop so the model's wheels meet the terrain contact plane


@dataclass
class SampleTarget:
    id: str
    xy: tuple[float, float]
    collected: bool = False
    science_value: float = 0.5      # data-driven (CRISM); higher = better science
    mineral_class: str = "unknown"


def _rover_xml(hf: dict, targets: list[SampleTarget], render_mesh: bool = False) -> str:
    """Build the MJCF for terrain + skid-steer rover + camera + sample targets.

    render_mesh=True drapes the NASA Perseverance visual mesh over the chassis and hides the
    physics box + wheels (alpha 0). Physics is identical either way; the mesh is cosmetic.
    """
    start_z = hf["elevation_m"] + 0.35
    use_mesh = render_mesh and _os.path.exists(ROVER_MESH_PATH)
    box_alpha = 0.0 if use_mesh else 1.0
    wheel_alpha = 0.0 if use_mesh else 1.0
    compiler = f'<compiler meshdir="{_os.path.dirname(ROVER_MESH_PATH)}"/>' if use_mesh else ""
    visual = '<visual><global offwidth="1280" offheight="960"/></visual>' if use_mesh else ""
    mesh_asset = (f'<mesh name="rover_mesh" file="{_os.path.basename(ROVER_MESH_PATH)}" '
                  f'scale="{ROVER_MESH_SCALE} {ROVER_MESH_SCALE} {ROVER_MESH_SCALE}"/>') if use_mesh else ""
    # mass="0": visual-only skin must add no mass/inertia (else default density pins the chassis).
    mesh_geom = (f'<geom type="mesh" mesh="rover_mesh" contype="0" conaffinity="0" group="1" '
                 f'mass="0" pos="0 0 {ROVER_MESH_Z}" euler="0 0 {ROVER_MESH_YAW}" '
                 f'rgba="0.82 0.80 0.78 1"/>') if use_mesh else ""
    target_bodies = "\n".join(
        f'''
        <body name="{t.id}" pos="{t.xy[0]:.3f} {t.xy[1]:.3f} {start_z + 0.5:.3f}">
          <freejoint/>
          <geom type="box" size="0.05 0.05 0.05" rgba="0.85 0.55 0.2 1" mass="0.2"/>
        </body>'''
        for t in targets
    )
    wheels = ""
    for name, sx, sy in [("fl", 0.16, 0.13), ("fr", 0.16, -0.13),
                         ("rl", -0.16, 0.13), ("rr", -0.16, -0.13)]:
        wheels += f'''
        <body name="wheel_{name}" pos="{sx} {sy} -0.04">
          <joint name="wheel_{name}" type="hinge" axis="0 1 0" armature="0.01" damping="0.05"/>
          <geom type="cylinder" fromto="0 -0.03 0 0 0.03 0" size="{WHEEL_RADIUS}"
                rgba="0.2 0.2 0.22 {wheel_alpha}" friction="1.5 0.02 0.001" mass="0.3"
                condim="4" margin="0.001"/>
        </body>'''
    actuators = "\n".join(
        f'<velocity name="drive_{n}" joint="wheel_{n}" kv="8" ctrllimited="true" ctrlrange="-30 30"/>'
        for n in ("fl", "fr", "rl", "rr")
    )
    return f'''<mujoco model="marsim">
  <option timestep="0.004" gravity="0 0 -{MARS_GRAVITY}" integrator="implicitfast" iterations="50"/>
  {compiler}
  {visual}
  <asset>
    <hfield name="terrain" nrow="{hf['nrow']}" ncol="{hf['ncol']}"
            size="{TERRAIN_RADIUS} {TERRAIN_RADIUS} {max(hf['elevation_m'], 0.05):.3f} 0.5"/>
    {mesh_asset}
  </asset>
  <worldbody>
    <light pos="0 0 6" dir="0 0 -1" diffuse="0.9 0.85 0.8"/>
    <geom name="terrain" type="hfield" hfield="terrain" pos="0 0 0"
          rgba="0.72 0.42 0.28 1" friction="1.2 0.02 0.001"/>
    <body name="chassis" pos="0 0 {start_z:.3f}">
      <freejoint/>
      <geom type="box" size="0.22 0.16 0.05" rgba="0.8 0.8 0.85 {box_alpha}" mass="3.0"/>
      {mesh_geom}
      <camera name="rover_cam" pos="0.24 0 0.10" xyaxes="1 0 0 0 -1 1" fovy="58"/>
      {wheels}
    </body>
    {target_bodies}
  </worldbody>
  <actuator>
    {actuators}
  </actuator>
</mujoco>'''


class MarsSim:
    """The ground-truth world + a basic rover controller (drive / sample / sense)."""

    def __init__(self, terrain: np.ndarray | None = None, seed: int = 42,
                 targets: list[tuple[str, float, float]] | None = None,
                 dem_path: str | None = None, render_mesh: bool = False) -> None:
        # Default to the real Jezero MOLA DEM if it's been fetched (data.fetch_jezero); else synthetic.
        import os as _os
        _default_dem = "data/derived/jezero_dem.npy"
        if dem_path is None and terrain is None and _os.path.exists(_default_dem):
            dem_path = _default_dem
        # Load real DEM if path provided, otherwise use synthetic or provided terrain
        if dem_path:
            try:
                self.terrain, metadata = demlib.load_dem(dem_path)
                # Terrain is rendered into the fixed-size sim patch, so slope uses sim scale.
                self.meters_per_cell = 2 * TERRAIN_RADIUS / (self.terrain.shape[0] - 1)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Failed to load DEM from {dem_path}: {e}, using synthetic")
                self.terrain = demlib.synthesize(seed=seed)
                self.meters_per_cell = 2 * TERRAIN_RADIUS / (self.terrain.shape[0] - 1)
        else:
            self.terrain = demlib.synthesize(seed=seed) if terrain is None else terrain
            self.meters_per_cell = 2 * TERRAIN_RADIUS / (self.terrain.shape[0] - 1)
        
        self.slope = demlib.slope_map(self.terrain, self.meters_per_cell)
        self.hf = demlib.dem_to_hfield(self.terrain)
        if targets is None:
            targets = [("sample_a", 1.6, 0.8), ("sample_b", -1.2, 1.5), ("sample_c", 0.5, -1.8)]
        # targets may be (id, x, y) or (id, x, y, science_value[, mineral_class])
        self.targets = [
            SampleTarget(t[0], (t[1], t[2]),
                         science_value=t[3] if len(t) > 3 else 0.5,
                         mineral_class=t[4] if len(t) > 4 else "unknown")
            for t in targets
        ]

        self.model = mujoco.MjModel.from_xml_string(_rover_xml(self.hf, self.targets, render_mesh))
        self.model.hfield_data[:] = self.hf["data"].ravel().astype(np.float32)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        self._chassis = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
        self._drives = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"drive_{n}")
                        for n in ("fl", "fr", "rl", "rr")]
        self.battery_pct = 100.0
        self.sol_time = 0.0
        self.dust_tau = 0.5
        self._settle(400)
        self.trajectory: list[tuple[float, float]] = []   # (x, y) path, for visualization

    def reset(self) -> None:
        """Restore the initial state on the *same* model (so a live viewer can keep its window)."""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        for t in self.targets:
            t.collected = False
        self.battery_pct = 100.0
        self.sol_time = 0.0
        self._settle(400)
        self.trajectory = []

    # ---- physics -----------------------------------------------------------------
    def _settle(self, n: int) -> None:
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    def step(self, vL: float, vR: float, n: int = 1) -> None:
        self.data.ctrl[self._drives[0]] = vL   # fl
        self.data.ctrl[self._drives[1]] = vR   # fr
        self.data.ctrl[self._drives[2]] = vL   # rl
        self.data.ctrl[self._drives[3]] = vR   # rr
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
        self.sol_time += n * self.model.opt.timestep
        if hasattr(self, "trajectory"):
            self.trajectory.append(self.pose()[:2])

    # ---- state -------------------------------------------------------------------
    def pose(self) -> tuple[float, float, float]:
        x, y = self.data.xpos[self._chassis][:2]
        w, xq, yq, zq = self.data.xquat[self._chassis]
        yaw = math.atan2(2 * (w * zq + xq * yq), 1 - 2 * (yq * yq + zq * zq))
        return float(x), float(y), float(yaw)

    def sensed_dem(self, grid: int = 48, noise_m: float = 0.02, seed: int = 1) -> np.ndarray:
        """The rover's PERCEIVED heightmap — the true surface sampled at sensor resolution with
        noise, NOT the ground-truth array the physics integrates. The planner (A*, lightsim) builds
        its terrain model from this, so it plans on what it *sees*, and the surrogate is honestly a
        model of the world rather than a copy of it."""
        from scipy.ndimage import map_coordinates
        n = self.terrain.shape[0]
        idx = np.linspace(0, n - 1, grid)
        ii, jj = np.meshgrid(idx, idx, indexing="ij")
        sensed = map_coordinates(self.terrain, [ii.ravel(), jj.ravel()], order=1).reshape(grid, grid)
        sensed = sensed + np.random.default_rng(seed).normal(0.0, noise_m, sensed.shape)
        return sensed.astype(np.float64)

    def slope_at(self, x: float, y: float) -> float:
        n = self.terrain.shape[0]
        j = int(np.clip((x + TERRAIN_RADIUS) / (2 * TERRAIN_RADIUS) * (n - 1), 0, n - 1))
        i = int(np.clip((y + TERRAIN_RADIUS) / (2 * TERRAIN_RADIUS) * (n - 1), 0, n - 1))
        return float(self.slope[i, j])

    # ---- capabilities ------------------------------------------------------------
    def drive_iter(self, tx: float, ty: float, tol: float = 0.18, max_steps: int = 6000,
                   yield_every: int = 4):
        """Skid-steer toward (tx, ty), yielding control every `yield_every` macro-steps so a caller
        can render a frame mid-drive. Handles the stop + battery cost at the end. `drive_to` wraps it."""
        from rover import energy
        x0, y0, _ = self.pose()
        steps = k = 0
        while steps < max_steps:
            x, y, yaw = self.pose()
            dx, dy = tx - x, ty - y
            dist = math.hypot(dx, dy)
            if dist < tol:
                break
            heading_err = math.atan2(math.sin(math.atan2(dy, dx) - yaw),
                                     math.cos(math.atan2(dy, dx) - yaw))
            # Turn in place when badly misaligned (avoids spin-stall on large heading changes),
            # then drive forward while correcting.
            base = 0.0 if abs(heading_err) > 0.6 else 14.0 * min(1.0, dist / 0.5)
            turn = 12.0 * heading_err
            vL = float(np.clip(base - turn, -30, 30))
            vR = float(np.clip(base + turn, -30, 30))
            self.step(vL, vR, n=10)
            steps += 10
            k += 1
            if k % yield_every == 0:
                yield
        self.step(0.0, 0.0, n=20)
        xf, yf, _ = self.pose()
        travelled = math.hypot(xf - x0, yf - y0)
        self.battery_pct -= energy.drive_cost_wh(travelled, self.slope_at(xf, yf), 3.0)
        self.battery_pct = max(0.0, self.battery_pct)

    def drive_to(self, tx: float, ty: float, tol: float = 0.18, max_steps: int = 6000) -> dict:
        """Skid-steer toward (tx, ty). Returns {success, distance_m, battery_pct}."""
        x0, y0, _ = self.pose()
        for _ in self.drive_iter(tx, ty, tol, max_steps):
            pass
        xf, yf, _ = self.pose()
        return {"success": math.hypot(tx - xf, ty - yf) < tol * 1.5,
                "distance_m": math.hypot(xf - x0, yf - y0), "battery_pct": self.battery_pct}

    def drive_to_reach(self, tx: float, ty: float, standoff: float = 0.35) -> dict:
        """Drive up to `standoff` metres SHORT of (tx, ty) and stop, facing it — so the rover parks
        beside a sample to reach it with the arm instead of parking on top of it.
        (standoff + drive tolerance stays inside the 0.6 m sampling reach.)"""
        x, y, _ = self.pose()
        d = math.hypot(tx - x, ty - y)
        if d <= standoff:
            return self.drive_to(x, y)
        ux, uy = (tx - x) / d, (ty - y) / d
        return self.drive_to(tx - ux * standoff, ty - uy * standoff)

    def sample(self, target_id: str, reach: float = 0.6) -> dict:
        """Pick up a sample if the rover is close enough (abstract instrument action)."""
        t = next((t for t in self.targets if t.id == target_id), None)
        if t is None:
            return {"success": False, "reason": "unknown target"}
        x, y, _ = self.pose()
        if math.hypot(t.xy[0] - x, t.xy[1] - y) > reach:
            return {"success": False, "reason": "out of reach"}
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, target_id)
        adr = self.model.jnt_qposadr[self.model.body_jntadr[bid]]
        self.data.qpos[adr:adr + 3] = [t.xy[0], t.xy[1], -5.0]   # stow below terrain
        self.data.qvel[self.model.body_dofadr[bid]:self.model.body_dofadr[bid] + 6] = 0
        mujoco.mj_forward(self.model, self.data)
        t.collected = True
        self.battery_pct = max(0.0, self.battery_pct - 0.4)
        return {"success": True, "collected": target_id, "battery_pct": self.battery_pct}
