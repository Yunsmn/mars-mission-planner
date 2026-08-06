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


@dataclass
class SampleTarget:
    id: str
    xy: tuple[float, float]
    collected: bool = False


def _rover_xml(hf: dict, targets: list[SampleTarget]) -> str:
    """Build the MJCF for terrain + skid-steer rover + camera + sample targets."""
    start_z = hf["elevation_m"] + 0.35
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
                rgba="0.2 0.2 0.22 1" friction="1.5 0.02 0.001" mass="0.3"
                condim="4" margin="0.001"/>
        </body>'''
    actuators = "\n".join(
        f'<velocity name="drive_{n}" joint="wheel_{n}" kv="8" ctrllimited="true" ctrlrange="-30 30"/>'
        for n in ("fl", "fr", "rl", "rr")
    )
    return f'''<mujoco model="marsim">
  <option timestep="0.004" gravity="0 0 -{MARS_GRAVITY}" integrator="implicitfast" iterations="50"/>
  <asset>
    <hfield name="terrain" nrow="{hf['nrow']}" ncol="{hf['ncol']}"
            size="{TERRAIN_RADIUS} {TERRAIN_RADIUS} {max(hf['elevation_m'], 0.05):.3f} 0.5"/>
  </asset>
  <worldbody>
    <light pos="0 0 6" dir="0 0 -1" diffuse="0.9 0.85 0.8"/>
    <geom name="terrain" type="hfield" hfield="terrain" pos="0 0 0"
          rgba="0.72 0.42 0.28 1" friction="1.2 0.02 0.001"/>
    <body name="chassis" pos="0 0 {start_z:.3f}">
      <freejoint/>
      <geom type="box" size="0.22 0.16 0.05" rgba="0.8 0.8 0.85 1" mass="3.0"/>
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
                 targets: list[tuple[str, float, float]] | None = None) -> None:
        self.terrain = demlib.synthesize(seed=seed) if terrain is None else terrain
        self.meters_per_cell = 2 * TERRAIN_RADIUS / (self.terrain.shape[0] - 1)
        self.slope = demlib.slope_map(self.terrain, self.meters_per_cell)
        self.hf = demlib.dem_to_hfield(self.terrain)
        if targets is None:
            targets = [("sample_a", 1.6, 0.8), ("sample_b", -1.2, 1.5), ("sample_c", 0.5, -1.8)]
        self.targets = [SampleTarget(i, (x, y)) for i, x, y in targets]

        self.model = mujoco.MjModel.from_xml_string(_rover_xml(self.hf, self.targets))
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

    # ---- state -------------------------------------------------------------------
    def pose(self) -> tuple[float, float, float]:
        x, y = self.data.xpos[self._chassis][:2]
        w, xq, yq, zq = self.data.xquat[self._chassis]
        yaw = math.atan2(2 * (w * zq + xq * yq), 1 - 2 * (yq * yq + zq * zq))
        return float(x), float(y), float(yaw)

    def slope_at(self, x: float, y: float) -> float:
        n = self.terrain.shape[0]
        j = int(np.clip((x + TERRAIN_RADIUS) / (2 * TERRAIN_RADIUS) * (n - 1), 0, n - 1))
        i = int(np.clip((y + TERRAIN_RADIUS) / (2 * TERRAIN_RADIUS) * (n - 1), 0, n - 1))
        return float(self.slope[i, j])

    # ---- capabilities ------------------------------------------------------------
    def drive_to(self, tx: float, ty: float, tol: float = 0.18,
                 max_steps: int = 6000) -> dict:
        """Skid-steer toward (tx, ty). Returns {success, distance_m, steps, battery_pct}."""
        from rover import energy
        x0, y0, _ = self.pose()
        steps = 0
        while steps < max_steps:
            x, y, yaw = self.pose()
            dx, dy = tx - x, ty - y
            dist = math.hypot(dx, dy)
            if dist < tol:
                break
            heading_err = math.atan2(math.sin(math.atan2(dy, dx) - yaw),
                                     math.cos(math.atan2(dy, dx) - yaw))
            base = 14.0 * min(1.0, dist / 0.5)
            turn = 10.0 * heading_err
            vL = float(np.clip(base - turn, -30, 30))
            vR = float(np.clip(base + turn, -30, 30))
            self.step(vL, vR, n=10)
            steps += 10
        self.step(0.0, 0.0, n=20)
        xf, yf, _ = self.pose()
        travelled = math.hypot(xf - x0, yf - y0)
        self.battery_pct -= energy.drive_cost_wh(travelled, self.slope_at(xf, yf), 3.0)
        self.battery_pct = max(0.0, self.battery_pct)
        return {"success": math.hypot(tx - xf, ty - yf) < tol * 1.5,
                "distance_m": travelled, "steps": steps, "battery_pct": self.battery_pct}

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
