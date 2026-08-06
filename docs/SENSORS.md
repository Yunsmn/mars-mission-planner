# Onboard sensors

The rover senses the world through noisy instruments — the planner acts on **perception,
never the ground-truth state**. That gap is deliberate: it's what makes replan-on-new-
perception and Value-of-Information meaningful. Implemented in `world/sensors.py`.

## Camera

A forward/down camera mounted on the chassis (`rover_cam`, vertical FOV 58°). Intrinsics are
derived analytically from the FOV and a chosen resolution — no rendering required (offscreen
render is unreliable in this environment, so we avoid it):

```
fy = (H / 2) / tan(fovy / 2)      # focal length in pixels (vertical)
fx = fy                            # square pixels
cx = W / 2 ,  cy = H / 2           # principal point
K  = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
```

Default resolution **640×480**, fovy **58°** → `fx = fy ≈ 433.0`, `cx = 320`, `cy = 240`.
Change resolution via `camera_intrinsics(model, width=…, height=…)`; intrinsics scale with it.

> If real image frames are ever needed, MuJoCo can render offscreen through `rover_cam`; the
> intrinsics above are the matching pinhole model for projecting world points to pixels.

## Pose / odometry (IMU-style)

`read_pose(sim, rng)` returns an **estimated** pose = true pose + Gaussian noise:
- position σ ≈ **0.05 m** per axis
- heading σ ≈ **2°**

So the planner's believed position drifts from truth — a short reposition or extra scan
(Value-of-Information) can be worth spending energy to reduce that uncertainty before acting.

## Perception

`observe(sim, rng)` returns a `Perception`:
- `visible_targets` — uncollected sample targets within ~3 m, with **detection noise** (~0.08 m)
  on the reported location.
- `slope_deg`, `roughness` — a local terrain window around the rover (hazard signal).
- `dust_tau` — current atmospheric optical depth (drives the solar-power budget; see
  `docs/DATA.md`).

## What Bob wires on top

The planner should treat these as its only view of the world: call `scan()` to perceive,
reason under the reported uncertainty, and (per the design) decide when an extra observation
is worth its energy cost. Science value per target currently defaults to a placeholder in
`observe()` — the real value comes from CRISM mineralogy (`data/science_value.py`) once the
Jezero data tile is prepared.
