# Component interfaces (the contracts Bob implements)

This is the layering blueprint. Types and signatures are fixed here so modules stay
decoupled; **IBM Bob fills in the bodies** (marked `TODO(bob)` in the stubs). Keep files
small (<400 lines), no magic numbers (read `config.yaml`), immutable data where practical.

## Shared types — `common/types.py`

```python
Vec2 = tuple[float, float]

@dataclass(frozen=True)
class Pose:            xy: Vec2; heading_rad: float
@dataclass(frozen=True)
class Target:         id: str; xy: Vec2; science_value: float; mineral_class: str
@dataclass(frozen=True)
class Constraints:    battery_reserve_pct: float; risk_ceiling: float; p_success_min: float
@dataclass(frozen=True)
class MissionState:   pose: Pose; battery_pct: float; sol_time: float
                      localization_sigma: float; collected: tuple[str, ...]
                      remaining: tuple[Target, ...]
@dataclass(frozen=True)
class Perception:     slope_deg: "np.ndarray"; roughness: "np.ndarray"
                      visible_targets: tuple[Target, ...]; dust_tau: float

class ActionKind(Enum): DRIVE; SCAN; SAMPLE; OBSERVE; HOLD
@dataclass(frozen=True)
class Action:         kind: ActionKind; params: dict          # e.g. {"xy": (x,y)} or {"target": id}
ActionSeq = tuple[Action, ...]

@dataclass(frozen=True)
class RolloutBatch:   success: "np.ndarray"; energy: "np.ndarray"; hazard: "np.ndarray"  # shape (n,)
@dataclass(frozen=True)
class CandidateScore: seq: ActionSeq; p_success: float; cvar_energy: float
                      risk: float; value: float
@dataclass(frozen=True)
class Decision:       action: Action; rationale: str; scores: tuple[CandidateScore, ...]
```

## Execution layer — `rover/capabilities.py` (in-process, NO server)

```python
def get_pose() -> Pose: ...
def battery() -> float: ...                         # percent remaining
def assess_slope(region: Vec2) -> "SlopeStats": ...
def scan() -> Perception: ...                       # refresh perception from onboard sensors
def drive_to(x: float, y: float) -> "DriveResult": ...
def sample(target_id: str) -> "SampleResult": ...   # abstract instrument action (no arm sim)
```

Energy model — `rover/energy.py`
```python
def drive_cost_wh(dist_m: float, slope_deg: float, payload_kg: float) -> float: ...
def solar_charge_wh(dt_s: float, dust_tau: float) -> float: ...   # uses power_factor(tau)
def power_factor(dust_tau: float) -> float: ...
```

## World (MuJoCo ground truth) — BUILT (`world/sim.py`, `world/dem.py`, `world/sensors.py`)

Implemented and tested. `MarsSim` replaced the earlier free-function `scene.py`:

```python
# world/sim.py
class MarsSim:                                          # the ground-truth world + rover control
    def __init__(self, terrain=None, seed=42, targets=None): ...
    def step(self, vL, vR, n=1): ...                    # advance full physics
    def pose(self) -> tuple[float, float, float]: ...   # true (x, y, yaw)
    def slope_at(self, x, y) -> float: ...
    def drive_to(self, tx, ty, ...) -> dict: ...        # {success, distance_m, steps, battery_pct}
    def sample(self, target_id, ...) -> dict: ...       # abstract pickup

# world/dem.py       synthesize() / load_dem() / slope_map() / dem_to_hfield()  — BUILT
# world/sensors.py   camera_intrinsics(model) / read_pose(sim, rng) / observe(sim, rng)  — BUILT
```

## Surrogate calibration — TODO(bob) `world/calibrate.py`

```python
def calibrate_surrogate(model, data, moves) -> "SurrogateEnv":  # fit slip/energy/drift from physics
    ...
def surrogate_fidelity(env, model, data, seq) -> float: ...     # surrogate vs full-physics error
```

The surrogate (`planner/surrogate.py`) samples the ranges from `calibrate_surrogate`;
`surrogate_fidelity` gives the validation number for the writeup.

## Surrogate simulator (the planner's imagination) — `planner/surrogate.py`

```python
@dataclass(frozen=True)
class SurrogateEnv:  slope_deg: "np.ndarray"; roughness: "np.ndarray"; dust_tau: float
                     traction_range: Vec2; loc_drift_range: Vec2; draw_mult_range: Vec2

def rollout_batch(seq: ActionSeq, state: MissionState, env: SurrogateEnv,
                  n: int, rng: "np.random.Generator") -> RolloutBatch: ...
```
- **Vectorized over all `n` rollouts at once** (one batched NumPy pass — no Python loop).
- Uncertainty (traction, localization drift, battery-draw) sampled per rollout from `env`
  ranges → that's what makes the `n` outcomes differ and exposes tail risk.
- No rendering, no contact solve. Target: sub-millisecond for n=20, horizon=100.

## Gating (tail-risk + budget) — `planner/gating.py`

```python
def tail_worst(arr: "np.ndarray", q: float) -> float: ...        # CVaR-style worst (1-q) mean
def score_candidate(seq, batch, state, cfg) -> CandidateScore: ...
def gate(scores: list[CandidateScore], c: Constraints, battery_pct: float
         ) -> list[CandidateScore]: ...                          # safe set only
```

## Value of Information — `planner/voi.py`

```python
def best_gap(safe: list[CandidateScore]) -> float: ...
def cheapest_observation(state: MissionState, perception: Perception) -> Action | None: ...
def voi(obs: Action, state, env) -> float: ...                   # expected risk reduction
def cost(obs: Action) -> float: ...                              # energy/time price
def maybe_observe(safe, state, perception, env, cfg) -> Action | None: ...
```

## Justification log — `planner/justify.py`

```python
def justify(action: Action, scores: list[CandidateScore], model: "Proposer") -> str: ...
```
- Renders a human-readable rationale that cites the actual numbers (success %, tail risk,
  battery). Ground-team audit trail; the backbone of the demo video.

## Model (proposer) — `model/propose.py`

```python
class Proposer:
    def __init__(self, name: str, host: str, temperature: float): ...
    def propose(self, state: MissionState, perception: Perception, k: int) -> list[ActionSeq]:
        ...   # calls Gemma 4 via Ollama; robust structured parse + schema validation
```
- **The model only proposes.** It never executes. Malformed/invalid proposals are dropped.

## Planner loop — `planner/loop.py`

```python
def decide_next_action(state, perception, env, model, cfg) -> Decision: ...
def run_mission(objective: str, world, model, cfg) -> "MissionLog": ...
```
Order inside `decide_next_action`: **propose → rollout_batch → score → gate → maybe_observe
→ select robust best (or HOLD) → justify → return Decision.**

## Demo — `demo/run.py`

Scripted scenario runner: loads data, builds the MuJoCo scene, runs `run_mission`, renders
the dashboard (rollout fan, chosen vs. rejected, battery-vs-science, Earth-vs-onboard timeline).
