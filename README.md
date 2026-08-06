# MARVIN — Mars Autonomous Reasoning & Verification INtelligence

> An **onboard, offline** mission planner for a planetary rover: a **local model
> proposes** action sequences and a **fast probabilistic simulator verifies** them under
> uncertainty. The model never commands the rover directly.
> **The tiny model proposes; the physics-lite simulator disposes.**

Built for the **IBM AI Builders Challenge — August (Space Exploration)**.
Turning space missions from *data-heavy* into *insight-driven* systems.

## The problem

Today a Mars rover collects data → waits for an Earth comms window → humans build,
validate, and uplink commands. That loop is bottlenecked by light-time delay and human
throughput, so the rover is idle or over-conservative between windows. MARVIN moves the
*decision-making* onboard so the rover turns raw data into **safe, justified actions**
between windows — without a heavy world model, without a network, and without blindly
trusting a small model.

## How it works (four layers)

1. **Data** — real NASA orbital + mission data (see [`docs/DATA.md`](docs/DATA.md)).
2. **World** — MuJoCo ground truth: a rover on a **real Mars DEM** heightfield.
3. **Execution** — a plain in-process **capability API** (`drive_to`, `scan`,
   `sample`, `assess_slope`, `battery`) — **no server, no MCP, no network**.
4. **Planner brain** — **Propose-and-Verify**: local model proposes → vectorized
   surrogate runs 20 uncertain rollouts → tail-risk gate → Value-of-Information → execute
   the robust action → log a natural-language justification → replan on new perception.

Full design: [`docs/DESIGN.md`](docs/DESIGN.md).
Component contracts: [`docs/INTERFACES.md`](docs/INTERFACES.md).

## Real space data used

| Stream | Source | Feeds |
|---|---|---|
| Terrain elevation | HiRISE/CTX **DEM** (GALE-Lab / PDS), Jezero crater | MuJoCo heightfield + slope/roughness hazard |
| Orbital imagery | HiRISE **orthoimage (DRG)** | science-target candidates |
| Mineralogy | **CRISM MTRDR** (carbonate / phyllosilicate / olivine) | **data-driven science value** of targets |
| Dust / weather | **MEDA / REMS** dust opacity (τ), PDS Atmospheres | solar-power/energy budget → dynamic replanning |

## Setup & run

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.txt
.venv/bin/python -m demo.run          # scripted drive-and-grab demo
.venv/bin/python -m pytest            # tests
```

## Status

**Working simulation** (built + tested): a skid-steer rover drives terrain under Mars gravity,
grabs samples, senses via camera / noisy pose / perception, and drains a battery. Onboard
capability API is in-process (no server/MCP). Sensors documented in
[`docs/SENSORS.md`](docs/SENSORS.md).

The **LLM planning brain** (propose-and-verify, surrogate, gating, VoI, explanations) is
authored with **IBM Bob** — see [`docs/BUILD_WITH_BOB.md`](docs/BUILD_WITH_BOB.md).

## How IBM Bob was used

IBM Bob is the primary developer of the **mission-planning intelligence layer**: the
propose-and-verify loop, the lightweight surrogate simulator, tail-risk gating,
Value-of-Information, and the explainable decision log. The MuJoCo simulation, sensors, and
capability API were assistant-scaffolded to give Bob a working world to reason over; Bob's
commits appear under its own identity in the history.
<!-- Add screenshots + per-goal notes as Bob builds. -->

## License

MIT — see [`LICENSE`](LICENSE).
