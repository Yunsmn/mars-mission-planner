# MARVIN — Mars Autonomous Reasoning & Verification INtelligence

**An onboard, offline mission planner for a planetary rover: a small IBM Granite model
_proposes_ action sequences and a fast probabilistic simulator _verifies_ them under
uncertainty — the model never commands the rover directly.**

[![IBM AI Builders Challenge](https://img.shields.io/badge/IBM-AI%20Builders%20Challenge-052FAD)](https://bit.ly/IBMBob-freetrial)
[![IBM Granite 4.1](https://img.shields.io/badge/runtime-IBM%20Granite%204.1-0f62fe)](https://ollama.com/library/granite4.1)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **The tiny model proposes; the physics-lite simulator disposes.**

**Challenge theme:** Space Exploration — turning missions from *data-heavy* into
*insight-driven*, decided onboard.

---

## The problem

A Mars rover cannot think for itself. Today it collects data, waits for an Earth comms
window (**light-time delay of 4–24 minutes each way**), and human planners on Earth analyze
the data, build a command sequence, validate it, and uplink it. Rovers receive **roughly one
command cycle per sol** (Martian day), so a science campaign that needs *K* sequential
decisions takes about *K* sols — a 120-decision campaign is **~4 months** of calendar time,
and the rover sits idle or over-conservative between windows. Every irreversible choice funnels
through Earth.

## What MARVIN does

MARVIN moves the *decision-making* onboard. It turns raw perception into **safe, justified
actions between comms windows** — without a heavy world model, without a network, and without
blindly trusting a small model. It doesn't replace navigation or human oversight; it's the
reasoning layer that decides *what to do next* and delegates execution.

## Results

- **Completes a full 2-sample mission autonomously** on **IBM Granite 4.1 running locally** —
  5 decisions, **~83 seconds**, 4.2% battery, zero ground contact.
- The fast "imagination" simulator **predicts full-physics outcomes within ~3%**, so every
  plan is *verified against reality*, not guessed.
- **Why onboard autonomy matters** (same terrain, same targets):

![Mission playback](docs/figures/mission_frames.png)

![Onboard vs. fixed plan vs. Earth-in-the-loop](docs/figures/comparison.png)

A **deterministic fixed plan built from orbital data alone misses the samples** — orbital
localization error (~1 m) is larger than the instrument's reach (0.55 m), so it drives to the
wrong spot. **Earth-in-the-loop replanning** succeeds but turns a 120-decision campaign into
**~4 months**. MARVIN does it between comms windows. *(Watch the full run in
[`docs/figures/mission.gif`](docs/figures/mission.gif).)*

## AI approach & architecture

Each decision cycle:

1. **Perceive** — noisy onboard sensors (camera intrinsics, pose/odometry drift, target
   detection). The planner acts on *perception*, never ground truth.
2. **Propose** — **IBM Granite 4.1** (local, via Ollama, offline) proposes candidate action
   sequences, following a fixed procedure in [`model/SKILL.md`](model/SKILL.md) so a compact
   model plans reliably instead of improvising.
3. **Verify** — a **vectorized NumPy surrogate** rolls each candidate out **20× under injected
   uncertainty** (traction, localization drift, battery draw), calibrated from the real MuJoCo
   world. Sub-millisecond; the model never commands directly.
4. **Gate** — **tail-risk (CVaR)** on the worst 10% of rollouts, plus the 15% battery reserve.
   Unsafe or over-budget plans are dropped.
5. **Value of Information** — take an extra observation *only* when it would actually change
   the decision (never blind data collection).
6. **Execute** — a **reflex** samples the instant the rover is in reach; a **progress
   guarantee** stops the rover ever idling while the objective is unmet; driving goes through a
   plain in-process capability API — **no server, no MCP, no network**.
7. **Replan** on new perception.

**Two simulators, never confused:** MuJoCo is the *real world* (high-fidelity rover physics on
a Mars heightfield, runs once); the surrogate is the planner's *imagination* (cheap, runs 20×
per decision). The surrogate's uncertainty is calibrated from — and measured against — the
MuJoCo world (the ~3% fidelity number above). Full design in [`docs/DESIGN.md`](docs/DESIGN.md).

## Real space data

MARVIN runs on **real Jezero terrain**. [`data/fetch_jezero.py`](data/fetch_jezero.py) reads the
Jezero window from NASA/USGS's global **MOLA** DEM via GDAL `/vsicurl/` HTTP range requests (no
multi-GB download) — the mission figures show the actual crater basin and rim. Provenance: real
relief **−2781 to −1543 m over 60 km**, amplitude-normalized to the rover-scale patch (the sim
patch is abstracted; the *morphology* is real). Per-target **science value** comes from
CRISM-style mineralogy (carbonate/clay high — see the [scenario](demo/scenario.py)), and **dust
opacity** drives the solar-power budget. Details in [`docs/DATA.md`](docs/DATA.md).

## How IBM Bob was used

**IBM Bob was the primary development tool for MARVIN's intelligence layer.** Bob designed and
authored the first working version of the reasoning stack — the parts that make the rover
*think*:

- the **propose-and-verify loop** ([`planner/loop.py`](planner/loop.py)),
- the **vectorized surrogate simulator** ([`planner/surrogate.py`](planner/surrogate.py)),
- the **tail-risk / CVaR gating** ([`planner/gating.py`](planner/gating.py)),
- the **Value-of-Information** logic ([`planner/voi.py`](planner/voi.py)),
- the **explainable decision log** ([`planner/justify.py`](planner/justify.py)),
- the **surrogate calibration + fidelity metric** ([`world/calibrate.py`](world/calibrate.py)),
- the **IBM Granite proposer** ([`model/propose.py`](model/propose.py)),
- and the **test suite**.

Bob's contributions are visible in the git history under the `IBM Bob` author. The MuJoCo
physics simulation, sensors, and capability API were built as the world for Bob's planner to
reason over, and the integrated system was then hardened for reliability (the sample reflex,
the progress guarantee that breaks stall-loops, the Granite swap, and the planning skill).

## Setup & run

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.txt
ollama pull granite4.1:3b          # IBM Granite 4.1, local & offline

.venv/bin/python -m demo.run        # full autonomous mission (Granite)
.venv/bin/python -m demo.cli        # interactive: drive the rover yourself, or ask Granite to plan
.venv/bin/python -m demo.scenario   # value-aware vs. naive target selection (2.3x science)
.venv/bin/python -m demo.ablation   # no-planner vs. planner-no-sim vs. full (risk earns its place)
.venv/bin/python -m demo.compare    # onboard vs. fixed-plan vs. Earth-in-the-loop
.venv/bin/python -m demo.animate    # render the mission playback (gif + montage)
.venv/bin/python -m web.build_showcase   # build the interactive 3D showcase page
.venv/bin/python -m pytest          # tests
```

## License

MIT — see [`LICENSE`](LICENSE).
