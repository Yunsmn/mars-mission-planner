# MARVIN — Mars Autonomous Reasoning & Verification INtelligence

**Master concept + technical design.** This is the authoritative spec: the plan is
"fixed" here, and this document is what we hand to **IBM Bob** to build against.

> **One-liner:** An onboard, offline mission planner for a planetary rover in which a
> *small local language model proposes* action sequences and a *fast probabilistic
> simulator verifies* them under uncertainty — the model never commands the rover
> directly. **The tiny model proposes; the physics-lite simulator disposes.**

Working title **MARVIN** — rename freely.

---

## 1. Problem

Modern planetary missions are **data-heavy but insight-poor**. Today a Mars rover:
1. collects data, 2. waits for an Earth comms window, 3. humans on Earth analyze it,
build a command sequence, validate it, and 4. uplink it — a loop bounded by
light-time delay (4–24 min one way) and human throughput. The rover is idle or
conservative between windows, and irreversible/risky decisions all funnel through Earth.

**Goal:** move decision-making *onboard* so the rover turns raw data into *actions*
between comms windows — smarter, safer, more autonomous — **without** a large world
model, **without** a network, and **without** trusting an unreliable small model blindly.

## 2. Solution overview

A self-contained onboard planner with a **Propose-and-Verify** architecture:

- A **local LLM** run onboard (**Gemma 4** via Ollama) reads the current perception +
  mission state and **proposes** candidate action sequences.
- A **fast, vectorized surrogate simulator** stress-tests each candidate **20× under
  injected uncertainty** (traction, localization, energy) in a fraction of a second.
- Candidates are **gated on tail-risk** (worst-case, not average) and scored on science
  value vs. energy/time budget. Only **robust** actions execute.
- Every accept/skip emits a **human-readable justification** (auditable ground trace).
- A **Value-of-Information** step decides *when it is worth spending energy/time to
  observe more* before committing, rather than collecting data blindly.

Everything runs **onboard and offline**: the planner drives its actuators through a plain
in-process **capability API** — **no server, no MCP, no network** at flight time (you
can't mount a server on Mars). The flight artifact is a single self-contained process.

## 3. Challenge fit & how this goes beyond NASA

NASA already deploys pieces of autonomy, but **separate, ground-side, or deterministic**:

| NASA capability | What it does | Our advance |
|---|---|---|
| **AEGIS** | Onboard science-target selection from imagery | We *plan the whole action sequence* onboard, not just pick a target |
| **Onboard Planner (OBP)** | Deterministic energy/time scheduling (~20% energy, ~25% fewer days) | We add **probabilistic lookahead + tail-risk gating** (uncertainty-aware, not deterministic) |
| **Dec-2025 AI-planned drive** (JPL + Anthropic/Claude) | LLM plans waypoints — **but Claude ran on Earth** | Our LLM is **small, local, offline**, and **never trusted directly** — verified by simulation |

**Novel contribution (the spine):** *onboard + offline + small-model + probabilistic
propose-and-verify with tail-risk gating.* Nobody has deployed this combination. It is
also the **safety story**: this is *how you make a 4B model mission-safe.*

Two stacked innovations:
- **Value-of-Information (VoI) perception** — active, budgeted observation. The direct
  embodiment of "insight-driven, not data-heavy."
- **Explainable decision log** — natural-language rationale per decision, an auditable
  trace for the ground team → trust.

*(Stretch, Phase-3 only: comms-window-aware planning — defer irreversible risky actions
until after the next Earth check-in when a window is near.)*

## 4. System architecture

```
                    ┌──────────────────────────────────────────────┐
                    │              MISSION OBJECTIVE                 │
                    │   e.g. "characterize the outcrop to the NE,    │
                    │    cache 2 high-value samples, stay >15% batt" │
                    └──────────────────────┬───────────────────────┘
                                           ▼
   ┌───────────────────────────  PLANNER BRAIN (onboard, offline)  ──────────────────┐
   │                                                                                  │
   │   PERCEPTION ──►  small LLM PROPOSES ──►  SURROGATE VERIFIES 20×  ──► GATE ──►    │
   │   (latest obs)    (Gemma 4, N cands)      (vectorized numpy, noise)  (tail-risk    │
   │        ▲                                                              + budget)    │
   │        │                                                                 │        │
   │        │            VALUE-OF-INFORMATION: observe more or act now? ◄─────┤        │
   │        │                                                                 ▼        │
   │        │                                                    EXECUTE robust action │
   │        │                                                    + log justification   │
   │        │                                                                 │        │
   │        └──────────────── replan on new perception ◄─────────────────────┘        │
   └──────────────────────────────────────┬───────────────────────────────────────────┘
                                           ▼  (direct in-process calls — no server)
   ┌───────────────  EXECUTION LAYER — onboard capability API (in-process fns)  ────────┐
   │  drive_to(x,y) · assess_slope() · scan() · sample(target) · get_pose() · battery() │
   └──────────────────────────────────────┬───────────────────────────────────────────┘
                                           ▼
   ┌────────────  WORLD = MuJoCo ground truth — HIGH FIDELITY (runs once)  ─────────────┐
   │  Detailed rover (Rover4We fork): suspension + per-wheel drive, wheel–terrain        │
   │  contact & slip on a REAL Mars DEM hfield; NOISY onboard sensors                    │
   │  (pose/IMU/odometry/vision) feed PERCEPTION. Sampling = instrument action (no arm). │
   └───────────────────────────────────────────────────────────────────────────────────┘
```

**Two simulators, never confused:**
- **MuJoCo = the real world — high fidelity.** Full rigid-body dynamics, rover suspension,
  wheel–terrain contact and slip on the real DEM, noisy sensors. Runs *once* as the
  environment. Slow and detailed on purpose — this is what makes the demo credible and the
  yardstick the surrogate is measured against.
- **Surrogate = the rover's imagination — lightweight.** Pure NumPy, no rendering/contact
  solve; runs 20× per decision in ~ms. Approximate on purpose.
- **They are linked (this is important):** the surrogate's uncertainty ranges are
  **calibrated from the detailed world** — drive a set of moves in MuJoCo, fit the
  slip/energy/drift distributions the surrogate samples. We can then report
  **surrogate-vs-world fidelity** (how well the cheap model predicts full-physics outcomes) —
  a concrete validation number for the writeup, and the honest basis for trusting the fast sim.

## 5. Component detail

### 5.1 The Propose-and-Verify loop (core algorithm)

```python
def decide_next_action(state, perception):
    # 1. PROPOSE — small local LLM, offline
    candidates = llm.propose(state, perception, k=N_CANDIDATES)   # list of action seqs

    # 2. VERIFY — vectorized surrogate rollouts under uncertainty
    scored = []
    for cand in candidates:
        outcomes = surrogate.rollout_batch(cand, state, n=20)     # (20,) results, ~µs
        p_success   = mean(outcomes.success)
        cvar_energy = tail_worst(outcomes.energy, q=0.90)         # worst 10% energy use
        risk        = tail_worst(outcomes.hazard, q=0.90)         # worst 10% hazard
        value       = science_value(cand) - energy_penalty(cvar_energy)
        scored.append((cand, p_success, risk, value))

    # 3. GATE on tail-risk + budget (safety-first, not average-case)
    safe = [c for c in scored if c.risk <= RISK_CEIL
                              and c.p_success >= P_MIN
                              and respects_battery_reserve(c)]

    # 4. VALUE OF INFORMATION — is it worth observing more before committing?
    if best_gap(safe) < VOI_THRESHOLD:            # top candidates too close / too risky
        obs = cheapest_disambiguating_observation(state)
        if voi(obs) > cost(obs):                  # expected risk reduction beats cost
            return Observe(obs)                   # act to learn, then re-decide

    # 5. SELECT robust best + EXPLAIN
    choice = argmax_value(safe) if safe else Safe_hold()
    log.justify(choice, scored)                   # NL rationale from the model
    return choice
```

### 5.2 Surrogate rollout simulator (the "20× in fractions of a second")

- **Reduced model**, pure NumPy, fully **vectorized over the 20 samples at once**
  (one batched op, not a Python loop).
- State per step: position, heading, battery, localization uncertainty.
- Dynamics: path integrated over the **DEM-derived slope map**; energy = f(distance,
  slope, payload); slip/hazard = f(slope, terrain roughness) with a probability of
  getting stuck rising sharply past a slope threshold.
- **Uncertainty injected** each rollout: traction coefficient, localization drift,
  battery-draw multiplier — sampled from ranges **calibrated against the detailed MuJoCo
  world** (§5.6). This is what makes the 20 rollouts *differ* and reveals tail-risk.
- Output per rollout: `success` (reached goal?), `energy` used, peak `hazard`.
- Budget check: 20 rollouts × ~100-step horizon = ~2k vectorized ops ≈ **sub-millisecond**;
  even N candidates × 20 rollouts stays well under a second — the AI runs its plans *before* it acts.
- **Fidelity metric:** periodically compare surrogate predictions to a full MuJoCo rollout and
  report the error (e.g. arrival/energy within X%) — keeps the cheap model honest.

### 5.3 Execution layer — a plain onboard capability API

The planner drives the rover through ordinary **in-process Python functions** — a small
capability API: `drive_to(x,y)`, `assess_slope()`, `scan()`, `sample(target)`,
`get_pose()`, `battery()`.

**No server, no MCP, no network.** You can't mount an MCP server on Mars; the flight
system is a single self-contained process where the planner calls its actuators directly.
Simpler, faster, and honest to the deployment target. (A ground-operator interface, if
ever wanted, is a separate concern built later — *not* part of the flight architecture.)

`sample(target)` is an **abstract instrument action** (drill/scoop/spectrometer at a
reached location) — **no full arm sim**. This keeps the physics effort on mobility over
real terrain and makes the project fully independent of any prior work. All of this is
**new code, authored during the challenge with IBM Bob** (§11).

### 5.4 Value-of-Information perception

- The planner does **not** collect data blindly. When candidate actions are close in
  value or one is borderline-risky, it asks: *would a cheap observation (an extra scan,
  a short reposition) reduce my decision risk by more than it costs in energy/time?*
- If yes → take the observation, update perception, re-decide. If no → act now.
- This is the concrete "**insight-driven, not data-heavy**" mechanism and a clean demo beat.

### 5.5 Explainable decision log

- Every accept/skip is logged with a model-generated rationale referencing the actual
  numbers, e.g.:
  > *"Skipped target B (science 0.3): 42% traverse-failure over 20 rollouts due to a
  > 22° slope on the direct path; detour would breach the 15% battery reserve. Chose
  > target A (science 0.8, 4% failure, 31% battery remaining)."*
- Serves as the **ground-team audit trail** and the backbone of the demo video.

### 5.6 World fidelity (the detailed ground-truth sim)

The MuJoCo world is deliberately **detailed** — it stands in for reality, so the demo is
credible and the surrogate has a real yardstick:
- **Rover dynamics:** Rover4We fork with **suspension and per-wheel drive**; full rigid-body
  **wheel–terrain contact** on the DEM, so slip / sink / tip **emerge from physics**, not a lookup.
- **Terrain:** the real Jezero DEM as a MuJoCo `hfield`, with friction / rolling-resistance
  tuned per terrain type.
- **Noisy onboard sensors** (so perception is realistic): pose/odometry drift, IMU tilt, and a
  forward/down view used to detect targets and local hazards. **Perception is what the planner
  sees — never the ground-truth state.**
- **Power draw** measured from actual motion, feeding the energy model.

The surrogate (§5.2) is a cheap approximation of *this* world, **calibrated from it**
(`world/calibrate.py`). Keeping the two separate is the whole point: the detailed world is
truth; the lightweight sim is how the AI thinks fast before committing.

## 6. Data (real NASA, not toy)

- **Source:** UCLA GALE Lab Mars DEM archive on NASA PDS / USGS Astrogeology
  (`github.com/GALE-Lab/Mars_DEMs`). HiRISE DEMs @ **1 m/px GeoTIFF**, each bundled with
  its **orthoimage (DRG)**.
- **Region:** a **Jezero crater** tile — Perseverance's real terrain (narrative payoff).
- **DEM → terrain:** GeoTIFF → NumPy array → MuJoCo `<hfield>` heightfield (rover drives
  authentic elevation). Slope map = gradient of the DEM (feeds hazard model).
- **Ortho → targets:** the orthoimage provides candidate science targets (AEGIS-style),
  plus a believable visual backdrop for the demo.

## 7. Runtime constraints (design contract)

- **No network at flight time** — planner + verbs are in-process.
- **Local model** — **Gemma 4** run locally via Ollama, offline. The propose-and-verify
  gate is what makes an onboard model's proposals trustworthy without blind faith.
- **Compute budget** — a decision cycle = 1 LLM proposal call + a sub-ms surrogate batch.
- **Determinism for the demo** — fixed RNG seeds so the recorded run is reproducible.

## 8. Tech stack

| Concern | Choice |
|---|---|
| Physics / world | MuJoCo (already in use) |
| Rover body | fork of `griloHBG/Rover4We` (MJCF, diff-drive/Ackerman) |
| Surrogate sim | NumPy (vectorized) |
| Local model | Ollama **Gemma 4** (offline) |
| Execution layer | Plain Python capability API, in-process — **no server, no MCP** |
| Data | GDAL/rasterio to read GeoTIFF DEM+ortho → heightfield |
| Dashboard / demo | lightweight (matplotlib/HTML) — rollout fan, plan, battery-vs-science, Earth-vs-onboard timeline |

## 9. Repo structure (fresh PUBLIC repo)

```
mars-mission-planner/
├── README.md               # problem, solution, architecture, challenge fit, "How IBM Bob was used"
├── docs/DESIGN.md          # this document
├── data/                   # DEM/ortho tile + loader (or fetch script; heavy files gitignored)
├── world/                  # MuJoCo scene: rover + hfield terrain (real Mars DEM)
├── rover/                  # onboard capability API (drive, scan, sample, telemetry) — in-process, no server
├── planner/                # propose-and-verify loop, surrogate sim, VoI, gating, justification log
├── model/                  # Gemma 4 proposal prompts + parsing
├── demo/                   # scripted scenario + dashboard + video assets
└── tests/                  # surrogate correctness, gate logic, end-to-end scenario
```

Fresh public repo (not the private internship repo) — clean history, no internship IP
leak, tidy "How Bob was used" story. Commit identity: `younes.menfalouti@um6p.ma`.

## 10. Phased plan → Aug 31 (solo)

| Phase | Dates | Outcome |
|---|---|---|
| **0 — Skeleton & data** | Aug 6–9 | Fork Rover4We; download a Jezero DEM+ortho tile; DEM→hfield loader; rover drives real terrain headless in MuJoCo |
| **1 — Execution layer** | Aug 10–15 | Rover capability API (drive/scan/sample) + battery/energy model; scripted "drive to target → sample → cache" end-to-end |
| **2 — The brain** | Aug 16–23 | Propose-and-Verify loop, surrogate sim, tail-risk gate, VoI, justification log, LLM wiring. **Bob does the heavy lifting here.** |
| **3 — Demo polish** | Aug 24–27 | Dashboard + scripted scenario (replan + look-before-leap + triage); record ≤3-min video. *(Stretch: comms-window layer.)* |
| **4 — Submit** | Aug 28–31 | Public repo + README (incl. "How IBM Bob was used"), video upload, platform submission. Buffer. |

**IBM Bob trial (40 Bobcoins / 30 days):** start at **Phase 2**, so it covers the
build-heavy stretch through submission — not now.

## 11. How IBM Bob is used (the required proof)

Bob is our **primary development partner across the SDLC** — the entire codebase is new
and authored during the challenge with Bob (no reused prior project), so the "built with
Bob" story is clean and unambiguous.

- **Scaffolding & planning:** Bob sets up the `planner/`, `rover/`, and `world/` packages
  and their structure.
- **Core implementation:** Bob writes the propose-and-verify loop, the vectorized
  surrogate simulator, the tail-risk gate, VoI logic, and the justification log.
- **Testing:** Bob generates the unit + end-to-end tests (surrogate correctness, gate
  logic, scenario).
- **Evidence:** a commit/dev trail + a README "How IBM Bob was used" section with
  screenshots — Bob is demonstrably the tool that built the prototype.
- The flight artifact stays self-contained (no MCP/server); Bob is the *dev* tool, not a
  runtime dependency.

## 12. Demo / video script (≤3 min)

1. **Hook (0:20):** real Jezero terrain in MuJoCo; "everything you'll see runs onboard,
   offline, on a 4-billion-parameter model — no Earth in the loop."
2. **Propose-and-Verify (0:50):** LLM proposes 3 routes to a science target; surrogate
   fans out 20 uncertain rollouts; a risky route shows high tail-risk and is **rejected**;
   a robust one is chosen — with its natural-language justification on screen.
3. **Replan on new perception (0:40):** a hazard/slope is revealed mid-traverse; the
   planner visibly changes plan, no human input.
4. **VoI + triage (0:40):** planner spends a small scan to disambiguate, then drops a
   low-value far sample to protect battery for a high-value one.
5. **Impact close (0:20):** Earth-in-the-loop timeline vs. onboard — acts now vs. waits.

## 13. Novel contributions (summary)

1. **Onboard propose-and-verify** with a local model — the model never commands
   directly; a physics-lite simulator gates every action. *(How to trust an onboard model.)*
2. **Tail-risk (worst-case) gating** over cheap uncertain rollouts — safety-first, beyond
   NASA's deterministic OBP.
3. **Value-of-Information perception** — budgeted active observation ("insight, not data").
4. **Explainable decision log** — auditable NL rationale for the ground team.

## 14. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Mobile-rover sim eats the timeline | Fork Rover4We + DEM→hfield (reuse, not build); Phase 0 de-risks first |
| Headless MuJoCo camera render crashes (known from internship) | Perception from DEM/ortho arrays, not live camera render |
| Onboard model proposes garbage | It only *proposes* — the verify-gate rejects unsafe/incoherent plans; fall back to `Safe_hold()` |
| Surrogate too slow | Vectorize all 20 rollouts as one batched NumPy op; short horizon |
| Bob trial (30 days) expiring | Start trial at Phase 2, not now |

## 15. Open decisions / assumptions

- **Assumed:** fresh public repo named `mars-mission-planner` (rename OK).
- **Assumed:** **Gemma 4** as the flight model (run locally via Ollama, offline).
- **Assumed:** sampling is an **abstract instrument action** (no full arm sim); project is
  fully independent of any prior work.
- **Assumed:** **no MCP / no server** anywhere in the flight architecture.
- **To pick in Phase 0:** the exact Jezero DEM tile (from the GALE-Lab list).
- **Deferred:** comms-window-aware planning (Phase-3 stretch only).
```
