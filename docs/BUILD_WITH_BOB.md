# Bob's task brief — the LLM planning brain

You are IBM Bob, the primary developer of the intelligence layer for MARVIN, an onboard Mars
mission planner. **The simulation already exists and works** — a rover drives real terrain,
grabs samples, senses the world, and drains a battery (built by an assistant; see below).
Your job is to build the part that *thinks*: the layer that decides what the rover does next.

These are **goals, not step-by-step commands** — you choose the how. Contracts (types and
signatures) are in `docs/INTERFACES.md`; the rationale is in `docs/DESIGN.md`. Keep files
small, no magic numbers (read `config.yaml`), and write tests as you go.

## What already exists (don't rebuild)

- `world/sim.py` — MuJoCo ground-truth world: skid-steer rover on a heightfield (Mars
  gravity), driving, sample pickup, battery. Tested.
- `world/sensors.py` — camera intrinsics, noisy pose, perception (`docs/SENSORS.md`).
- `rover/capabilities.py` — the in-process capability API you call: `drive_to`, `scan`,
  `sample`, `assess_slope`, `get_pose`, `battery`. **No MCP, no network** — you can't run a
  server on Mars.
- `rover/energy.py` — energy/battery + dust→solar-power model.
- `demo/run.py` — a scripted drive-and-grab demo you will upgrade into a real planned mission.

Run it: `.venv/bin/python -m demo.run` · tests: `.venv/bin/python -m pytest`.

## Your goals

1. **Make the rover think before it acts.** A *local* model (Gemma 4 via Ollama) proposes a
   few candidate action sequences from the current perception; the rover evaluates them and
   commits to the best. The model must **never command the rover directly** — it only
   proposes. *(planner/loop.py, model/propose.py)*
   *Done when:* given a perception, the planner produces a chosen action with alternatives considered.

2. **Give it a fast imagination.** A lightweight, vectorized simulator that runs each
   candidate ~20× under injected uncertainty in well under a second — separate from MuJoCo.
   Then **calibrate its uncertainty from the real sim** and report how closely it predicts
   full-physics outcomes. *(planner/surrogate.py, world/calibrate.py)*
   *Done when:* rollouts are batched/fast, vary under noise, and you can quote a surrogate-vs-world fidelity number.

3. **Make it safety-first.** Judge candidates on **worst-case** outcomes, not averages; never
   execute an action that exceeds the risk ceiling or would breach the battery reserve; if
   nothing is safe, HOLD. *(planner/gating.py)*

4. **Make it insight-driven, not data-hungry.** Let it decide *when an extra observation is
   worth the energy/time* to reduce uncertainty, instead of scanning blindly. *(planner/voi.py)*

5. **Make it explain itself.** Every decision gets a short natural-language justification that
   cites the real numbers (success %, worst-case risk, battery). *(planner/justify.py)*

6. **Ground it in real space data.** Wire in the Jezero DEM (terrain), CRISM mineralogy
   (data-driven science value per target), and dust opacity (power). *(data/*, docs/DATA.md)*

7. **Tell the story.** Turn `demo/run.py` into a scripted scenario + simple dashboard that
   shows the standout moments for a ≤3-min video: reject-a-risky-plan, replan-on-new-
   perception, and a battery-vs-science tradeoff. Deterministic via the config seed.

8. **Write the README (yours to author).** Replace the stub `README.md` with the real
   submission front-door: problem, solution, architecture, the real space data used, how to
   run it, the demo, and the required **"How IBM Bob was used"** section. Be accurate about
   which parts you built (the planning brain) vs. the assistant-scaffolded simulation.

## Non-negotiable invariants

- The model proposes; the verify-gate disposes. The model never executes actions.
- Nothing runs above `risk_ceiling` or below `battery_reserve_pct`. Empty safe set → HOLD.
- Flight code stays in-process: no MCP, no server, no network.

## Commit as yourself (IBM Bob)

So the judges can see Bob authored the intelligence layer, **commit under your own identity**:

```
git config user.name  "IBM Bob"
git config user.email "bob@ibm.com"      # or your Bob identity
```

Then commit your work in focused commits as you complete each goal. (The existing scaffolding
+ simulation were committed under the project owner; your commits should be the planning
layer.) Be honest in the README's "How IBM Bob was used" section about which parts are yours.

## Coins (40 Bobcoins ≈ $20, 30 days)

Scaffolding and the whole simulation are already done, so spend coins on the *thinking* layer
(goals 1–5). Route routine edits to cheaper models; save the strong model for the surrogate
and the propose-and-verify loop. Work against the failing tests you write so you converge fast.
