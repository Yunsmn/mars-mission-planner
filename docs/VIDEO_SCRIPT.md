# MARVIN — 3-minute demo video script

**Target: 2:55 (hard cap 3:00).** One continuous story: a real rover got stuck → why → MARVIN
reruns that moment onboard → it works → why it's trustworthy → what it could become.
Record narration (VO) over screen-capture + the rendered clips. Everything on screen is real and
reproducible from this repo.

**Assets to have open/ready:** `web/vendor/purgatory.mp4` (the rover clip), the showcase page
(`web/showcase.html`), a terminal in the venv, `docs/figures/purgatory_rover.png`.

---

### 0:00 – 0:18 · Hook — the stranded rover
**On screen:** slow push on a real NASA image of Opportunity / the Purgatory ripple (or open on
`purgatory_rover.png`, held still), a stark date card: **SOL 446 · 26 APR 2005**.
**VO:**
> "In 2005, NASA's Opportunity rover drove into a wind-blown ripple of soft sand and sank its
> wheels. Getting it out took **38 sols** — five weeks. Four years later, Spirit hit the same kind
> of trap and never got out. The rovers never saw it coming."

### 0:18 – 0:38 · The problem — the decision was made on Earth
**On screen:** simple diagram — rover → 4-24 min light-delay → Earth → command uplink → back.
**VO:**
> "That's because a Mars rover doesn't decide for itself. It sends data to Earth, waits through a
> comms window, and human planners send back one command sequence — about one per day. The choice
> to drive into that dune was made on Earth, from orbit, days ahead of time. No second look."

### 0:38 – 0:58 · Enter MARVIN — the thesis
**On screen:** title **MARVIN**, one line: *the model proposes, the physics-lite simulator
disposes.* Show the propose → verify → gate loop as three chips.
**VO:**
> "MARVIN moves that decision onboard. A small **IBM Granite** model *proposes* what to do — but it
> never commands the rover directly. A fast physics-lite simulator we call the **lightsim** *verifies*
> every proposal against uncertainty first. Propose, then verify. Let me show you on the exact
> moment that stranded Opportunity."

### 0:58 – 1:52 · The Purgatory run — the core (screen-capture the CLI)
**On screen:** terminal. Type it live:
```
python -m demo.cli
marvin> dune          # the ASCII map shows the dune wall ^ between rover @ and outcrop *
marvin> route
```
Let the output play; cut to `purgatory.mp4` for the drive.
**VO (over the CLI output):**
> "Here's the trap — a dune between the rover and the outcrop it needs to sample. Watch the lightsim
> roll each route out twenty times under uncertainty. The direct path — the orbital plan — comes back
> at **100 percent** entrapment risk. So does the north detour. Only the **south detour** is safe, at
> **9 percent**. Granite reads those numbers and commits:"
**On screen highlight the line:** *DECISION (GRANITE): detour south — the only safe route under 10%.*
**VO (cut to the rover clip):**
> "And a safety gate guarantees it can never pick a route the lightsim flagged as a trap. Now it
> drives — real MuJoCo physics, NASA's actual Perseverance model — around the dune, and caches the
> sample. Reached. Sampled. **Zero sols lost**, against Opportunity's thirty-eight."

### 1:52 – 2:18 · Why you can trust a small model
**On screen:** the showcase page — Situation panel (38 → 0), the route table, telemetry.
**VO:**
> "The trick is that the model never acts on trust. Two simulators, never confused: MuJoCo is the
> real world; the lightsim is the rover's cheap imagination, and its risk numbers are calibrated
> against that real world to within about three percent. Every plan is gated on **worst-case**
> risk, not the average — and if the model ever picks something unsafe, the gate overrides it. It
> runs on real Jezero terrain from NASA's MOLA elevation data, fully **offline** — no server, no
> network, nothing you can't run on a spacecraft."

### 2:18 – 2:42 · What it could become + IBM Bob
**On screen:** quick montage — the decision log, `git log --author="IBM Bob"`, the architecture chips.
**VO:**
> "Nothing here is specific to one dune. It's a reusable **onboard decision layer** — give it any
> rover, any terrain, any hazard, and it proposes, verifies, and justifies every move. The reasoning
> stack — the propose-verify loop, the surrogate, the risk gating — was built with **IBM Bob**, and
> it runs on **IBM Granite 4.1**."

### 2:42 – 2:55 · Close
**On screen:** MARVIN mark, repo URL, "Onboard · Offline · Verified."
**VO:**
> "Opportunity waited five weeks for Earth to dig it out. MARVIN decides in seconds, onboard, and
> never drives in. That's the difference between data-heavy and insight-driven. MARVIN."

---

**Recording tips**
- Do the CLI segment in one clean take; if Granite's call is slow, pre-warm it (`route` once before
  recording) so the on-camera run is snappy — the decision is deterministic (seeded).
- Keep VO under the cap; 2:55 leaves headroom. Trust the visuals during the drive — don't over-talk.
- Captions on the risk numbers (100% / 100% / 9%) and the DECISION line land the point without VO.
- Cut, don't fade — judges watch fast.
