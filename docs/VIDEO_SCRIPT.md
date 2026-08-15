# MARVIN — 3-minute demo video (cinematic cut)

**Target 2:55, hard cap 3:00.** One story, cut fast, scored like a mission film. Everything on
screen is real and reproducible from this repo. You record VO + the CLI capture; the rendered clips
are already in `web/vendor/`.

**Tone:** restrained, high-stakes, a little cinematic — think mission control, not a startup sizzle
reel. Low ambient pad under the whole thing; a single music swell at the reveal (~0:58) and again at
the split-screen (~1:40). Cut, don't fade. Let the visuals breathe; don't over-narrate.

**Clips ready to drop in:**
- `web/vendor/purgatory.mp4` — hero rover clip (Perseverance mesh rounding the dune, chase-cam).
- `web/vendor/compare.mp4` — the split-screen proof (stuck vs routed).
- `docs/figures/purgatory_rover.png` — hero still for the title card.
- Live screen-capture of `demo.cli` for the control beat.

---

### 0:00 – 0:16 · COLD OPEN — the true story
**Visual:** black. A single line types on: **`SOL 446 · MERIDIANI PLANUM · 2005`**. Cut to a real
NASA image of Opportunity / the Purgatory ripple (or hold on `purgatory_rover.png`, desaturated).
**VO (quiet):**
> "In 2005, a rover named Opportunity drove into a ripple of soft sand and sank. Digging it out
> took thirty-eight days. Four years later, Spirit hit the same kind of trap — and never moved
> again."
**Text card:** *38 sols lost. The decision to drive in was made on Earth, days earlier.*

### 0:16 – 0:36 · THE GAP
**Visual:** clean motion graphic — rover → dashed line → **4–24 min** → Earth → command → back.
The loop pulses slowly, deliberately sluggish.
**VO:**
> "A Mars rover can't decide for itself. It ships data to Earth, waits out a comms window, and
> gets back one command sequence a day. Between windows, it's blind. It does exactly what it was
> told — even when what it was told is *drive into the dune*."

### 0:36 – 0:58 · THE IDEA — MARVIN
**Visual:** title **MARVIN** — *Mars Autonomous Reasoning & Verification INtelligence*. One line
under it: **the model proposes · the physics-lite sim disposes.** Three chips light in sequence:
**PROPOSE → VERIFY → GATE**.
**VO:**
> "MARVIN puts the decision back on the rover. A small IBM Granite model *proposes* what to do —
> but it never touches the wheels. A fast physics simulator we call the lightsim *verifies* every
> idea against uncertainty first, and a safety gate throws out anything that looks like a trap.
> Here's that exact 2005 moment, rerun onboard."
*(music swell)*

### 0:58 – 1:38 · YOU'RE IN CONTROL — the CLI (live screen-capture)
**Visual:** terminal, big readable font. Type it live; let the output print.
```
$ python -m demo.cli
marvin> dune          # ASCII map: the dune wall ^ sits between rover @ and outcrop *
marvin> route
```
Let the assessment print, then **freeze-frame + zoom** on these three lines:
```
direct         entrapment risk 100%   WOULD GET STUCK
detour north   entrapment risk 100%   WOULD GET STUCK
detour south   entrapment risk   9%   SAFE
DECISION (GRANITE): detour south — the only safe route under 10%.
```
**VO (over the CLI):**
> "This isn't a canned animation — it's the tool a mission operator would actually drive. Load the
> trap, and ask MARVIN to plan. The lightsim rolls each route out twenty times. The direct path —
> the orbital plan — comes back at a hundred percent risk. So does the north detour. Only the south
> detour is safe. Granite reads the numbers and commits — and the gate guarantees it can never pick
> a route the sim flagged. You can also drive it yourself, or hand Granite the whole mission."

### 1:38 – 2:06 · THE PROOF — split-screen (drop in `compare.mp4`)
**Visual:** full-frame `compare.mp4`. Same dune, same camera. Left augers in and stalls; right arcs
around and the sample box vanishes. Let the **STUCK AT DUNE** / **SAMPLE CACHED** labels land.
*(music swell)*
**VO:**
> "Same dune. Same rover. On the left, the plan from orbit — into the sand, wheels buried, exactly
> like Opportunity. On the right, MARVIN's verified route — around the dune, sample cached.
> Thirty-eight sols lost… versus zero."

### 2:06 – 2:30 · REAL DATA, REAL ROVER
**Visual:** Jezero MOLA terrain figure; then a simple block diagram — **flight sensors → MARVIN
(propose · verify · gate) → drive system** — drawn as a drop-in module beside the existing autonomy
stack.
**VO:**
> "The terrain is real — NASA's MOLA elevation data for Jezero Crater, pulled from public archives.
> Today those archives stand in for a live feed; on a real rover, the same interfaces take real-time
> telemetry instead. MARVIN is built to slot in as an onboard module — between the sensors the rover
> already has and the drive system it already trusts. Two simulators, never confused: the real
> physics, and the rover's cheap imagination, calibrated to it within about three percent."

### 2:30 – 2:44 · WHY IT'S DIFFERENT + IBM
**Visual:** quick montage — decision log scrolling, `git log --author="IBM Bob"`, "Onboard ·
Offline · Verified."
**VO:**
> "It runs fully offline — no server, nothing you couldn't fly. watsonx lives on Earth; the whole
> point is Granite reasoning *onboard*. The reasoning stack was built with IBM Bob, on IBM Granite
> 4.1. And nothing here is specific to one dune — it's a reusable decision layer for any rover, any
> hazard."

### 2:44 – 2:55 · CLOSE
**Visual:** the MARVIN mark, repo URL, final line: *Opportunity waited five weeks. MARVIN decides
in seconds.*
**VO:**
> "Opportunity waited five weeks for Earth to dig it out. MARVIN decides in seconds — onboard — and
> never drives in. Data-heavy, made insight-driven. MARVIN."

---

**Recording notes**
- Pre-warm Granite (`route` once) before the live take so the on-camera call is instant — the
  decision is seeded and deterministic, so it'll say the same thing.
- Captions on the risk numbers (100 / 100 / 9) and the DECISION line carry the point even on mute.
- The split-screen is your strongest 10 seconds — don't rush it; let the labels read.
- Keep total VO under the cap; 2:55 leaves headroom for the cold-open silence.
