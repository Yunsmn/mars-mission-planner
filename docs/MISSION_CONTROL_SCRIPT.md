# Mission Control — 3-minute video script (two Granites, one link)

**Target 2:55.** Build only what appears on screen (spec's scope order). The console *is* the diagram:
HOUSTON left, MARVIN right, a terminal between them, the light-time link in the middle.

**Verified core beat (already running headless via `python -m services.flow_demo`):** operator intent
→ HOUSTON routes + briefs → uplink pays the delay → MARVIN deviates from Earth's orbital route → sample.

---

### 0:00 – 0:20 · The two segments
**Visual:** the console. Two mission-patch chips — a dish (HOUSTON, left), a rover (MARVIN, right) —
a terminal between them, a comms bar: `LINK ▸ 12s (compressed; real Mars one-way: 4–24 min)`.
**VO:**
> "Two AIs run this mission. On Earth: HOUSTON — a cloud Granite model, the strategist. On Mars:
> MARVIN — a small Granite model on the rover itself, the tactician. They're split by the one thing
> you can't engineer away: the speed of light."

### 0:20 – 0:55 · One sentence becomes a mission
**Visual:** operator types *"get me a sample from the carbonate outcrop east of the dunes."* HOUSTON's
chip pulses; a **briefing card** unfolds — objectives, science rationale, constraints, and a
`route_advisory: direct approach appears shortest (orbital)`.
**VO:**
> "The operator says one sentence. HOUSTON — instant, both on Earth — expands it into a full mission
> briefing: objectives, priorities, the science case, and a route suggestion from orbital data. Then
> it hands it to the rover."
**On screen:** *"Copy. Relaying to MARVIN — uplink window in 12s."*

### 0:55 – 1:20 · The link
**Visual:** a signal dot travels left→right across the gap; HOUSTON's chip dims; the comms bar counts
down. On arrival, MARVIN's chip starts its thinking dots.
**VO:**
> "Nothing reaches Mars for free. The briefing crosses the link and pays the light-time. Now the
> rover is on its own — no Earth in the loop until it reports back."

### 1:20 – 2:05 · The rover knows better (the deviation)
**Visual:** the **ops map** panel: DEM heatmap, the rover, A* candidate routes colored by entrapment
risk. The direct route lights **red — 100%**. The decision log streams Granite's reasoning. A
**deviation report** downlinks.
**VO:**
> "MARVIN's own sensors build a cost map Earth never had. Its navigator lays down candidate routes;
> the onboard simulator rolls each out twenty times. The route Earth suggested — straight across —
> comes back at a hundred percent entrapment. So MARVIN overrules it."
**On screen (downlink):** *"Negative on the direct approach: soft-sand ridge, 100% entrapment across
20 rollouts. Taking the balanced route — 9 m at 0%."* The rover drives the arc; a hazcam still arrives;
sample cached.

### 2:05 – 2:30 · Why this is the architecture
**Visual:** split — HOUSTON (cloud, briefings) | link | MARVIN (edge, execution). Caption: *the LLM
decides, the gate approves, the controller drives.*
**VO:**
> "The big model plans; the small model acts; neither one ever touches the wheels directly — a
> verified gate does. This isn't a product choice. It's the only shape that fits when your two halves
> are minutes apart. And it's the same shape terrestrial robotics needs today: cloud planner, edge
> executor."

### 2:30 – 2:55 · Close
**Visual:** the map, the cached sample, repo + live URL.
**VO:**
> "Earth planned from orbit. The rover routed on what it sensed. Zero sols lost — Opportunity's
> Purgatory Dune cost thirty-eight. Two Granites, one link, decisions made where they have to be made."

---

**Recording notes**
- The relay → signal → dots handoff is the one animation worth polishing — it explains the whole
  architecture in two seconds.
- Pre-warm both Granite backends before recording; the on-screen "thinking" should read as
  deliberation, not load time.
- If watsonx latency or keys aren't ready, HOUSTON runs on the local fallback — same beat, footer
  notes the backend honestly.
