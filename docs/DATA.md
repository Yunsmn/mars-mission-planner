# Space data plan

The challenge theme is *"data-heavy → insight-driven."* So MARVIN fuses **three real
orbital/mission data streams** into the **three inputs a planner actually needs** —
traversability, science value, and energy. This is the concrete "insight, not data" story.

| Planner input | Data stream | Source | Role |
|---|---|---|---|
| **Traversability** | HiRISE/CTX **DEM** + **orthoimage (DRG)** | GALE-Lab Mars DEMs → NASA PDS / USGS Astrogeology | elevation → MuJoCo `hfield`; slope/roughness → hazard + energy cost; ortho → target candidates + backdrop |
| **Science value** | **CRISM MTRDR** mineral maps | MRO CRISM, PDS Geosciences | carbonate / Fe-Mg phyllosilicate / olivine strength → a **data-derived** science score per target (not a made-up number) |
| **Energy / power** | **Dust opacity (τ)** time series | MEDA (Perseverance) / REMS (Curiosity), PDS Atmospheres | drives a solar-power budget; a dust event = the dynamic "conditions changed → replan" beat |

Why these three (and why it matters): each is a **different, real** dataset, and together
they touch every arm of the decision — *can I get there safely?* (DEM), *is it worth it?*
(CRISM), *can I afford it?* (dust/power). That breadth is what separates an "insight-driven"
entry from a terrain-only demo.

## 1. Terrain — HiRISE/CTX DEM + ortho (Jezero)

- **Site:** Jezero crater (Perseverance's real terrain → narrative payoff; also the region
  with the richest CRISM mineralogy).
- **Access:** GALE-Lab archive `github.com/GALE-Lab/Mars_DEMs` (footprint shapefile + list)
  → files on NASA PDS / USGS Astrogeology / S3. Each HiRISE DEM ships with its ortho (DRG).
- **Format:** GeoTIFF. HiRISE = 1 m/px (big), CTX = ~6 m/px (lighter).
- **Disk note (`/home` is tight, ~8 GB free):** prefer a **CTX-scale** tile or **downsample**
  a HiRISE tile to ~6 m/px and crop to the working area. Commit only a small sample under
  `data/samples/`; keep full rasters in `data/raw/` (git-ignored).
- **Derived (built in `world/dem.py`, `data/prepare.py`):**
  - `jezero_dem.npy` — elevation grid → MuJoCo `hfield`.
  - `jezero_slope.npy` — gradient magnitude (deg) → hazard/energy.
  - `jezero_ortho.png` — orbital image for targets + visuals.

## 2. Science value — CRISM MTRDR mineralogy

- **What:** CRISM Map-projected Targeted Reduced Data Records over Jezero — atmospheric- and
  noise-corrected mineral maps. Jezero is carbonate/phyllosilicate-rich (astrobiology-grade).
- **Access:** PDS Geosciences node (CRISM MTRDR / summary-parameter browse products).
- **Use:** align a mineral-strength raster to the DEM extent → `jezero_science_value.npy`,
  a per-cell score (e.g. carbonate + clay favored). `data/science_value.py` maps a target's
  location → its real, data-driven science value. **This replaces arbitrary target values.**
- **Simplification allowed:** if full MTRDR processing is too heavy for the timeline, use a
  published summary-parameter map or a derived mineral index; document the exact product used.

## 3. Energy — dust opacity (τ) time series

- **What:** measured atmospheric optical depth over time (dust). Higher τ → less sunlight →
  less solar charge; a dust storm sharply cuts power.
- **Access:** PDS Atmospheres node — MEDA (Perseverance) or REMS (Curiosity) products.
- **Use:** `data/dust.py` loads τ(t) → a `power_factor(τ)` the energy model applies to solar
  charge (`config.yaml: power`). A scripted τ spike drives the demo's "environment changed →
  replan / defer" moment. *(We frame the rover as solar-powered so dust→power is physical;
  Perseverance itself is nuclear, so this is a generic future-rover assumption — stated openly.)*

## Provenance & licensing

- NASA PDS / USGS data are public domain (cite the products + missions).
- GALE-Lab DEMs are released for public use (cite the archive).
- Record exact product IDs + download dates in `data/raw/PROVENANCE.md` when fetched.

## Fetch workflow (Phase 0)

1. Pick a Jezero tile from the GALE-Lab footprint list (has DEM + ortho + CRISM overlap).
2. Download raw rasters to `data/raw/` (git-ignored).
3. `data/prepare.py`: reproject/crop/downsample to a common grid; write the `derived/` arrays
   referenced in `config.yaml`.
4. Commit only the small `data/samples/` tile + `PROVENANCE.md`.

> Optional 4th stream (stretch): classify the ortho image (sand vs. bedrock) → per-terrain
> traction in the surrogate. Higher payoff-to-effort is lower than the three above; defer.
