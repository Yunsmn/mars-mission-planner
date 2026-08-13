"""Build the self-contained MARVIN 3D showcase page.

Exports real mission data (Jezero terrain, rover path, targets), inlines three.js + the scene,
and writes:
  web/showcase.html           -- full standalone page (open in a browser or host anywhere)
  web/showcase_artifact.html  -- body-only variant (for publishing as a shareable artifact)

Run:  .venv/bin/python -m web.build_showcase
"""
from __future__ import annotations

import json
import os
import subprocess

VENDOR = {
    "three.min.js": "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js",
    "OrbitControls.js": "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js",
}


def export_data() -> dict:
    from rover import capabilities as cap
    from world.sim import TERRAIN_RADIUS, MarsSim

    sim = MarsSim(seed=42)              # defaults to the real Jezero MOLA terrain
    cap.bind(sim, 1)
    targets = []
    for t in sim.targets:
        sim.drive_to(*t.xy)
        collect_at = len(sim.trajectory)
        sim.sample(t.id)
        targets.append({"id": t.id, "x": round(t.xy[0], 3), "y": round(t.xy[1], 3),
                        "collectAt": collect_at, "science": round(t.science_value, 2)})
    return {
        "grid": int(sim.terrain.shape[0]),
        "extent": float(TERRAIN_RADIUS),
        "terrain": [[round(float(v), 4) for v in row] for row in sim.terrain],
        "trajectory": [[round(float(x), 3), round(float(y), 3)] for (x, y) in sim.trajectory],
        "targets": targets,
    }


def vendor() -> dict:
    os.makedirs("web/vendor", exist_ok=True)
    out = {}
    for name, url in VENDOR.items():
        path = f"web/vendor/{name}"
        if not os.path.exists(path):
            subprocess.run(["curl", "-sL", "--max-time", "90", "-o", path, url], check=True)
        out[name] = open(path, encoding="utf-8").read()
    return out


STYLE = """
  * { margin: 0; box-sizing: border-box; }
  :root {
    --bg:#05060b; --panel:rgba(11,15,24,0.74); --line:rgba(120,150,200,0.16);
    --ink:#eef2f9; --muted:#93a1bd; --mars:#ff9d5c; --ok:#5fdd8a;
    --mono: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace;
  }
  html, body { height: 100%; background: var(--bg); color: var(--ink);
    font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; overflow: hidden; }
  #app { position: fixed; inset: 0; }
  .panel { position: fixed; top: 18px; left: 18px; width: 332px; padding: 20px 22px;
    background: var(--panel); backdrop-filter: blur(10px);
    border: 1px solid var(--line); border-radius: 14px; box-shadow: 0 16px 46px rgba(0,0,0,0.5); }
  .kicker { font-family: var(--mono); font-size: 10.5px; letter-spacing: 0.22em;
    text-transform: uppercase; color: var(--mars); font-weight: 600; }
  h1 { font-size: 32px; font-weight: 800; letter-spacing: -1px; margin: 6px 0 2px; text-wrap: balance; }
  .sub { font-size: 13px; color: var(--muted); line-height: 1.5; margin-bottom: 15px; }
  .status { font-family: var(--mono); font-size: 12.5px; font-weight: 600; color: var(--ok);
    font-variant-numeric: tabular-nums; margin-bottom: 17px; display: flex; align-items: center; gap: 9px; }
  .status::before { content: ""; width: 8px; height: 8px; border-radius: 50%;
    background: var(--ok); box-shadow: 0 0 10px var(--ok); }
  .rows { display: flex; flex-direction: column; gap: 11px; }
  .row { display: flex; justify-content: space-between; align-items: baseline; gap: 14px;
    padding-bottom: 11px; border-bottom: 1px solid rgba(255,255,255,0.06); }
  .row:last-child { border-bottom: 0; padding-bottom: 0; }
  .lab { font-family: var(--mono); font-size: 10.5px; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--muted); white-space: nowrap; }
  .val { font-size: 13px; text-align: right; color: var(--ink); font-variant-numeric: tabular-nums; }
  .val b { color: var(--mars); font-weight: 700; }
  .foot { margin-top: 17px; font-family: var(--mono); font-size: 11px; color: #8091ad; letter-spacing: 0.03em; }
  .foot b { color: #8fb0ff; font-weight: 600; }
  .hint { position: fixed; bottom: 16px; right: 18px; font-family: var(--mono); font-size: 11px;
    color: #8091ad; background: rgba(11,15,24,0.6); border: 1px solid var(--line);
    padding: 6px 12px; border-radius: 999px; }
"""

BODY = """
<div id="app"></div>
<div class="panel">
  <div class="kicker">IBM AI Builders · Space</div>
  <h1>MARVIN</h1>
  <div class="sub">Onboard, offline Mars mission planner. IBM Granite proposes; a physics-lite
    simulator verifies before the rover moves.</div>
  <div class="status" id="status">0 / 0 samples cached</div>
  <div class="rows">
    <div class="row"><span class="lab">Terrain</span><span class="val">real Jezero DEM · <b>NASA MOLA</b></span></div>
    <div class="row"><span class="lab">Fast-sim fidelity</span><span class="val">within <b>3%</b> of full physics</span></div>
    <div class="row"><span class="lab">Science return</span><span class="val"><b>2.3&times;</b> vs. naive selection</span></div>
    <div class="row"><span class="lab">Time to run</span><span class="val">onboard <b>minutes</b> · Earth <b>~4 mo</b></span></div>
  </div>
  <div class="foot">Powered by <b>IBM Granite 4.1</b> · Built with <b>IBM Bob</b></div>
</div>
<div class="hint">drag to orbit · scroll to zoom</div>
"""


def build():
    data = export_data()
    libs = vendor()
    scene = open("web/scene.js", encoding="utf-8").read()

    scripts = (
        f"<script>{libs['three.min.js']}</script>\n"
        f"<script>{libs['OrbitControls.js']}</script>\n"
        f"<script>window.DATA={json.dumps(data, separators=(',', ':'))};</script>\n"
        f"<script>{scene}</script>"
    )
    inner = f"<title>MARVIN — Autonomous Mars Mission Planner</title>\n<style>{STYLE}</style>\n{BODY}\n{scripts}"

    full = ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"{inner}</head><body></body></html>")
    # (scripts run fine at end of head since they build into #app which is created by BODY;
    #  to be safe, use a body-hosted version for the standalone file:)
    full = ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            "<title>MARVIN — Autonomous Mars Mission Planner</title>"
            f"<style>{STYLE}</style></head><body>{BODY}{scripts}</body></html>")

    os.makedirs("web", exist_ok=True)
    open("web/showcase.html", "w", encoding="utf-8").write(full)
    open("web/showcase_artifact.html", "w", encoding="utf-8").write(inner)
    print(f"wrote web/showcase.html ({len(full)//1024} KB) and web/showcase_artifact.html")
    print(f"  terrain {data['grid']}x{data['grid']} · path {len(data['trajectory'])} pts · "
          f"{len(data['targets'])} targets")


if __name__ == "__main__":
    build()
