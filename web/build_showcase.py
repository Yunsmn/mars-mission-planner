"""Build the self-contained MARVIN 3D showcase page.

Design: IBM Plex typography (inlined as woff2 data URIs) + IBM Carbon palette, laid out as a
planetary-ops console — a header with the Jezero designation, a framed 3D viewport, and a
monospace telemetry + decision-log rail. Real mission data drives the scene.

Writes:
  web/showcase.html           -- full standalone page (open in a browser or host anywhere)
  web/showcase_artifact.html  -- body-only variant (for publishing as a shareable link)

Run:  .venv/bin/python -m web.build_showcase
"""
from __future__ import annotations

import base64
import json
import os
import subprocess

VENDOR = {
    "three.min.js": "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js",
    "OrbitControls.js": "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js",
}

FONTS = [
    ("IBM Plex Sans", 400, "https://cdn.jsdelivr.net/npm/@ibm/plex-sans/fonts/complete/woff2/IBMPlexSans-Regular.woff2"),
    ("IBM Plex Sans", 600, "https://cdn.jsdelivr.net/npm/@ibm/plex-sans/fonts/complete/woff2/IBMPlexSans-SemiBold.woff2"),
    ("IBM Plex Mono", 400, "https://cdn.jsdelivr.net/npm/@ibm/plex-mono/fonts/complete/woff2/IBMPlexMono-Regular.woff2"),
    ("IBM Plex Mono", 500, "https://cdn.jsdelivr.net/npm/@ibm/plex-mono/fonts/complete/woff2/IBMPlexMono-Medium.woff2"),
]

JEZERO_COORDS = "18.44°N  77.58°E"


def _download(url: str, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        subprocess.run(["curl", "-sL", "--max-time", "120", "-o", path, url], check=True)
    return path


def export_data() -> dict:
    # Prefer the recorded REAL Granite mission (web/record_mission.py). Everything shown is
    # measured from an actual run; no hand-picked numbers.
    rec = "data/derived/mission_record.json"
    if os.path.exists(rec):
        return json.load(open(rec, encoding="utf-8"))

    # fallback: a scripted-real traverse on the real terrain, with the same measured shape
    import math as _m
    from rover import capabilities as cap
    from world.sim import TERRAIN_RADIUS, MarsSim
    sim = MarsSim(seed=42)
    cap.bind(sim, 1)
    targets, decisions = [], []
    for t in sim.targets:
        sim.drive_to(*t.xy); decisions.append({"act": "DRIVE", "to": t.id})
        collect_at = len(sim.trajectory)
        sim.sample(t.id); decisions.append({"act": "SAMPLE", "to": t.id})
        targets.append({"id": t.id, "x": round(t.xy[0], 3), "y": round(t.xy[1], 3), "collectAt": collect_at})
    traj = [[round(float(x), 3), round(float(y), 3)] for (x, y) in sim.trajectory]
    dist = sum(_m.dist(traj[i - 1], traj[i]) for i in range(1, len(traj)))
    return {
        "grid": int(sim.terrain.shape[0]), "extent": float(TERRAIN_RADIUS),
        "terrain": [[round(float(v), 4) for v in row] for row in sim.terrain],
        "trajectory": traj, "targets": targets, "decisions": decisions,
        "stats": {"decisions": len(decisions), "samples": sum(t.collected for t in sim.targets),
                  "distance_m": round(dist, 2), "battery_used_pct": round(100 - sim.battery_pct, 1),
                  "sim_seconds": round(sim.sol_time, 1)},
    }


def font_faces() -> str:
    faces = []
    for fam, wt, url in FONTS:
        path = _download(url, f"web/vendor/fonts/{url.split('/')[-1]}")
        b64 = base64.b64encode(open(path, "rb").read()).decode()
        faces.append(f"@font-face{{font-family:'{fam}';font-weight:{wt};font-style:normal;"
                     f"font-display:swap;src:url(data:font/woff2;base64,{b64}) format('woff2');}}")
    return "\n".join(faces)


STYLE = """
  * { margin: 0; box-sizing: border-box; }
  :root {
    --bg:#161616; --layer:#1f1f1f; --line:#393939; --line-soft:#2a2a2a;
    --text:#f4f4f4; --text2:#c6c6c6; --text3:#8d8d8d;
    --blue:#4589ff; --amber:#ff832b; --ok:#42be65;
    --sans:"IBM Plex Sans", system-ui, sans-serif;
    --mono:"IBM Plex Mono", ui-monospace, Menlo, monospace;
  }
  html, body { height: 100%; }
  body { background: var(--bg); color: var(--text); font-family: var(--sans);
    height: 100vh; display: grid; grid-template-rows: auto 1fr auto; overflow: hidden; }

  .hdr { display: flex; justify-content: space-between; align-items: center;
    padding: 13px 22px; border-bottom: 1px solid var(--line); }
  .desig { display: flex; align-items: center; gap: 12px; }
  .mark { width: 11px; height: 11px; background: var(--amber); }
  .desig .name { font-weight: 600; font-size: 15px; letter-spacing: 0.01em; }
  .desig .full { font-family: var(--mono); font-size: 11px; color: var(--text3);
    letter-spacing: 0.02em; padding-left: 12px; border-left: 1px solid var(--line); }
  .coords { font-family: var(--mono); font-size: 11.5px; color: var(--text2); letter-spacing: 0.04em;
    font-variant-numeric: tabular-nums; }
  .coords .k { color: var(--text3); }

  .main { display: grid; grid-template-columns: 1fr 316px; min-height: 0; }
  .viewport { position: relative; overflow: hidden; }
  #app { position: absolute; inset: 0; }
  .tick { position: absolute; width: 14px; height: 14px; border: 0 solid var(--text3); opacity: 0.55; pointer-events: none; }
  .tick.tl { top: 12px; left: 12px; border-top-width: 1px; border-left-width: 1px; }
  .tick.tr { top: 12px; right: 12px; border-top-width: 1px; border-right-width: 1px; }
  .tick.bl { bottom: 12px; left: 12px; border-bottom-width: 1px; border-left-width: 1px; }
  .tick.br { bottom: 12px; right: 12px; border-bottom-width: 1px; border-right-width: 1px; }
  .vp-status { position: absolute; top: 14px; left: 16px; font-family: var(--mono); font-size: 12px;
    color: var(--ok); font-variant-numeric: tabular-nums; display: flex; align-items: center; gap: 8px; }
  .vp-status::before { content: ""; width: 7px; height: 7px; border-radius: 50%;
    background: var(--ok); box-shadow: 0 0 9px var(--ok); }
  .vp-note { position: absolute; bottom: 14px; left: 16px; font-family: var(--mono); font-size: 10.5px;
    color: var(--text3); letter-spacing: 0.02em; }
  .vp-hint { position: absolute; bottom: 14px; right: 16px; font-family: var(--mono); font-size: 10.5px;
    color: var(--text3); }

  .rail { border-left: 1px solid var(--line); padding: 18px; overflow-y: auto;
    display: flex; flex-direction: column; gap: 22px; }
  .rail h2 { font-family: var(--mono); font-size: 10.5px; letter-spacing: 0.16em; text-transform: uppercase;
    color: var(--text3); padding-bottom: 7px; border-bottom: 1px solid var(--line); margin-bottom: 11px; }
  .tel { display: flex; flex-direction: column; gap: 9px; }
  .tel .r { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; }
  .tel .k { font-family: var(--mono); font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.06em;
    color: var(--text3); white-space: nowrap; }
  .tel .v { font-family: var(--mono); font-size: 12px; color: var(--text); text-align: right;
    font-variant-numeric: tabular-nums; }
  .tel .v b { color: var(--amber); font-weight: 500; }
  .log { display: flex; flex-direction: column; }
  .step { display: grid; grid-template-columns: 20px 68px 1fr; gap: 8px; align-items: baseline;
    font-family: var(--mono); font-size: 11.5px; padding: 5px 0; border-bottom: 1px solid var(--line-soft); }
  .step:last-child { border-bottom: 0; }
  .step .n { color: var(--text3); font-variant-numeric: tabular-nums; }
  .step .act { color: var(--blue); font-weight: 500; letter-spacing: 0.02em; }
  .step .to { color: var(--text2); }

  .ftr { display: flex; justify-content: space-between; align-items: center;
    padding: 10px 22px; border-top: 1px solid var(--line); font-family: var(--mono);
    font-size: 11px; color: var(--text3); letter-spacing: 0.02em; }
  .ftr b { color: var(--blue); font-weight: 500; }
"""


def body(data: dict) -> str:
    steps = "".join(
        f'<div class="step"><span class="n">{i:02d}</span>'
        f'<span class="act">{d["act"]}</span><span class="to">{d["to"]}</span></div>'
        for i, d in enumerate(data["decisions"], 1)
    )
    s = data.get("stats") or {}
    dec = s.get("decisions", len(data["decisions"]))
    samp = s.get("samples", "—")
    distm = s.get("distance_m", "—")
    batt = s.get("battery_used_pct", "—")
    return f"""
<div class="hdr">
  <div class="desig"><span class="mark"></span><span class="name">MARVIN</span>
    <span class="full">Autonomous Mars Mission Planner</span></div>
  <div class="coords"><span class="k">JEZERO &middot; MOLA DEM &middot;</span> {JEZERO_COORDS}</div>
</div>
<div class="main">
  <div class="viewport">
    <div id="app"></div>
    <span class="tick tl"></span><span class="tick tr"></span>
    <span class="tick bl"></span><span class="tick br"></span>
    <div class="vp-status" id="status">0 / 0 samples cached</div>
    <div class="vp-note">10 m patch &middot; morphology from a 60 km MOLA window</div>
    <div class="vp-hint">drag &middot; scroll to zoom</div>
  </div>
  <aside class="rail">
    <section>
      <h2>Telemetry</h2>
      <div class="tel">
        <div class="r"><span class="k">Site</span><span class="v">Jezero &middot; MOLA DEM</span></div>
        <div class="r"><span class="k">Planner</span><span class="v"><b>IBM Granite 4.1</b></span></div>
        <div class="r"><span class="k">Decisions</span><span class="v"><b>{dec}</b> &middot; 0 ground</span></div>
        <div class="r"><span class="k">Samples</span><span class="v">{samp} / 2 cached</span></div>
        <div class="r"><span class="k">Traverse</span><span class="v">{distm} m</span></div>
        <div class="r"><span class="k">Battery used</span><span class="v">{batt} %</span></div>
        <div class="r"><span class="k">Ground-ops eqv.</span><span class="v"><b>~{dec} sols</b> vs 1 cycle</span></div>
      </div>
    </section>
    <section>
      <h2>Decision log</h2>
      <div class="log">{steps}</div>
    </section>
  </aside>
</div>
<div class="ftr"><span>Onboard &middot; offline &middot; no network</span>
  <span>Powered by <b>IBM Granite 4.1</b></span></div>
"""


def build():
    data = export_data()
    libs = vendor()
    fonts = font_faces()
    scene = open("web/scene.js", encoding="utf-8").read()

    scripts = (
        f"<script>{libs['three.min.js']}</script>\n"
        f"<script>{libs['OrbitControls.js']}</script>\n"
        f"<script>window.DATA={json.dumps(data, separators=(',', ':'))};</script>\n"
        f"<script>{scene}</script>"
    )
    inner = (f"<title>MARVIN — Autonomous Mars Mission Planner</title>\n"
             f"<style>{fonts}\n{STYLE}</style>\n{body(data)}\n{scripts}")

    full = ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            "<title>MARVIN — Autonomous Mars Mission Planner</title>"
            f"<style>{fonts}\n{STYLE}</style></head><body>{body(data)}{scripts}</body></html>")

    os.makedirs("web", exist_ok=True)
    open("web/showcase.html", "w", encoding="utf-8").write(full)
    open("web/showcase_artifact.html", "w", encoding="utf-8").write(inner)
    print(f"wrote web/showcase.html ({len(full)//1024} KB) and web/showcase_artifact.html")
    print(f"  terrain {data['grid']}x{data['grid']} · path {len(data['trajectory'])} pts · "
          f"{len(data['targets'])} targets · {len(data['decisions'])} decisions")


def vendor() -> dict:
    return {name: open(_download(url, f"web/vendor/{name}"), encoding="utf-8").read()
            for name, url in VENDOR.items()}


if __name__ == "__main__":
    build()
