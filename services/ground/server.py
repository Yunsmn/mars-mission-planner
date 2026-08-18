"""HOUSTON ground service — serves the Mission Control console and orchestrates the whole loop:

    operator (WS) -> HOUSTON routes -> [HOUSTON answers]  OR  [brief -> uplink(delay) -> MARVIN runs
    the existing planner -> downlink(delay) products] -> console (WS broadcast to everyone watching).

Every Earth<->Mars crossing goes through services.ground.link (the delay lives there, nowhere else).
Blocking work (Granite calls, MuJoCo driving) runs in threads so the event loop stays responsive.
One rover, one shared world; missions serialize on a lock (the channel is single).

Run:  .venv/bin/python -m uvicorn services.ground.server:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import itertools
import os
import pathlib

import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from services.ground.houston import Houston
from services.ground.link import DeepSpaceLink
from services.rover.intent_api import Rover

DELAY = float(os.getenv("LINK_DELAY_S", "12"))
SCENARIO = ("Known target: carbonate outcrop at (0.0, 3.0), high science value. Surrounding terrain "
            "is rolling, mapped only by orbital DEM (no view of local soft sand).")

_CONSOLE = pathlib.Path(__file__).resolve().parents[2] / "console"
_cfg0 = yaml.safe_load(open("config.yaml"))
_cfg = {**_cfg0["planner"], **_cfg0["constraints"], "energy_penalty_factor": 0.01}

app = FastAPI(title="MARVIN Mission Control")
houston = Houston()
link = DeepSpaceLink(delay_s=DELAY)
clients: set[WebSocket] = set()
_mission_no = itertools.count(447)
_channel = asyncio.Lock()
_rover: Rover | None = None

# ALL sim + MuJoCo-render work runs in ONE dedicated thread. The GL context is created and used in
# that same thread (no cross-thread OpenGL = no white-noise frames), and every sim access serialises
# through it (no data races between stepping and rendering).
_sim_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="sim")


async def sim_call(fn, *args):
    """Run a sim/render operation on the dedicated sim thread."""
    return await asyncio.get_running_loop().run_in_executor(_sim_pool, fn, *args)


def _build_rover() -> Rover:
    from model.propose import Proposer
    m = _cfg0["model"]
    return Rover(_cfg, Proposer(m["name"], m["host"], m["temperature"]))


def get_rover() -> Rover:
    return _rover


RENDER_DT = 0.5                                  # idle live-view refresh (seconds)


async def _render_loop():
    """Always-on: stream the rover's current view whenever nothing else is driving. During a mission
    the sim thread is busy driving (and yields its own frames), so this naturally pauses until it ends."""
    while True:
        await asyncio.sleep(RENDER_DT)
        if _rover is None or not clients:
            continue
        try:
            frame = await sim_call(_rover.frame)
            if frame:
                await broadcast({"type": "frame", "data": frame})
        except Exception:
            pass


@app.on_event("startup")
async def _startup():
    global _rover
    _rover = await sim_call(_build_rover)         # sim + GL renderer created ON the sim thread
    asyncio.create_task(_render_loop())


async def broadcast(msg: dict) -> None:
    for ws in list(clients):
        try:
            await ws.send_json(msg)
        except Exception:
            clients.discard(ws)


def _wire(kind: str, payload) -> dict:
    if kind == "say":
        return {"type": "chat", "who": "marvin", "text": payload}
    if kind == "frame":
        return {"type": "frame", "data": payload}
    if kind in ("deviation", "panel", "telemetry", "summary"):
        return {"type": kind, "payload": payload}
    return {"type": "log", "text": payload if isinstance(payload, str) else str(payload)}


async def _chip(who: str, state: str):
    await broadcast({"type": "chip", "who": who, "state": state})


async def _stream_downlink(gen) -> None:
    """Run a blocking rover generator in a thread and downlink its events — the Mars->Earth light-time
    is paid once, then the products stream (the rover already produced them; the signal is in flight)."""
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()

    def worker():
        try:
            for ev in gen:
                asyncio.run_coroutine_threadsafe(q.put(ev), loop)
        finally:
            asyncio.run_coroutine_threadsafe(q.put(None), loop)

    loop.run_in_executor(_sim_pool, worker)      # driving + rendering on the single sim thread
    await broadcast({"type": "comms", "dir": "downlink", "seconds": DELAY})
    first = True
    while True:
        ev = await q.get()
        if ev is None:
            break
        if first:
            await asyncio.sleep(DELAY)
            first = False
        await broadcast(_wire(*ev))
    await _chip("marvin", "idle")


async def handle(text: str) -> None:
    await broadcast({"type": "chat", "who": "operator", "text": text})
    if text.lower().strip() in ("reset", "restart", "reset world"):
        async with _channel:                      # don't reset mid-drive
            await sim_call(get_rover().reset)
        await broadcast({"type": "chat", "who": "houston", "text": "World reset. Rover at the start line."})
        await broadcast({"type": "reset"})
        return
    await _chip("houston", "thinking")
    r = await asyncio.to_thread(houston.route, text, DELAY)
    await _chip("houston", "speaking")
    if r.get("panel") == "weather":               # weather = on-site conditions -> ASK MARVIN + panel
        from services.ground.nasa_feeds import mars_weather
        ack = r["reply"] if r["target"] != "answer" else \
            f"Relaying to MARVIN for on-site conditions — uplink window in {int(DELAY)}s."
        await broadcast({"type": "chat", "who": "houston", "text": ack})
        async with _channel:
            w = await asyncio.to_thread(mars_weather, get_rover().sim.dust_tau)
            await broadcast({"type": "weather", "payload": w})   # local reading + regional feed
            await run_command(text)                              # MARVIN reports what its sensors read
        await _chip("houston", "idle")
        return
    await broadcast({"type": "chat", "who": "houston", "text": r["reply"]})
    await asyncio.sleep(0.3)
    if r["target"] == "answer":                   # HOUSTON handled it on Earth
        await _chip("houston", "idle")
        return
    async with _channel:                          # one rover, one transmission at a time
        if r["target"] == "brief":                # a science objective -> HOUSTON briefs, MARVIN runs it
            await run_mission(text)
        else:                                     # relay -> MARVIN's own conversation replies or acts
            await run_command(text)
    await _chip("houston", "idle")


async def run_mission(text: str) -> None:
    mid = f"sol-{next(_mission_no):04d}-a"
    await _chip("houston", "thinking")
    brief = await asyncio.to_thread(houston.brief, text, SCENARIO, mid, DELAY)
    if not brief:
        await broadcast({"type": "chat", "who": "houston",
                         "text": "Ground segment degraded — retry your last transmission."})
        return
    await broadcast({"type": "briefing", "briefing": brief})
    await _chip("houston", "relaying")
    await broadcast({"type": "comms", "dir": "uplink", "seconds": DELAY})
    await _chip("marvin", "transmitting")
    await link.uplink(brief, "briefing")
    await _chip("houston", "idle")
    await _chip("marvin", "thinking")
    await _stream_downlink(get_rover().run_briefing(brief))


async def run_command(text: str) -> None:
    """Relay anything to MARVIN and let its own conversation decide: reply, or act and stream it."""
    await broadcast({"type": "comms", "dir": "uplink", "seconds": DELAY})
    await _chip("marvin", "transmitting")
    await link.uplink({"msg": text}, "message")
    await _chip("marvin", "thinking")
    await _stream_downlink(get_rover().run_command(text))


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    await websocket.send_json({"type": "hello", "delay_s": DELAY,
                               "houston_backend": houston.backend_label})
    try:
        while True:
            data = await websocket.receive_json()
            if data.get("type") == "operator" and str(data.get("text", "")).strip():
                asyncio.create_task(handle(str(data["text"]).strip()))
            elif data.get("type") == "camera":
                get_rover().set_camera(str(data.get("mode", "satellite")))
    except WebSocketDisconnect:
        clients.discard(websocket)


@app.get("/health")
async def health():
    return {"ok": True, "houston_backend": houston.backend_label, "delay_s": DELAY}


@app.get("/")
async def index():
    return FileResponse(_CONSOLE / "index.html")


app.mount("/static", StaticFiles(directory=str(_CONSOLE)), name="static")
