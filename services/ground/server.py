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
MISSION_WORDS = ("collect", "sample", "acquire", "get ", "go to", "drive", "cache", "retrieve", "image")

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


def get_rover() -> Rover:
    global _rover
    if _rover is None:
        from model.propose import Proposer
        m = _cfg0["model"]
        _rover = Rover(_cfg, Proposer(m["name"], m["host"], m["temperature"]))
    return _rover


async def broadcast(msg: dict) -> None:
    for ws in list(clients):
        try:
            await ws.send_json(msg)
        except Exception:
            clients.discard(ws)


def _wire(kind: str, payload) -> dict:
    if kind in ("deviation", "panel", "telemetry", "summary"):
        return {"type": kind, "payload": payload}
    return {"type": "log", "text": payload if isinstance(payload, str) else str(payload)}


async def _chip(who: str, state: str):
    await broadcast({"type": "chip", "who": who, "state": state})


async def handle(text: str) -> None:
    await broadcast({"type": "chat", "who": "operator", "text": text})
    await _chip("houston", "thinking")
    r = await asyncio.to_thread(houston.route, text, DELAY)
    await _chip("houston", "speaking")
    await broadcast({"type": "chat", "who": "houston", "text": r["reply"]})
    await asyncio.sleep(0.3)
    if r["target"] != "marvin":
        await _chip("houston", "idle")
        return
    async with _channel:                         # one rover, one mission at a time
        if any(w in text.lower() for w in MISSION_WORDS):
            await run_mission(text)
        else:
            await run_question(text)
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
    await link.uplink(brief, "briefing")         # Earth -> Mars light-time
    await _chip("houston", "idle")
    await _chip("marvin", "thinking")

    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    rover = get_rover()
    rover.reset()

    def worker():
        try:
            for ev in rover.run_briefing(brief):
                asyncio.run_coroutine_threadsafe(q.put(ev), loop)
        finally:
            asyncio.run_coroutine_threadsafe(q.put(None), loop)

    loop.run_in_executor(None, worker)
    await broadcast({"type": "comms", "dir": "downlink", "seconds": DELAY})
    first = True
    while True:
        ev = await q.get()
        if ev is None:
            break
        if first:
            await asyncio.sleep(DELAY)            # Mars -> Earth light-time, once, then stream
            first = False
        await broadcast(_wire(*ev))
    await _chip("marvin", "idle")


async def run_question(text: str) -> None:
    await broadcast({"type": "comms", "dir": "uplink", "seconds": DELAY})
    await _chip("marvin", "transmitting")
    await link.uplink({"q": text}, "message")
    await _chip("marvin", "thinking")
    ans = await asyncio.to_thread(get_rover().answer, text)
    await broadcast({"type": "comms", "dir": "downlink", "seconds": DELAY})
    await asyncio.sleep(DELAY)
    await _chip("marvin", "speaking")
    await broadcast({"type": "chat", "who": "marvin", "text": ans})
    await _chip("marvin", "idle")


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
    except WebSocketDisconnect:
        clients.discard(websocket)


@app.get("/health")
async def health():
    return {"ok": True, "houston_backend": houston.backend_label, "delay_s": DELAY}


@app.get("/")
async def index():
    return FileResponse(_CONSOLE / "index.html")


app.mount("/static", StaticFiles(directory=str(_CONSOLE)), name="static")
