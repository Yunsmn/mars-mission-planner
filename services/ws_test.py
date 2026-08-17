"""Headless websocket client — drives the console's server the way the browser would, so the full
orchestration is verifiable without a display. Run after starting the server."""
import asyncio
import json

import websockets

INTENT = "get me a sample from the carbonate outcrop east of the dunes"


async def main():
    async with websockets.connect("ws://127.0.0.1:8000/ws", max_size=4_000_000) as ws:
        hello = json.loads(await ws.recv())
        print("HELLO backend:", hello.get("houston_backend"), "delay", hello.get("delay_s"))
        await ws.send(json.dumps({"type": "operator", "text": INTENT}))
        seen = {}
        while True:
            m = json.loads(await asyncio.wait_for(ws.recv(), timeout=240))
            t = m["type"]; seen[t] = seen.get(t, 0) + 1
            if t == "chat": print(f"CHAT[{m['who']}]: {m['text'][:90]}")
            elif t == "briefing": print("BRIEFING:", m["briefing"]["objective_summary"],
                                        "| advisory:", m["briefing"]["route_advisory"]["note"])
            elif t == "comms": print(f"COMMS {m['dir']} {m['seconds']}s")
            elif t == "chip": pass
            elif t == "deviation": p = m["payload"]; print(
                f"DEVIATION: {p['advisory_route']}({p['advisory_risk_pct']}%) -> {p['chosen_route']}({p['chosen_risk_pct']}%)")
            elif t == "panel": print("PANEL:", m["payload"]["type"])
            elif t == "log": print("LOG:", m["text"][:80])
            elif t == "telemetry": pass
            elif t == "summary": print("SUMMARY sampled =", m["payload"]["sampled"]); break
        print("EVENT TYPES SEEN:", {k: seen[k] for k in sorted(seen)})


if __name__ == "__main__":
    asyncio.run(main())
