"""End-to-end proof of the two-Granite architecture, headless — no web server, no cloud keys.

operator intent -> HOUSTON routes it -> HOUSTON writes a full briefing -> uplink pays the (compressed)
light-time -> MARVIN runs the existing planner and DEVIATES from the orbital route advisory -> downlink.

Run:  LINK_DELAY_S=3 .venv/bin/python -m services.flow_demo
"""
from __future__ import annotations

import asyncio
import os
import time

import yaml

from demo import scene
from model.propose import Proposer
from services.ground import link as linkmod
from services.ground.houston import Houston
from services.rover.intent_api import Rover

INTENT = "get me a sample from the carbonate outcrop east of the dunes"
SCENARIO = ("Known target: carbonate outcrop at (0.0, 3.0), high science value. Surrounding terrain "
            "is rolling, mapped only by orbital DEM (no view of local soft sand).")


async def main():
    cfg0 = yaml.safe_load(open("config.yaml"))
    cfg = {**cfg0["planner"], **cfg0["constraints"], "energy_penalty_factor": 0.01}
    delay = float(os.getenv("LINK_DELAY_S", "3"))
    link = linkmod.DeepSpaceLink(delay_s=delay)
    houston = Houston()                                    # watsonx if keys, else local fallback
    marvin = Proposer(cfg0["model"]["name"], cfg0["model"]["host"], cfg0["model"]["temperature"])
    rover = Rover(cfg, model=marvin)

    print(f"\nOPERATOR (Earth):  {INTENT}")
    print(f"HOUSTON backend:   {houston.backend_label}\n")

    r = houston.route(INTENT, delay)
    print(f"HOUSTON [{r['target']}]:  {r['reply']}")
    if r["target"] != "marvin":
        return

    brief = houston.brief(INTENT, SCENARIO, "sol-0447-a", delay)
    if not brief:
        print("HOUSTON: briefing generation failed."); return
    print(f"\nBRIEFING sol-0447-a")
    print(f"  summary : {brief['objective_summary']}")
    for o in brief["objectives"]:
        print(f"  obj     : [{o.get('priority','?')}] {o['type']} {o['target']} — {o.get('science_rationale','')[:70]}")
    print(f"  advisory: {brief['route_advisory']['note']} (basis={brief['route_advisory']['basis']}, binding={brief['route_advisory']['binding']})")

    print(f"\n  ...UPLINK crossing the link ({delay:.0f}s compressed light-time)...")
    t0 = time.time()
    await link.uplink(brief, "briefing")
    print(f"  ...received on Mars after {time.time()-t0:.0f}s\n")

    deviated = False
    for kind, payload in rover.run_briefing(brief):
        if kind == "deviation":
            deviated = True
            print(f"  *** DEVIATION REPORT (downlink) ***")
            print(f"      Earth advised : {payload['advisory_route']} route ({payload['advisory_risk_pct']}% entrapment)")
            print(f"      MARVIN chose  : {payload['chosen_route']} route ({payload['chosen_risk_pct']}%, {payload['chosen_length_m']} m)")
            print(f"      decided by    : {payload['decided_by']}")
            print(f"      justification : {payload['justification']}")
        elif kind == "log":
            print(f"  MARVIN: {payload}")
        elif kind == "summary":
            print(f"  MARVIN SUMMARY: sampled={payload['sampled']} via {payload['route']} route, "
                  f"battery {payload['battery_pct']}%")

    print(f"\nRESULT: deviation reported = {deviated}")
    print("Earth planned from orbit. The rover routed on what it sensed.\n")


if __name__ == "__main__":
    asyncio.run(main())
