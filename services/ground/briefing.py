"""The mission briefing — the contract between the two Granites.

HOUSTON (cloud) expands a one-sentence operator intent into this structured document; the rover's
intent_api consumes it and hands the objectives to MARVIN's existing planner. The `route_advisory`
is written from ORBITAL knowledge only (the DEM) and is explicitly non-binding — MARVIN owns live
perception and may deviate. That deviation is the whole point of the Purgatory story.

`validate` is strict on the structure the rover depends on; HOUSTON re-asks once on failure. The
schema is intentionally small — objectives, constraints, an advisory, abort conditions.
"""
from __future__ import annotations

import json
import re

VALID_OBJECTIVE_TYPES = {"sample", "image", "drive", "observe"}


def schema_hint() -> str:
    """A compact example of the exact JSON shape, injected into HOUSTON's prompt."""
    return json.dumps({
        "mission_id": "sol-0447-a",
        "objective_summary": "Acquire and cache one carbonate sample.",
        "objectives": [
            {"id": "obj-1", "type": "sample", "target": "carbonate_outcrop", "priority": 1,
             "science_rationale": "Carbonates record ancient water chemistry; primary driver."}
        ],
        "constraints": {"battery_reserve_pct": 15, "max_decisions": 12, "keep_out_zones": []},
        "route_advisory": {"note": "Direct approach appears shortest on the orbital DEM.",
                           "basis": "orbital", "binding": False},
        "abort_conditions": ["all candidate routes exceed 20% entrapment risk",
                             "battery below reserve before objective complete"],
    }, indent=2)


def validate(b: dict) -> list[str]:
    """Return a list of problems (empty == valid). Strict on what the rover relies on."""
    errs: list[str] = []
    if not isinstance(b, dict):
        return ["briefing is not an object"]
    if not b.get("objective_summary"):
        errs.append("missing objective_summary")
    objs = b.get("objectives")
    if not isinstance(objs, list) or not objs:
        errs.append("objectives must be a non-empty list")
    else:
        for i, o in enumerate(objs):
            if not isinstance(o, dict):
                errs.append(f"objective {i} is not an object"); continue
            if o.get("type") not in VALID_OBJECTIVE_TYPES:
                errs.append(f"objective {i}: type must be one of {sorted(VALID_OBJECTIVE_TYPES)}")
            if not o.get("target"):
                errs.append(f"objective {i}: missing target")
    cons = b.get("constraints", {})
    if not isinstance(cons, dict):
        errs.append("constraints must be an object")
    adv = b.get("route_advisory", {})
    if not isinstance(adv, dict) or "note" not in adv:
        errs.append("route_advisory must be an object with a note")
    return errs


def normalize(b: dict, *, mission_id: str, operator_intent: str, delay_s: float) -> dict:
    """Fill server-authoritative fields the model shouldn't be trusted to set."""
    b = dict(b)
    b["mission_id"] = mission_id
    b["issued_by"] = "HOUSTON"
    b["operator_intent"] = operator_intent
    b.setdefault("constraints", {}).setdefault("battery_reserve_pct", 15)
    b["constraints"].setdefault("max_decisions", 12)
    b["constraints"].setdefault("keep_out_zones", [])
    adv = b.setdefault("route_advisory", {"note": "Direct approach appears shortest on the orbital DEM."})
    adv["basis"] = "orbital"
    adv["binding"] = False                      # invariant: advisories are never binding
    b.setdefault("abort_conditions", [
        "all candidate routes exceed 20% entrapment risk",
        "battery below reserve before objective complete"])
    b["comm"] = {"uplink_delay_s": delay_s, "downlink_delay_s": delay_s}
    return b


def parse_model_json(text: str) -> dict | None:
    """Pull the first JSON object out of an LLM reply (it may wrap it in prose or fences)."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
