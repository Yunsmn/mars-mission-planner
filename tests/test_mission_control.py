"""Mission Control spine — the invariants that protect the two-AI architecture (no LLM needed)."""
from __future__ import annotations

import asyncio
import time

from services.ground import briefing as B
from services.ground.link import DeepSpaceLink


def test_link_pays_the_light_time_each_way():
    link = DeepSpaceLink(delay_s=0.3)

    async def go():
        t0 = time.time()
        await link.uplink({"hi": 1}, "message")
        return time.time() - t0

    assert asyncio.run(go()) >= 0.3          # nothing crosses instantly


def test_link_is_fifo_channel_busy():
    link = DeepSpaceLink(delay_s=0.2)

    async def go():
        t0 = time.time()
        await asyncio.gather(link.uplink("a"), link.uplink("b"))   # b waits behind a
        return time.time() - t0

    assert asyncio.run(go()) >= 0.4          # two transmissions serialize on the one channel


def test_downlink_reaches_handlers_after_delay():
    link = DeepSpaceLink(delay_s=0.2)
    got = []
    link.on_downlink(lambda tx: got.append(tx.payload) or asyncio.sleep(0))
    asyncio.run(link.downlink({"telemetry": 1}, "telemetry"))
    assert got == [{"telemetry": 1}]


def test_briefing_validation_rejects_garbage():
    assert B.validate({}), "empty briefing must be invalid"
    assert B.validate({"objective_summary": "x", "objectives": [], "route_advisory": {"note": "n"}})
    good = {"objective_summary": "cache a sample", "route_advisory": {"note": "direct"},
            "objectives": [{"id": "obj-1", "type": "sample", "target": "carbonate", "priority": 1}]}
    assert B.validate(good) == []


def test_normalize_enforces_invariants():
    b = B.normalize({"objective_summary": "s", "objectives": [{"type": "sample", "target": "c"}],
                     "route_advisory": {"note": "direct"}},
                    mission_id="sol-1-a", operator_intent="get a sample", delay_s=12.0)
    assert b["issued_by"] == "HOUSTON"
    assert b["route_advisory"]["binding"] is False       # advisories are NEVER binding
    assert b["comm"]["uplink_delay_s"] == 12.0


def test_parse_model_json_survives_prose_and_fences():
    assert B.parse_model_json('sure!\n```json\n{"a": 1}\n```') == {"a": 1}
    assert B.parse_model_json("no json here") is None
