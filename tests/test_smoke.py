"""Baseline smoke test — the skeleton is importable and config loads.

Gives Bob a green starting point before implementing B1-B12. Behavior tests come with
each task (see docs/BUILD_WITH_BOB.md).
"""
import importlib

MODULES = [
    "common.types", "common.config",
    "rover.capabilities", "rover.energy",
    "world.dem", "world.sim", "world.sensors", "world.calibrate",
    "planner.surrogate", "planner.gating", "planner.voi", "planner.justify", "planner.loop",
    "model.propose",
    "data.science_value", "data.dust", "data.prepare",
    "demo.run",
]


def test_all_modules_import():
    for m in MODULES:
        importlib.import_module(m)


def test_config_loads():
    from common.config import constraints_from, load_config

    cfg = load_config()
    c = constraints_from(cfg)
    assert 0 <= c.battery_reserve_pct <= 100
    assert 0 < c.risk_ceiling < 1
    assert cfg["model"]["name"] == "gemma3:4b"
