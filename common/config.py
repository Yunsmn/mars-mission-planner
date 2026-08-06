"""Config loader — reads config.yaml so no magic numbers live in code.

Trivial scaffolding (not project logic). Bob may extend with validation in task B1.
"""
from __future__ import annotations

from pathlib import Path

from common.types import Constraints

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(path: str | Path = _DEFAULT_PATH) -> dict:
    """Load the YAML config into a plain dict."""
    import yaml  # lazy import so this module imports even before deps are installed

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def constraints_from(cfg: dict) -> Constraints:
    """Build the typed Constraints object from the loaded config."""
    c = cfg["constraints"]
    return Constraints(
        battery_reserve_pct=float(c["battery_reserve_pct"]),
        risk_ceiling=float(c["risk_ceiling"]),
        p_success_min=float(c["p_success_min"]),
    )
