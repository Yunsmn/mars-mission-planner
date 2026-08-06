"""Energy / power model. Simple, physical-ish, tunable. Solar charge falls with dust (τ).

Battery is tracked in percent; nominal capacity ~100 Wh so Wh ≈ percentage points here.
See docs/DATA.md (dust) and config.yaml (power).
"""
from __future__ import annotations

import math

BASE_WH_PER_M = 0.6          # flat-ground draw
SLOPE_WH_PER_M_PER_DEG = 0.05
PAYLOAD_WH_PER_KG_FRAC = 0.02
NOMINAL_CHARGE_W = 100.0
DUST_TAU_REFERENCE = 0.5


def drive_cost_wh(dist_m: float, slope_deg: float, payload_kg: float) -> float:
    """Energy (Wh) to traverse `dist_m` at `slope_deg` carrying `payload_kg`."""
    per_m = BASE_WH_PER_M + SLOPE_WH_PER_M_PER_DEG * max(0.0, slope_deg)
    return per_m * max(0.0, dist_m) * (1.0 + PAYLOAD_WH_PER_KG_FRAC * max(0.0, payload_kg))


def power_factor(dust_tau: float) -> float:
    """Solar-charge multiplier in [0, 1] as a function of dust optical depth (higher τ → less)."""
    return float(max(0.0, min(1.0, math.exp(-max(0.0, dust_tau) / DUST_TAU_REFERENCE))))


def solar_charge_wh(dt_s: float, dust_tau: float) -> float:
    """Solar energy (Wh) gained over `dt_s` seconds given dust τ."""
    return NOMINAL_CHARGE_W * power_factor(dust_tau) * max(0.0, dt_s) / 3600.0
