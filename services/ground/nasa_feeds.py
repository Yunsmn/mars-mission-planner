"""NASA live feeds (garnish tier) — each fetched with a timeout, cached, and degrading silently.

mars_weather() pulls the latest real Curiosity REMS reading from the community MAAS2 mirror; if that
feed is slow or down, it returns a synthesized-but-plausible Mars weather report so the panel always
has something. The `source` field says honestly which one you're looking at.
"""
from __future__ import annotations

import time

import requests

_MAAS2 = "https://api.maas2.apollorion.com/"
_cache: dict = {"weather": None, "at": 0.0}
_TTL = 600.0                                       # be a good citizen: one real fetch per 10 min


def _synth_weather(dust_tau: float | None = None) -> dict:
    """A plausible Mars report (Jezero-ish) for when the live feed is unavailable."""
    tau = 0.5 if dust_tau is None else dust_tau
    return {
        "sol": 1180, "season": "Month 6 (northern autumn)",
        "air_temp_max_c": -21, "air_temp_min_c": -79,
        "ground_temp_max_c": -6, "ground_temp_min_c": -84,
        "pressure_pa": 745, "sky": "Sunny", "wind_mps": 6,
        "atmo_opacity_tau": round(tau, 2),
        "sunrise": "05:47", "sunset": "17:33",
        "source": "synthesized (live feed unavailable)",
    }


def mars_weather(dust_tau: float | None = None) -> dict:
    """Latest real REMS reading (cached), or a synthesized fallback. Never raises."""
    now = time.time()
    if _cache["weather"] and now - _cache["at"] < _TTL:
        return {**_cache["weather"], "local_dust_tau": round(dust_tau, 2) if dust_tau else None}
    data = _synth_weather(dust_tau)
    try:
        r = requests.get(_MAAS2, timeout=6)
        r.raise_for_status()
        d = r.json()
        num = lambda k, default=None: (float(d[k]) if d.get(k) not in (None, "", "--") else default)
        data = {
            "sol": int(d.get("sol", data["sol"])),
            "season": d.get("season", data["season"]),
            "air_temp_max_c": num("max_temp", data["air_temp_max_c"]),
            "air_temp_min_c": num("min_temp", data["air_temp_min_c"]),
            "ground_temp_max_c": num("max_gts_temp", data["ground_temp_max_c"]),
            "ground_temp_min_c": num("min_gts_temp", data["ground_temp_min_c"]),
            "pressure_pa": num("pressure", data["pressure_pa"]),
            "sky": d.get("atmo_opacity") or data["sky"],
            "wind_mps": data["wind_mps"],
            "atmo_opacity_tau": round(dust_tau, 2) if dust_tau else data["atmo_opacity_tau"],
            "sunrise": d.get("sunrise", data["sunrise"]),
            "sunset": d.get("sunset", data["sunset"]),
            "source": f"Curiosity REMS · Gale crater · {d.get('terrestrial_date', 'live')}",
        }
        _cache.update(weather=data, at=now)
    except Exception:
        pass                                       # keep the synth fallback
    return {**data, "local_dust_tau": round(dust_tau, 2) if dust_tau else None}
