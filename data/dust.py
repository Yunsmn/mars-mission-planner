"""Dust opacity (τ) time series from MEDA/REMS -> the dynamic energy driver.

A τ spike (dust event) is the demo's 'conditions changed -> replan/defer' moment.

TODO(bob): implement (task B5).
"""
from __future__ import annotations


def load_dust_tau(path: str) -> "np.ndarray":  # noqa: F821
    """Load a τ(t) series (e.g. CSV of sol -> optical depth)."""
    raise NotImplementedError("TODO(bob): load dust optical-depth time series")


def tau_at(sol_time: float, series: "np.ndarray") -> float:  # noqa: F821
    raise NotImplementedError("TODO(bob): interpolate τ at a given sol time")
