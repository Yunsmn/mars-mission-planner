"""Dust opacity data and solar power modeling.

Loads MEDA/REMS dust opacity (τ) time series and models its impact on solar power generation.
Higher dust opacity → less sunlight → reduced charging rate. A dust storm event triggers
replanning as a dynamic environmental change.

See docs/DATA.md for MEDA/REMS data sources.

Authored by IBM Bob for the MARVIN mission planner.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def load_dust_series(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load dust opacity time series.
    
    Args:
        path: Path to dust_tau_series.csv
    
    Returns:
        Tuple of (sols, tau_values)
    """
    path = Path(path)
    if not path.exists():
        logger.warning(f"Dust series not found at {path}, using baseline")
        # Return baseline constant tau
        return np.array([0]), np.array([0.5])
    
    try:
        import pandas as pd
        df = pd.read_csv(path)
        sols = df['sol'].values
        tau = df['dust_tau'].values
        logger.info(f"Loaded dust series: {len(sols)} sols, "
                   f"tau range [{tau.min():.2f}, {tau.max():.2f}]")
        return sols, tau
    except Exception as e:
        logger.error(f"Failed to load dust series: {e}")
        return np.array([0]), np.array([0.5])


def get_dust_tau(sol: float, sols: np.ndarray, tau_values: np.ndarray) -> float:
    """Get dust opacity at a given sol via interpolation.
    
    Args:
        sol: Current sol (mission day)
        sols: Array of sol values from time series
        tau_values: Array of tau values from time series
    
    Returns:
        Interpolated dust opacity
    """
    if len(sols) == 1:
        return float(tau_values[0])
    
    # Linear interpolation
    tau = np.interp(sol, sols, tau_values)
    return float(tau)


def solar_power_factor(tau: float, tau_reference: float = 0.5) -> float:
    """Calculate solar power reduction factor from dust opacity.
    
    Uses Beer-Lambert law approximation: transmission ∝ exp(-tau)
    Normalized to 1.0 at tau_reference (baseline conditions).
    
    Args:
        tau: Current dust opacity
        tau_reference: Reference opacity for normalization (default: 0.5)
    
    Returns:
        Power factor (0-1 scale, 1.0 = full power at reference conditions)
    """
    # Beer-Lambert: I = I0 * exp(-tau)
    # Normalize to reference conditions
    factor = np.exp(-tau) / np.exp(-tau_reference)
    return float(np.clip(factor, 0.1, 2.0))  # Clip to reasonable range


def detect_dust_event(
    current_tau: float,
    previous_tau: float,
    threshold: float = 0.5
) -> bool:
    """Detect if a significant dust event has occurred.
    
    A dust event (storm) is a rapid increase in opacity that warrants replanning.
    
    Args:
        current_tau: Current dust opacity
        previous_tau: Previous dust opacity
        threshold: Minimum tau increase to trigger event (default: 0.5)
    
    Returns:
        True if dust event detected
    """
    return (current_tau - previous_tau) > threshold


def estimate_charge_rate(
    tau: float,
    nominal_charge_w: float = 100.0,
    tau_reference: float = 0.5
) -> float:
    """Estimate solar charging rate given dust conditions.
    
    Args:
        tau: Current dust opacity
        nominal_charge_w: Nominal charge rate at reference conditions (Watts)
        tau_reference: Reference opacity (default: 0.5)
    
    Returns:
        Estimated charge rate in Watts
    """
    power_factor = solar_power_factor(tau, tau_reference)
    return nominal_charge_w * power_factor