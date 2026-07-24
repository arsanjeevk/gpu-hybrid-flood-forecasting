"""Diagnostics for the portion of a rainfall record used by a simulation."""

from __future__ import annotations

from typing import Any

import numpy as np

M_S_TO_MM_HR = 3.6e6


def summarize_rainfall_window(
    elapsed_seconds: np.ndarray,
    rate_m_s: np.ndarray,
    duration_s: float,
    *,
    default_rate_m_s: float = 0.0,
) -> dict[str, Any]:
    """Summarize forcing coverage and integrate rainfall over the run window."""
    times = np.asarray(elapsed_seconds, dtype=np.float64)
    rates = np.asarray(rate_m_s, dtype=np.float64)
    if times.ndim != 1 or rates.ndim != 1 or len(times) != len(rates) or len(times) == 0:
        raise ValueError("Rainfall times and rates must be non-empty matching vectors.")
    if duration_s <= 0 or np.any(np.diff(times) <= 0):
        raise ValueError("Duration must be positive and rainfall times strictly increasing.")
    if not np.isfinite(times).all() or not np.isfinite(rates).all():
        raise ValueError("Rainfall times and rates must be finite.")

    overlap_start = max(0.0, float(times[0]))
    overlap_end = min(float(duration_s), float(times[-1]))
    integrated_depth_m = 0.0
    if overlap_end > overlap_start:
        interior = times[(times > overlap_start) & (times < overlap_end)]
        integration_times = np.concatenate(([overlap_start], interior, [overlap_end]))
        integration_rates = np.interp(integration_times, times, rates)
        integrated_depth_m += float(np.trapezoid(integration_rates, integration_times))
    integrated_depth_m += max(0.0, min(duration_s, times[0])) * default_rate_m_s
    integrated_depth_m += max(0.0, duration_s - max(times[-1], 0.0)) * default_rate_m_s

    peak_index = int(np.argmax(rates))
    peak_time_s = float(times[peak_index])
    record_duration_s = float(times[-1] - times[0])
    return {
        "simulation_duration_s": float(duration_s),
        "record_start_time_s": float(times[0]),
        "record_end_time_s": float(times[-1]),
        "record_duration_s": record_duration_s,
        "record_fraction_covered": (
            float(min(duration_s, times[-1]) - max(0.0, times[0])) / record_duration_s
            if record_duration_s > 0
            else 1.0
        ),
        "peak_time_s": peak_time_s,
        "peak_rate_mm_hr": float(rates[peak_index] * M_S_TO_MM_HR),
        "peak_in_simulation_window": bool(0.0 <= peak_time_s <= duration_s),
        "integrated_rainfall_depth_mm": integrated_depth_m * 1.0e3,
        "complete_record_covered": bool(duration_s >= times[-1] and times[0] >= 0.0),
    }
