"""Apply cleaned, unit-checked rainfall forcing to an ANUGA domain."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anuga
import numpy as np
import pandas as pd

MM_HR_TO_M_S = 1.0e-3 / 3600.0
EXPECTED_UNITS = "mm/hr"


@dataclass(frozen=True)
class RainfallRateSeries:
    """Piecewise-linear rainfall rate callable in ANUGA-compatible m/s."""

    elapsed_seconds: np.ndarray
    rate_m_s: np.ndarray
    scenario: str
    source_units: str = EXPECTED_UNITS
    default_rate_m_s: float = 0.0

    def __call__(self, time_seconds: float) -> float:
        if time_seconds < self.elapsed_seconds[0] or time_seconds > self.elapsed_seconds[-1]:
            return self.default_rate_m_s
        return float(np.interp(time_seconds, self.elapsed_seconds, self.rate_m_s))

    def rate_mm_hr(self, time_seconds: float) -> float:
        return self(time_seconds) / MM_HR_TO_M_S


def load_rainfall_rate(
    rainfall_path: str | Path,
    *,
    scenario: str,
    default_rate_mm_hr: float = 0.0,
) -> RainfallRateSeries:
    """Read one clean scenario and validate its explicit units and timestamps."""
    rainfall = pd.read_parquet(rainfall_path)
    required = {
        "timestamp",
        "scenario",
        "rainfall_mm_hr",
        "is_missing_timestamp",
        "units",
    }
    missing = required.difference(rainfall.columns)
    if missing:
        raise ValueError(f"Clean rainfall is missing columns: {sorted(missing)}")
    units = set(rainfall["units"].dropna().astype(str))
    if units != {EXPECTED_UNITS}:
        raise ValueError(f"Expected rainfall units {EXPECTED_UNITS!r}, found {sorted(units)}")
    selected = rainfall.loc[rainfall["scenario"] == scenario].copy()
    if selected.empty:
        raise ValueError(
            f"Rainfall scenario {scenario!r} is unavailable; "
            f"choose from {sorted(rainfall['scenario'].unique())}."
        )
    selected = selected.sort_values("timestamp")
    if selected["timestamp"].duplicated().any():
        raise ValueError(f"Rainfall scenario {scenario!r} has duplicate timestamps.")
    if selected["is_missing_timestamp"].any() or selected["rainfall_mm_hr"].isna().any():
        raise ValueError(
            f"Rainfall scenario {scenario!r} contains missing timestamps or intensities; "
            "explicitly impute these before running the physics baseline."
        )
    if (selected["rainfall_mm_hr"] < 0).any():
        raise ValueError("Rainfall intensity cannot be negative.")

    timestamps = pd.to_datetime(selected["timestamp"])
    elapsed = (timestamps - timestamps.iloc[0]).dt.total_seconds().to_numpy(dtype=np.float64)
    rates = selected["rainfall_mm_hr"].to_numpy(dtype=np.float64) * MM_HR_TO_M_S
    return RainfallRateSeries(
        elapsed_seconds=elapsed,
        rate_m_s=rates,
        scenario=scenario,
        default_rate_m_s=default_rate_mm_hr * MM_HR_TO_M_S,
    )


def apply_uniform_rainfall(
    domain: anuga.Domain,
    rainfall_path: str | Path,
    *,
    scenario: str,
    default_rate_mm_hr: float = 0.0,
) -> tuple[anuga.Rate_operator, RainfallRateSeries, dict[str, Any]]:
    """Attach a spatially uniform ANUGA Rate_operator over all triangles."""
    series = load_rainfall_rate(
        rainfall_path,
        scenario=scenario,
        default_rate_mm_hr=default_rate_mm_hr,
    )
    operator = anuga.Rate_operator(
        domain,
        rate=series,
        factor=1.0,
        default_rate=series.default_rate_m_s,
        label=f"uniform_rainfall_{scenario}",
        description=f"Spatially uniform rainfall ({EXPECTED_UNITS} converted to m/s)",
    )
    report = {
        "scenario": scenario,
        "source_units": EXPECTED_UNITS,
        "anuga_units": "m/s",
        "conversion_factor": MM_HR_TO_M_S,
        "sample_count": len(series.elapsed_seconds),
        "start_time_s": float(series.elapsed_seconds[0]),
        "end_time_s": float(series.elapsed_seconds[-1]),
        "minimum_rate_mm_hr": float(series.rate_m_s.min() / MM_HR_TO_M_S),
        "maximum_rate_mm_hr": float(series.rate_m_s.max() / MM_HR_TO_M_S),
        "spatial_application": "uniform over all domain triangles",
    }
    return operator, series, report
