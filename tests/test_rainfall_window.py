"""Rainfall forcing-window diagnostics."""

from __future__ import annotations

import numpy as np

from hybrid_flood.data.rainfall_window import summarize_rainfall_window


def test_window_report_flags_excluded_peak_and_integrates_linear_rate() -> None:
    report = summarize_rainfall_window(
        np.asarray([0.0, 3600.0, 7200.0]),
        np.asarray([0.0, 1.0e-3 / 3600.0, 0.0]),
        3600.0,
    )
    assert report["peak_in_simulation_window"]
    assert not report["complete_record_covered"]
    assert report["record_fraction_covered"] == 0.5
    assert report["integrated_rainfall_depth_mm"] == 0.5


def test_production_window_excludes_late_scenario_peak() -> None:
    report = summarize_rainfall_window(
        np.arange(35, dtype=np.float64) * 3600.0,
        np.concatenate((np.ones(34), np.asarray([5.0]))) * 1.0e-3 / 3600.0,
        3600.0,
    )
    assert not report["peak_in_simulation_window"]
    assert report["record_fraction_covered"] < 0.03
