"""Integration checks for the real ANUGA domain inputs."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from hybrid_flood.anuga_solver.boundary_conditions import (
    FINAL_REFLECTIVE_POLICY,
    REFLECTIVE_ASSUMPTION,
    configure_boundary_conditions,
)
from hybrid_flood.anuga_solver.build_domain import MANNING_N_BY_LULC, build_domain
from hybrid_flood.anuga_solver.rainfall_forcing import (
    MM_HR_TO_M_S,
    apply_uniform_rainfall,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def real_domain():
    domain, report = build_domain(
        PROJECT_ROOT / "data/interim/mesh_elements_epsg32643.gpkg",
        PROJECT_ROOT / "data/interim/boundaries_epsg32643.gpkg",
        PROJECT_ROOT / "data/interim/roughness_epsg32643.gpkg",
        PROJECT_ROOT / "data/synthetic_dem/dem.tif",
    )
    return domain, report


def test_real_domain_uses_every_mesh_element_and_valid_dem(real_domain) -> None:
    domain, report = real_domain
    assert report["triangle_count"] == 82_921
    assert len(domain) == 82_921
    elevation = domain.quantities["elevation"].centroid_values
    assert np.isfinite(elevation).all()
    assert 200.0 <= elevation.min() < elevation.max() <= 250.0
    assert domain.geo_reference.epsg == 32643


def test_real_domain_assigns_documented_roughness_lookup(real_domain) -> None:
    domain, report = real_domain
    friction = domain.quantities["friction"].centroid_values
    assert set(np.unique(friction)) == set(MANNING_N_BY_LULC.values())
    assert sum(report["elements_by_lulc"].values()) == 82_921
    assert "Chow" in report["manning_reference"]


def test_final_real_boundary_policy_is_deliberately_reflective(real_domain, caplog) -> None:
    domain, _ = real_domain
    caplog.set_level(logging.INFO)
    report = configure_boundary_conditions(domain)
    assert report["policy"] == FINAL_REFLECTIVE_POLICY
    assert report["modeling_assumption"] == REFLECTIVE_ASSUMPTION
    assert REFLECTIVE_ASSUMPTION in caplog.text
    assert "warning" not in report
    assert set(report["condition_by_tag"].values()) == {"Reflective_boundary"}
    assert "segment_2Dboundary_1" in report["boundary_tags"]
    assert "exterior" in report["boundary_tags"]


def test_clean_rainfall_is_converted_from_mm_hr_to_m_s(real_domain) -> None:
    domain, _ = real_domain
    operator, series, report = apply_uniform_rainfall(
        domain,
        PROJECT_ROOT / "data/interim/rainfall_clean.parquet",
        scenario="45rp_rain",
    )
    assert operator.domain is domain
    assert series(0.0) == pytest.approx(2.73436 * MM_HR_TO_M_S)
    assert report["source_units"] == "mm/hr"
    assert report["anuga_units"] == "m/s"
