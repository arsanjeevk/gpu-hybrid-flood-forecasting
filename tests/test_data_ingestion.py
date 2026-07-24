"""Integration tests against the project's real raw GIS and rainfall files."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest

from hybrid_flood.data.load_rainfall import (
    RAINFALL_UNITS,
    load_rainfall,
    rainfall_to_xarray,
    save_clean_rainfall,
)
from hybrid_flood.data.load_shapefiles import LAYER_SPECS, load_all_shapefiles
from hybrid_flood.data.reproject import TARGET_CRS, reproject_all_layers
from hybrid_flood.data.validate_mesh import validate_mesh_topology

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


@pytest.fixture(scope="module")
def validated_layers() -> tuple[dict[str, gpd.GeoDataFrame], dict]:
    return load_all_shapefiles(RAW_DIR, strict=True)


def test_actual_shapefiles_have_no_critical_schema_errors(validated_layers) -> None:
    layers, report = validated_layers
    assert set(layers) == set(LAYER_SPECS)
    assert report["summary"]["critical_errors"] == 0
    assert all(layer_report["row_count"] > 0 for layer_report in report["layers"].values())


def test_actual_layers_reproject_to_utm_43n(validated_layers, tmp_path: Path) -> None:
    layers, _ = validated_layers
    outputs = reproject_all_layers(RAW_DIR, tmp_path, target_crs=TARGET_CRS, layers=layers)
    assert set(outputs) == set(LAYER_SPECS)
    for output in outputs.values():
        projected = gpd.read_file(output)
        assert projected.crs is not None
        assert projected.crs.to_epsg() == 32643
        assert not projected.geometry.isna().any()


def test_actual_rainfall_is_tidy_unit_aware_and_round_trips(tmp_path: Path) -> None:
    rainfall = load_rainfall(RAW_DIR / "rainfall" / "rain.xlsx")
    assert {
        "timestamp",
        "scenario",
        "rainfall_mm_hr",
        "is_missing_timestamp",
        "units",
    }.issubset(rainfall.columns)
    assert set(rainfall["units"]) == {RAINFALL_UNITS}
    assert rainfall["rainfall_mm_hr"].dropna().ge(0).all()
    assert rainfall["scenario"].nunique() == 3
    assert rainfall["timestamp"].nunique() == 35
    assert not rainfall["is_missing_timestamp"].any()

    dataset = rainfall_to_xarray(rainfall)
    assert dataset["rainfall_mm_hr"].attrs["units"] == RAINFALL_UNITS

    output = save_clean_rainfall(rainfall, tmp_path / "rainfall_clean.parquet")
    restored = pd.read_parquet(output)
    assert len(restored) == len(rainfall)


def test_actual_mesh_has_no_critical_topology_errors(validated_layers) -> None:
    layers, _ = validated_layers
    report = validate_mesh_topology(layers["mesh_elements"], layers["domain"])
    assert report["element_count"] == 82_921
    assert report["critical_errors"] == 0, report["issues"]
    assert report["internal_gap_count"] == 0
    assert report["connected_components"] == 1


def test_hydra_pipeline_writes_zero_critical_validation_report() -> None:
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "01_prepare_data.py")],
        cwd=PROJECT_ROOT,
        check=True,
    )
    report_path = PROJECT_ROOT / "data" / "interim" / "validation_report.json"
    with report_path.open(encoding="utf-8") as stream:
        report = json.load(stream)
    assert report["summary"]["critical_errors"] == 0
    assert report["mesh_topology"]["critical_errors"] == 0
    assert Path(PROJECT_ROOT / report["rainfall"]["output"]).is_file()
