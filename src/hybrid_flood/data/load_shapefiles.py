"""Load and validate the project's raw vector layers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LayerSpec:
    """Expected location and schema for one raw vector layer."""

    relative_path: str
    required_columns: frozenset[str]
    allowed_geometry_types: frozenset[str]


LAYER_SPECS: dict[str, LayerSpec] = {
    "boundaries": LayerSpec(
        relative_path="boundaries/Boundaries_Line.shp",
        required_columns=frozenset({"MUID", "BndTypeNo"}),
        allowed_geometry_types=frozenset({"LineString", "MultiLineString"}),
    ),
    "domain": LayerSpec(
        relative_path="mesh/Domain_definitions_Mesh_arcPolygons.shp",
        required_columns=frozenset({"MUID", "IsClosedNo"}),
        # This source stores its polygon perimeter as a closed LineString.
        allowed_geometry_types=frozenset(
            {"Polygon", "MultiPolygon", "LineString", "MultiLineString"}
        ),
    ),
    "mesh_elements": LayerSpec(
        relative_path="mesh/Gurugram_First_Model_grid_Elements.shp",
        required_columns=frozenset({"Bathymetry"}),
        allowed_geometry_types=frozenset({"Polygon", "MultiPolygon"}),
    ),
    "roughness": LayerSpec(
        relative_path="roughness/Gurugram_LULC_Roughness_fast.shp",
        required_columns=frozenset({"zone_value", "lulc_class", "roughness"}),
        allowed_geometry_types=frozenset({"Polygon", "MultiPolygon"}),
    ),
}


class SchemaValidationError(ValueError):
    """Raised when a vector layer has critical schema or geometry errors."""


def _json_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    return value if isinstance(value, (str, int, float, bool)) else str(value)


def _attribute_summary(gdf: gpd.GeoDataFrame) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for column in gdf.columns:
        if column == gdf.geometry.name:
            continue
        series = gdf[column]
        summary: dict[str, Any] = {
            "dtype": str(series.dtype),
            "null_count": int(series.isna().sum()),
            "unique_count": int(series.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if not numeric.empty:
                summary.update(
                    {
                        "min": _json_scalar(numeric.min()),
                        "max": _json_scalar(numeric.max()),
                        "mean": _json_scalar(numeric.mean()),
                    }
                )
        else:
            counts = series.dropna().astype(str).value_counts().head(10)
            summary["top_values"] = {str(key): int(value) for key, value in counts.items()}
        summaries[column] = summary
    return summaries


def validate_layer(
    gdf: gpd.GeoDataFrame,
    layer_name: str,
    spec: LayerSpec | None = None,
) -> dict[str, Any]:
    """Validate one layer and return a JSON-serializable report."""
    spec = spec or LAYER_SPECS[layer_name]
    issues: list[dict[str, Any]] = []

    missing_columns = sorted(spec.required_columns.difference(gdf.columns))
    if missing_columns:
        issues.append(
            {
                "severity": "critical",
                "code": "missing_required_columns",
                "message": f"Missing required columns: {missing_columns}",
                "count": len(missing_columns),
            }
        )

    geometry_types = sorted(str(value) for value in gdf.geometry.geom_type.dropna().unique())
    unexpected_types = sorted(set(geometry_types).difference(spec.allowed_geometry_types))
    if unexpected_types:
        issues.append(
            {
                "severity": "critical",
                "code": "unexpected_geometry_types",
                "message": f"Unexpected geometry types: {unexpected_types}",
                "count": int(gdf.geometry.geom_type.isin(unexpected_types).sum()),
            }
        )

    null_count = int(gdf.geometry.isna().sum())
    empty_count = int(gdf.geometry.is_empty.sum())
    invalid_count = int((~gdf.geometry.isna() & ~gdf.geometry.is_valid).sum())
    for code, count, label in (
        ("null_geometries", null_count, "null"),
        ("empty_geometries", empty_count, "empty"),
        ("invalid_geometries", invalid_count, "invalid"),
    ):
        if count:
            issues.append(
                {
                    "severity": "critical",
                    "code": code,
                    "message": f"Found {count} {label} geometries.",
                    "count": count,
                }
            )

    if gdf.crs is None:
        issues.append(
            {
                "severity": "critical",
                "code": "missing_crs",
                "message": "Layer has no declared coordinate reference system.",
                "count": 1,
            }
        )

    if layer_name == "domain":
        line_geometries = gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])]
        if not line_geometries.empty:
            not_closed = int((~line_geometries.geometry.is_closed).sum())
            if not_closed:
                issues.append(
                    {
                        "severity": "critical",
                        "code": "open_domain_boundary",
                        "message": f"Found {not_closed} open domain perimeter geometries.",
                        "count": not_closed,
                    }
                )
            else:
                issues.append(
                    {
                        "severity": "warning",
                        "code": "domain_stored_as_closed_line",
                        "message": (
                            "Domain is stored as a closed line perimeter rather than a polygon; "
                            "it will be polygonized only in memory for topology checks."
                        ),
                        "count": len(line_geometries),
                    }
                )

    bounds = None if gdf.empty else [float(value) for value in gdf.total_bounds]
    return {
        "source": str(spec.relative_path),
        "row_count": len(gdf),
        "crs": None if gdf.crs is None else gdf.crs.to_string(),
        "bounding_box": bounds,
        "geometry_types": {
            str(key): int(value)
            for key, value in gdf.geometry.geom_type.value_counts(dropna=False).items()
        },
        "required_columns": sorted(spec.required_columns),
        "columns": [str(column) for column in gdf.columns],
        "attribute_summary": _attribute_summary(gdf),
        "issues": issues,
        "critical_errors": sum(issue["severity"] == "critical" for issue in issues),
        "warnings": sum(issue["severity"] == "warning" for issue in issues),
    }


def load_layer(
    raw_data_dir: str | Path,
    layer_name: str,
    *,
    strict: bool = True,
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Load and validate one configured raw vector layer."""
    if layer_name not in LAYER_SPECS:
        raise KeyError(f"Unknown layer {layer_name!r}; expected one of {sorted(LAYER_SPECS)}")
    spec = LAYER_SPECS[layer_name]
    path = Path(raw_data_dir) / spec.relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Required shapefile does not exist: {path}")

    gdf = gpd.read_file(path)
    report = validate_layer(gdf, layer_name, spec)
    if strict and report["critical_errors"]:
        messages = [
            issue["message"] for issue in report["issues"] if issue["severity"] == "critical"
        ]
        raise SchemaValidationError(f"{layer_name}: {'; '.join(messages)}")
    return gdf, report


def load_boundary_lines(raw_data_dir: str | Path, *, strict: bool = True) -> gpd.GeoDataFrame:
    return load_layer(raw_data_dir, "boundaries", strict=strict)[0]


def load_domain(raw_data_dir: str | Path, *, strict: bool = True) -> gpd.GeoDataFrame:
    return load_layer(raw_data_dir, "domain", strict=strict)[0]


def load_mesh_elements(raw_data_dir: str | Path, *, strict: bool = True) -> gpd.GeoDataFrame:
    return load_layer(raw_data_dir, "mesh_elements", strict=strict)[0]


def load_roughness(raw_data_dir: str | Path, *, strict: bool = True) -> gpd.GeoDataFrame:
    return load_layer(raw_data_dir, "roughness", strict=strict)[0]


def write_validation_report(report: dict[str, Any], report_path: str | Path) -> Path:
    """Write a report atomically enough to avoid leaving partial JSON."""
    output_path = Path(report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    temporary_path.replace(output_path)
    return output_path


def load_all_shapefiles(
    raw_data_dir: str | Path,
    report_path: str | Path | None = None,
    *,
    strict: bool = True,
) -> tuple[dict[str, gpd.GeoDataFrame], dict[str, Any]]:
    """Load every raw vector layer and optionally write the validation report."""
    layers: dict[str, gpd.GeoDataFrame] = {}
    layer_reports: dict[str, Any] = {}
    for layer_name in LAYER_SPECS:
        layer, layer_report = load_layer(raw_data_dir, layer_name, strict=False)
        layers[layer_name] = layer
        layer_reports[layer_name] = layer_report

    critical_errors = sum(item["critical_errors"] for item in layer_reports.values())
    warnings = sum(item["warnings"] for item in layer_reports.values())
    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "raw_data_dir": str(Path(raw_data_dir)),
        "layers": layer_reports,
        "summary": {
            "critical_errors": critical_errors,
            "warnings": warnings,
        },
    }
    if report_path is not None:
        write_validation_report(report, report_path)
    if strict and critical_errors:
        raise SchemaValidationError(
            f"Raw vector validation found {critical_errors} critical error(s); "
            f"see {report_path or 'the returned report'}."
        )
    return layers, report
