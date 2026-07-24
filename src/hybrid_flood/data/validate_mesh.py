"""Report mesh topology defects without modifying mesh geometry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import shapely
from shapely.geometry import MultiPolygon, Polygon
from shapely.validation import explain_validity


def _domain_polygon(domain: gpd.GeoDataFrame | None) -> Polygon | MultiPolygon | None:
    if domain is None or domain.empty:
        return None
    polygons: list[Polygon | MultiPolygon] = []
    for geometry in domain.geometry:
        if geometry is None or geometry.is_empty:
            continue
        if geometry.geom_type in {"Polygon", "MultiPolygon"}:
            polygons.append(geometry)
        elif geometry.geom_type == "LineString" and geometry.is_closed:
            polygons.append(Polygon(geometry))
        elif geometry.geom_type == "MultiLineString":
            polygonized = list(shapely.polygonize(geometry.geoms).geoms)
            polygons.extend(polygonized)
    if not polygons:
        return None
    merged = shapely.union_all(polygons)
    return merged if isinstance(merged, (Polygon, MultiPolygon)) else None


def _polygon_parts(geometry: Polygon | MultiPolygon) -> list[Polygon]:
    return list(geometry.geoms) if isinstance(geometry, MultiPolygon) else [geometry]


def validate_mesh_topology(
    mesh: gpd.GeoDataFrame,
    domain: gpd.GeoDataFrame | None = None,
    *,
    degenerate_area_tolerance_m2: float = 1.0e-6,
    gap_relative_tolerance: float = 1.0e-9,
) -> dict[str, Any]:
    """Inspect mesh validity, triangle degeneracy, and coverage gaps."""
    if mesh.crs is None or not mesh.crs.is_projected:
        raise ValueError("Mesh topology validation requires a projected CRS.")

    null_mask = mesh.geometry.isna()
    empty_mask = mesh.geometry.is_empty
    invalid_mask = ~null_mask & ~mesh.geometry.is_valid
    invalid_indices = mesh.index[invalid_mask].tolist()
    invalid_examples = [
        {"index": int(index), "reason": explain_validity(mesh.geometry.loc[index])}
        for index in invalid_indices[:20]
    ]

    usable = mesh.loc[~null_mask & ~empty_mask]
    polygon_mask = usable.geometry.geom_type == "Polygon"
    non_polygon_indices = usable.index[~polygon_mask].tolist()
    polygons = usable.loc[polygon_mask]
    vertex_counts = polygons.geometry.map(lambda geometry: len(geometry.exterior.coords) - 1)
    non_triangle_indices = polygons.index[vertex_counts != 3].tolist()
    areas = polygons.geometry.area
    degenerate_indices = polygons.index[areas <= degenerate_area_tolerance_m2].tolist()

    mesh_union = shapely.union_all(polygons.geometry.array) if not polygons.empty else None
    internal_gap_count = 0
    internal_gap_area = 0.0
    connected_components = 0
    if isinstance(mesh_union, (Polygon, MultiPolygon)):
        parts = _polygon_parts(mesh_union)
        connected_components = len(parts)
        internal_gap_count = sum(len(polygon.interiors) for polygon in parts)
        internal_gap_area = float(
            sum(Polygon(ring).area for polygon in parts for ring in polygon.interiors)
        )

    domain_geometry = _domain_polygon(domain)
    uncovered_area = 0.0
    domain_area = None
    relative_uncovered_area = 0.0
    material_domain_gap = False
    if domain_geometry is not None and mesh_union is not None:
        domain_area = float(domain_geometry.area)
        uncovered_area = float(domain_geometry.difference(mesh_union).area)
        relative_uncovered_area = uncovered_area / domain_area if domain_area else 0.0
        material_domain_gap = relative_uncovered_area > gap_relative_tolerance

    issues: list[dict[str, Any]] = []
    checks = (
        ("null_geometries", int(null_mask.sum()), "critical"),
        ("empty_geometries", int(empty_mask.sum()), "critical"),
        ("self_intersections_or_invalid", len(invalid_indices), "critical"),
        ("non_polygon_elements", len(non_polygon_indices), "critical"),
        ("non_triangular_elements", len(non_triangle_indices), "critical"),
        ("degenerate_triangles", len(degenerate_indices), "critical"),
        ("internal_gaps", internal_gap_count, "critical"),
        ("disconnected_mesh_components", max(connected_components - 1, 0), "critical"),
    )
    for code, count, severity in checks:
        if count:
            issues.append({"severity": severity, "code": code, "count": count})
    if material_domain_gap:
        issues.append(
            {
                "severity": "critical",
                "code": "domain_coverage_gap",
                "count": 1,
                "uncovered_area_m2": uncovered_area,
                "relative_uncovered_area": relative_uncovered_area,
            }
        )

    return {
        "element_count": len(mesh),
        "crs": mesh.crs.to_string(),
        "area_units": "m2",
        "minimum_element_area_m2": None if areas.empty else float(areas.min()),
        "maximum_element_area_m2": None if areas.empty else float(areas.max()),
        "invalid_geometry_indices": [int(index) for index in invalid_indices[:100]],
        "invalid_geometry_examples": invalid_examples,
        "non_triangle_indices": [int(index) for index in non_triangle_indices[:100]],
        "degenerate_triangle_indices": [int(index) for index in degenerate_indices[:100]],
        "connected_components": connected_components,
        "internal_gap_count": internal_gap_count,
        "internal_gap_area_m2": internal_gap_area,
        "domain_area_m2": domain_area,
        "uncovered_domain_area_m2": uncovered_area,
        "relative_uncovered_domain_area": relative_uncovered_area,
        "gap_relative_tolerance": gap_relative_tolerance,
        "issues": issues,
        "critical_errors": sum(issue["severity"] == "critical" for issue in issues),
        "warnings": sum(issue["severity"] == "warning" for issue in issues),
    }


def validate_mesh_file(
    mesh_path: str | Path,
    domain_path: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Load mesh/domain files and return their topology report."""
    mesh = gpd.read_file(mesh_path)
    domain = gpd.read_file(domain_path) if domain_path is not None else None
    return validate_mesh_topology(mesh, domain, **kwargs)
