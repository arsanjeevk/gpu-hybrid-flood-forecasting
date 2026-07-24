"""Construct the ANUGA baseline domain from project geospatial inputs.

Manning coefficients are effective land-cover values based on the ranges
compiled in Chow, V. T. (1959), *Open-Channel Hydraulics*, McGraw-Hill,
Chapter 5. They represent unresolved surface obstruction at the LULC scale,
not surveyed channel-only roughness:

================  ===========
LULC class        Manning n
================  ===========
Water             0.030
Built-up          0.100
Bare/Open         0.025
Low vegetation    0.050
Tree cover        0.120
================  ===========

The lookup reproduces the values encoded in the supplied roughness layer and
is validated against its ``roughness`` attribute before assignment.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import anuga
import geopandas as gpd
import numpy as np
import rasterio
from anuga.coordinate_transforms.geo_reference import Geo_reference
from scipy.ndimage import map_coordinates
from shapely.geometry import Point

LOGGER = logging.getLogger(__name__)

MANNING_N_BY_LULC = {
    "Water": 0.030,
    "Built-up": 0.100,
    "Bare/Open": 0.025,
    "Low vegetation": 0.050,
    "Tree cover": 0.120,
}


def _triangle_coordinate_array(mesh: gpd.GeoDataFrame) -> np.ndarray:
    coordinates = np.empty((len(mesh), 3, 2), dtype=np.float64)
    for index, geometry in enumerate(mesh.geometry):
        if geometry is None or geometry.is_empty or geometry.geom_type != "Polygon":
            raise ValueError(f"Mesh element {index} is not a non-empty Polygon.")
        exterior = np.asarray(geometry.exterior.coords[:-1], dtype=np.float64)
        if exterior.shape != (3, 2):
            raise ValueError(
                f"Mesh element {index} has {len(exterior)} vertices; ANUGA requires triangles."
            )
        edge_1 = exterior[1] - exterior[0]
        edge_2 = exterior[2] - exterior[0]
        signed_twice_area = edge_1[0] * edge_2[1] - edge_1[1] * edge_2[0]
        if abs(signed_twice_area) <= np.finfo(np.float64).eps:
            raise ValueError(f"Mesh element {index} is degenerate.")
        if signed_twice_area < 0:
            exterior[[1, 2]] = exterior[[2, 1]]
        coordinates[index] = exterior
    return coordinates


def mesh_to_anuga_arrays(mesh: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Convert triangular polygons to unique ANUGA vertices and connectivity."""
    triangle_coordinates = _triangle_coordinate_array(mesh)
    points, inverse = np.unique(
        triangle_coordinates.reshape(-1, 2),
        axis=0,
        return_inverse=True,
    )
    triangles = inverse.reshape(-1, 3).astype(np.int64)
    if len(np.unique(np.sort(triangles, axis=1), axis=0)) != len(triangles):
        raise ValueError("Mesh contains duplicate triangle connectivity.")
    return points, triangles


def _safe_boundary_tag(value: Any) -> str:
    normalized = re.sub(r"[^0-9A-Za-z_]+", "_", str(value)).strip("_")
    return f"segment_{normalized or 'unnamed'}"


def tag_exterior_edges(
    domain: anuga.Domain,
    boundary_lines: gpd.GeoDataFrame,
    *,
    match_tolerance_m: float = 1.0,
) -> dict[str, Any]:
    """Tag exterior ANUGA edges by proximity to supplied boundary segments."""
    if "MUID" not in boundary_lines.columns:
        raise ValueError("Boundary layer must contain the MUID segment identifier.")
    if boundary_lines.crs is None or not boundary_lines.crs.is_projected:
        raise ValueError("Boundary tagging requires a projected CRS.")

    line_records = list(boundary_lines.itertuples())
    matched_counts = {_safe_boundary_tag(record.MUID): 0 for record in line_records}
    triangle_nodes = domain.triangles
    points = domain.nodes
    edge_node_positions = ((1, 2), (2, 0), (0, 1))

    for boundary_key in list(domain.boundary):
        triangle_index, edge_index = boundary_key
        local_a, local_b = edge_node_positions[edge_index]
        node_a = points[triangle_nodes[triangle_index, local_a]]
        node_b = points[triangle_nodes[triangle_index, local_b]]
        midpoint = Point((node_a + node_b) / 2.0)
        distances = np.asarray(
            [record.geometry.distance(midpoint) for record in line_records],
            dtype=np.float64,
        )
        nearest = int(np.argmin(distances))
        if distances[nearest] <= match_tolerance_m:
            tag = _safe_boundary_tag(line_records[nearest].MUID)
            domain.boundary[boundary_key] = tag
            matched_counts[tag] += 1

    unresolved_attributes = []
    for record in line_records:
        unresolved_attributes.append(
            {
                "MUID": str(record.MUID),
                "BndTypeNo": None
                if getattr(record, "BndTypeNo", None) is None
                else float(record.BndTypeNo),
                "ConstValue": None
                if getattr(record, "ConstValue", None) is None
                else float(record.ConstValue),
                "assigned_tag": _safe_boundary_tag(record.MUID),
            }
        )
    return {
        "matched_edge_counts": matched_counts,
        "unmatched_exterior_edge_count": sum(tag == "exterior" for tag in domain.boundary.values()),
        "source_segment_attributes": unresolved_attributes,
        "match_tolerance_m": match_tolerance_m,
    }


def interpolate_dem_at_centroids(
    domain: anuga.Domain,
    dem_path: str | Path,
    mesh_crs: Any,
) -> np.ndarray:
    """Bilinearly interpolate GeoTIFF elevations onto ANUGA centroids."""
    centroid_coordinates = np.asarray(domain.centroid_coordinates, dtype=np.float64)
    with rasterio.open(dem_path) as dataset:
        if dataset.crs is None or dataset.crs != mesh_crs:
            raise ValueError(f"DEM CRS {dataset.crs} does not match mesh CRS {mesh_crs}.")
        raster = dataset.read(1).astype(np.float32)
        invalid = ~np.isfinite(raster)
        if dataset.nodata is not None:
            invalid |= np.isclose(raster, dataset.nodata)
        raster[invalid] = np.nan
        inverse = ~dataset.transform
        columns, rows = inverse * (
            centroid_coordinates[:, 0],
            centroid_coordinates[:, 1],
        )
        rows = np.asarray(rows) - 0.5
        columns = np.asarray(columns) - 0.5
        values = map_coordinates(
            raster,
            [rows, columns],
            order=1,
            mode="constant",
            cval=np.nan,
            prefilter=False,
        )
        missing = ~np.isfinite(values)
        if missing.any():
            nearest_rows = np.clip(np.rint(rows[missing]).astype(int), 0, raster.shape[0] - 1)
            nearest_columns = np.clip(np.rint(columns[missing]).astype(int), 0, raster.shape[1] - 1)
            values[missing] = raster[nearest_rows, nearest_columns]
    if not np.isfinite(values).all():
        raise ValueError(
            f"DEM interpolation returned {int((~np.isfinite(values)).sum())} "
            "invalid mesh centroid elevations."
        )
    return values.astype(np.float64)


def assign_manning_roughness(
    domain: anuga.Domain,
    roughness_lulc: gpd.GeoDataFrame,
    mesh_crs: Any,
) -> tuple[np.ndarray, dict[str, int]]:
    """Spatially assign documented Manning n values to mesh centroids."""
    required = {"lulc_class", "roughness"}
    missing_columns = required.difference(roughness_lulc.columns)
    if missing_columns:
        raise ValueError(f"Roughness layer is missing columns: {sorted(missing_columns)}")
    layer = roughness_lulc if roughness_lulc.crs == mesh_crs else roughness_lulc.to_crs(mesh_crs)
    unknown = sorted(set(layer["lulc_class"].dropna()) - set(MANNING_N_BY_LULC))
    if unknown:
        raise ValueError(f"No documented Manning lookup exists for: {unknown}")
    expected = layer["lulc_class"].map(MANNING_N_BY_LULC)
    mismatch = ~np.isclose(layer["roughness"], expected, rtol=0, atol=1.0e-9)
    if mismatch.any():
        examples = layer.loc[mismatch, ["lulc_class", "roughness"]].head().to_dict("records")
        raise ValueError(f"Source roughness values disagree with the documented lookup: {examples}")

    centroids = gpd.GeoDataFrame(
        {"element_index": np.arange(len(domain.centroid_coordinates))},
        geometry=gpd.points_from_xy(
            domain.centroid_coordinates[:, 0],
            domain.centroid_coordinates[:, 1],
        ),
        crs=mesh_crs,
    )
    joined = gpd.sjoin(
        centroids,
        layer[["lulc_class", "geometry"]],
        how="left",
        predicate="within",
    )
    if joined.index.duplicated().any():
        raise ValueError("Some mesh centroids intersect multiple LULC polygons.")
    if joined["lulc_class"].isna().any():
        raise ValueError(
            f"{int(joined['lulc_class'].isna().sum())} mesh centroids are outside LULC coverage."
        )
    joined = joined.sort_values("element_index")
    values = joined["lulc_class"].map(MANNING_N_BY_LULC).to_numpy(dtype=np.float64)
    domain.set_quantity("friction", values, location="centroids")
    counts = {str(key): int(value) for key, value in joined["lulc_class"].value_counts().items()}
    return values, counts


def build_domain(
    mesh_elements_path: str | Path,
    boundary_lines_path: str | Path,
    roughness_lulc_path: str | Path,
    dem_path: str | Path,
    *,
    initial_depth_m: float = 0.0,
    boundary_match_tolerance_m: float = 1.0,
) -> tuple[anuga.Domain, dict[str, Any]]:
    """Build and initialize the real ANUGA domain."""
    mesh = gpd.read_file(mesh_elements_path)
    boundaries = gpd.read_file(boundary_lines_path)
    roughness = gpd.read_file(roughness_lulc_path)
    if mesh.crs is None or not mesh.crs.is_projected:
        raise ValueError("Mesh must have a projected CRS.")
    if initial_depth_m < 0:
        raise ValueError("Initial depth cannot be negative.")

    LOGGER.info("Converting %d mesh triangles to ANUGA connectivity", len(mesh))
    coordinates, triangles = mesh_to_anuga_arrays(mesh)
    epsg = mesh.crs.to_epsg()
    utm_zone = mesh.crs.utm_zone
    zone_match = re.match(r"(\d+)([NS])", utm_zone or "")
    if epsg is None or zone_match is None:
        raise ValueError(f"Mesh CRS must be a recognized UTM CRS, found {mesh.crs}.")
    zone = int(zone_match.group(1))
    hemisphere = "northern" if zone_match.group(2) == "N" else "southern"
    geo_reference = Geo_reference(
        zone=zone,
        hemisphere=hemisphere,
        epsg=epsg,
        xllcorner=0.0,
        yllcorner=0.0,
    )
    domain = anuga.Domain(
        coordinates,
        triangles,
        geo_reference=geo_reference,
    )
    boundary_report = tag_exterior_edges(
        domain,
        boundaries if boundaries.crs == mesh.crs else boundaries.to_crs(mesh.crs),
        match_tolerance_m=boundary_match_tolerance_m,
    )
    elevation = interpolate_dem_at_centroids(domain, dem_path, mesh.crs)
    domain.set_quantity("elevation", elevation, location="centroids")
    domain.set_quantity("stage", elevation + initial_depth_m, location="centroids")
    domain.set_quantity("xmomentum", 0.0)
    domain.set_quantity("ymomentum", 0.0)
    roughness_values, lulc_counts = assign_manning_roughness(domain, roughness, mesh.crs)

    report = {
        "triangle_count": len(triangles),
        "node_count": len(coordinates),
        "crs": mesh.crs.to_string(),
        "elevation_min_m": float(elevation.min()),
        "elevation_max_m": float(elevation.max()),
        "initial_depth_m": initial_depth_m,
        "manning_n_min": float(roughness_values.min()),
        "manning_n_max": float(roughness_values.max()),
        "elements_by_lulc": lulc_counts,
        "boundary_tagging": boundary_report,
        "manning_reference": (
            "Chow, V. T. (1959), Open-Channel Hydraulics, McGraw-Hill, Chapter 5."
        ),
    }
    domain.hybrid_flood_build_report = report
    return domain, report
