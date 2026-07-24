"""Condition synthetic terrain using LULC as a transparent urban proxy.

Assumptions
-----------
The source layer contains broad LULC polygons, not surveyed road centrelines
or building footprints. Consequently, this module does *not* claim to recover
real structures. Built-up interiors farther than ``road_proxy_width_m`` from a
class boundary are treated as building-block proxies and raised. Narrow
built-up edges are treated as road-corridor proxies and depressed. Water is
slightly lowered to retain drainage continuity; bare/open and vegetated areas
remain close to the base surface. A fixed number of seeded, shallow circular
depressions are introduced in built-up land as explicit waterlogging
scenarios. Every offset and generated depression centre is returned as
metadata so the assumptions can be reproduced and challenged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import geopandas as gpd
import numpy as np
from affine import Affine
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt

LULC_CODES = {
    "Water": 1,
    "Built-up": 2,
    "Bare/Open": 3,
    "Low vegetation": 4,
    "Tree cover": 5,
}


@dataclass(frozen=True)
class UrbanConditioningResult:
    """Conditioned elevation and masks needed by later hydrology steps."""

    elevation: np.ndarray
    lulc_codes: np.ndarray
    intentional_depression_mask: np.ndarray
    metadata: dict[str, Any]


def rasterize_lulc(
    roughness: gpd.GeoDataFrame,
    shape: tuple[int, int],
    transform: Affine,
    target_crs: Any,
) -> np.ndarray:
    """Rasterize normalized LULC class names to stable integer codes."""
    if "lulc_class" not in roughness.columns:
        raise ValueError("Roughness layer must contain the 'lulc_class' column.")
    layer = roughness if roughness.crs == target_crs else roughness.to_crs(target_crs)
    unknown = sorted(set(layer["lulc_class"].dropna()) - set(LULC_CODES))
    if unknown:
        raise ValueError(f"Unknown LULC classes: {unknown}")
    shapes = (
        (geometry, LULC_CODES[class_name])
        for geometry, class_name in zip(layer.geometry, layer["lulc_class"], strict=True)
        if geometry is not None and not geometry.is_empty and class_name in LULC_CODES
    )
    return rasterize(
        shapes,
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=False,
    )


def _add_intentional_depressions(
    elevation: np.ndarray,
    eligible_mask: np.ndarray,
    *,
    transform: Affine,
    resolution_m: float,
    count: int,
    radius_m: float,
    depth_m: float,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    depression_mask = np.zeros(elevation.shape, dtype=bool)
    eligible_rows, eligible_columns = np.nonzero(eligible_mask)
    if count <= 0 or eligible_rows.size == 0 or radius_m <= 0 or depth_m <= 0:
        return depression_mask, []

    rng = np.random.default_rng(seed)
    order = rng.permutation(eligible_rows.size)
    centers: list[dict[str, float]] = []
    radius_cells = max(1, int(np.ceil(radius_m / resolution_m)))
    minimum_spacing_squared = (2.0 * radius_m) ** 2

    for candidate in order:
        row = int(eligible_rows[candidate])
        column = int(eligible_columns[candidate])
        x, y = transform * (column + 0.5, row + 0.5)
        if any(
            (x - center["x"]) ** 2 + (y - center["y"]) ** 2 < minimum_spacing_squared
            for center in centers
        ):
            continue

        row_min = max(0, row - radius_cells)
        row_max = min(elevation.shape[0], row + radius_cells + 1)
        column_min = max(0, column - radius_cells)
        column_max = min(elevation.shape[1], column + radius_cells + 1)
        local_rows, local_columns = np.ogrid[row_min:row_max, column_min:column_max]
        distance_m = np.sqrt(
            ((local_rows - row) * resolution_m) ** 2
            + ((local_columns - column) * resolution_m) ** 2
        )
        local_mask = (distance_m <= radius_m) & eligible_mask[
            row_min:row_max, column_min:column_max
        ]
        bowl = depth_m * np.maximum(0.0, 1.0 - distance_m / max(radius_m, resolution_m))
        elevation[row_min:row_max, column_min:column_max][local_mask] -= bowl[local_mask]
        depression_mask[row_min:row_max, column_min:column_max] |= local_mask
        centers.append({"x": float(x), "y": float(y), "depth_m": depth_m, "radius_m": radius_m})
        if len(centers) == count:
            break
    return depression_mask, centers


def condition_urban_terrain(
    base_elevation: np.ndarray,
    domain_mask: np.ndarray,
    roughness: gpd.GeoDataFrame,
    *,
    transform: Affine,
    crs: Any,
    resolution_m: float,
    building_raise_m: float = 0.6,
    road_depression_m: float = 0.15,
    road_proxy_width_m: float = 10.0,
    water_lowering_m: float = 0.35,
    tree_raise_m: float = 0.05,
    intentional_depression_count: int = 8,
    intentional_depression_radius_m: float = 40.0,
    intentional_depression_depth_m: float = 0.3,
    seed: int = 42,
) -> UrbanConditioningResult:
    """Burn defensible LULC-based microtopographic offsets into the DEM."""
    if base_elevation.shape != domain_mask.shape:
        raise ValueError("Elevation and domain mask shapes must match.")
    if any(
        value < 0
        for value in (
            building_raise_m,
            road_depression_m,
            road_proxy_width_m,
            water_lowering_m,
            tree_raise_m,
            intentional_depression_count,
            intentional_depression_radius_m,
            intentional_depression_depth_m,
        )
    ):
        raise ValueError("Urban conditioning magnitudes and counts must be non-negative.")

    lulc = rasterize_lulc(roughness, base_elevation.shape, transform, crs)
    elevation = base_elevation.astype(np.float32, copy=True)
    built_up = (lulc == LULC_CODES["Built-up"]) & domain_mask
    distance_inside_built_up = distance_transform_edt(built_up, sampling=resolution_m)
    road_proxy = built_up & (distance_inside_built_up <= road_proxy_width_m)
    building_proxy = built_up & ~road_proxy

    elevation[road_proxy] -= road_depression_m
    elevation[building_proxy] += building_raise_m
    elevation[(lulc == LULC_CODES["Water"]) & domain_mask] -= water_lowering_m
    elevation[(lulc == LULC_CODES["Tree cover"]) & domain_mask] += tree_raise_m

    intentional_mask, centers = _add_intentional_depressions(
        elevation,
        built_up,
        transform=transform,
        resolution_m=resolution_m,
        count=intentional_depression_count,
        radius_m=intentional_depression_radius_m,
        depth_m=intentional_depression_depth_m,
        seed=seed,
    )
    elevation[~domain_mask] = np.nan
    class_counts = {
        class_name: int(((lulc == code) & domain_mask).sum())
        for class_name, code in LULC_CODES.items()
    }
    metadata = {
        "method": "LULC-proxy urban microtopographic conditioning",
        "assumptions": [
            "The LULC layer contains no surveyed road centrelines or building footprints.",
            "Built-up interiors proxy raised building blocks.",
            "Built-up class edges proxy narrow depressed road corridors.",
            "Water is lowered slightly; bare/open and low vegetation retain base elevation.",
            "Seeded shallow built-up depressions represent explicit waterlogging scenarios.",
        ],
        "building_raise_m": building_raise_m,
        "road_depression_m": road_depression_m,
        "road_proxy_width_m": road_proxy_width_m,
        "water_lowering_m": water_lowering_m,
        "tree_raise_m": tree_raise_m,
        "intentional_depression_count_requested": intentional_depression_count,
        "intentional_depression_count_created": len(centers),
        "intentional_depression_radius_m": intentional_depression_radius_m,
        "intentional_depression_depth_m": intentional_depression_depth_m,
        "intentional_depression_seed": seed,
        "intentional_depression_centers": centers,
        "lulc_codes": LULC_CODES,
        "domain_cell_counts_by_lulc": class_counts,
        "unclassified_domain_cells": int(((lulc == 0) & domain_mask).sum()),
        "road_proxy_cell_count": int(road_proxy.sum()),
        "building_proxy_cell_count": int(building_proxy.sum()),
    }
    return UrbanConditioningResult(elevation, lulc, intentional_mask, metadata)
