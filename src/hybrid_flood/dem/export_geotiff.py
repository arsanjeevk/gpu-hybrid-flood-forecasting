"""Export a reproducible synthetic DEM and generation metadata."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine

DEFAULT_NODATA = -9999.0


def _json_compatible(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Affine):
        return list(value)
    if hasattr(value, "to_string"):
        return value.to_string()
    return value


def export_dem(
    elevation: np.ndarray,
    output_path: str | Path,
    *,
    transform: Affine,
    crs: Any,
    domain_mask: np.ndarray,
    nodata: float = DEFAULT_NODATA,
) -> Path:
    """Write a single-band, compressed float32 GeoTIFF."""
    if elevation.shape != domain_mask.shape:
        raise ValueError("Elevation and domain mask shapes must match.")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    output = elevation.astype(np.float32, copy=True)
    output[~domain_mask | ~np.isfinite(output)] = nodata
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=output.shape[0],
        width=output.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="deflate",
        predictor=3,
        tiled=True,
        blockxsize=512,
        blockysize=512,
        BIGTIFF="IF_SAFER",
    ) as dataset:
        dataset.write(output, 1)
        dataset.update_tags(
            AREA_OR_POINT="Area",
            ELEVATION_UNITS="metres AMSL",
            GENERATION_METHOD="synthetic physics-informed terrain",
        )
    return path


def write_generation_metadata(metadata: dict[str, Any], output_path: str | Path) -> Path:
    """Write complete generation settings and statistics as JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "generated_at": datetime.now(UTC).isoformat(),
        **_json_compatible(metadata),
    }
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
    return path
