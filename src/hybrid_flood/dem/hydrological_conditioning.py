"""Condition drainage with PyFlwDir while preserving explicit flood scenarios.

RichDEM was originally requested for this stage, but its latest PyPI release
requires NumPy <2 while ANUGA >=3.2 requires NumPy >=2. PyFlwDir implements
Wang--Liu depression filling, supports NumPy 2, and is therefore used as the
compatible hydrological backend. This substitution is recorded in metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pyflwdir
from affine import Affine
from pyflwdir import dem

NODATA = -9999.0
Aggressiveness = Literal["none", "selective", "full"]


@dataclass(frozen=True)
class HydrologicalConditioningResult:
    """Conditioned DEM and reproducibility diagnostics."""

    elevation: np.ndarray
    d8: np.ndarray
    fill_depth: np.ndarray
    metadata: dict[str, Any]


def _as_nodata(elevation: np.ndarray, domain_mask: np.ndarray) -> np.ndarray:
    data = elevation.astype(np.float32, copy=True)
    data[~domain_mask | ~np.isfinite(data)] = NODATA
    return data


def find_unfilled_sink_mask(
    elevation: np.ndarray,
    domain_mask: np.ndarray,
    *,
    tolerance_m: float = 1.0e-4,
) -> np.ndarray:
    """Return cells that a complete depression-fill operation would raise."""
    source = _as_nodata(elevation, domain_mask)
    completely_filled, _ = dem.fill_depressions(source, nodata=NODATA, outlets="edge")
    return domain_mask & ((completely_filled - source) > tolerance_m)


def condition_hydrology(
    elevation: np.ndarray,
    domain_mask: np.ndarray,
    *,
    intentional_depression_mask: np.ndarray | None = None,
    aggressiveness: Aggressiveness = "selective",
    preserve_intentional_depressions: bool = True,
    tolerance_m: float = 1.0e-4,
) -> HydrologicalConditioningResult:
    """Fill spurious sinks according to a documented aggressiveness policy.

    ``none`` leaves all elevations unchanged. ``selective`` fills every
    algorithmic sink, then restores cells belonging to the explicit
    waterlogging mask. ``full`` fills every sink, including intentional
    depressions. This makes the trade-off between hydraulic connectivity and
    preservation of scenario features explicit.
    """
    if aggressiveness not in {"none", "selective", "full"}:
        raise ValueError("aggressiveness must be one of: none, selective, full")
    if elevation.shape != domain_mask.shape:
        raise ValueError("Elevation and domain mask shapes must match.")
    intentional = (
        np.zeros(elevation.shape, dtype=bool)
        if intentional_depression_mask is None
        else intentional_depression_mask.astype(bool, copy=False)
    )
    if intentional.shape != elevation.shape:
        raise ValueError("Intentional depression mask must match elevation shape.")

    source = _as_nodata(elevation, domain_mask)
    completely_filled, _ = dem.fill_depressions(source, nodata=NODATA, outlets="edge")
    if aggressiveness == "none":
        conditioned = source.copy()
    else:
        conditioned = completely_filled.copy()
        if aggressiveness == "selective" and preserve_intentional_depressions:
            conditioned[intentional & domain_mask] = source[intentional & domain_mask]

    # D8 used for diagnostics is derived from the fully connected surface. The
    # reported output elevations still retain intentional selective depressions.
    _, d8 = dem.fill_depressions(conditioned, nodata=NODATA, outlets="edge")
    fill_depth = np.zeros(elevation.shape, dtype=np.float32)
    fill_depth[domain_mask] = completely_filled[domain_mask] - source[domain_mask]
    output = conditioned.astype(np.float32)
    output[~domain_mask] = np.nan

    remaining = find_unfilled_sink_mask(output, domain_mask, tolerance_m=tolerance_m)
    unexpected_remaining = remaining & ~intentional
    metadata = {
        "method": "Wang-Liu depression filling",
        "backend": f"pyflwdir {pyflwdir.__version__}",
        "backend_substitution_reason": (
            "RichDEM 0.3.4 requires NumPy <2 and conflicts with ANUGA >=3.2."
        ),
        "aggressiveness": aggressiveness,
        "preserve_intentional_depressions": preserve_intentional_depressions,
        "fill_tolerance_m": tolerance_m,
        "cells_raised": int((fill_depth > tolerance_m).sum()),
        "maximum_fill_depth_m": float(fill_depth.max(initial=0.0)),
        "remaining_sink_cells": int(remaining.sum()),
        "unexpected_remaining_sink_cells": int(unexpected_remaining.sum()),
        "intentional_remaining_sink_cells": int((remaining & intentional).sum()),
    }
    return HydrologicalConditioningResult(output, d8, fill_depth, metadata)


def flow_accumulation(
    elevation: np.ndarray,
    domain_mask: np.ndarray,
    transform: Affine,
) -> np.ndarray:
    """Calculate D8 upstream contributing area in raster cells."""
    source = _as_nodata(elevation, domain_mask)
    flow_direction = pyflwdir.from_dem(
        source,
        nodata=NODATA,
        transform=transform,
        latlon=False,
        outlets="edge",
    )
    accumulation = flow_direction.upstream_area(unit="cell").astype(np.float32)
    accumulation[~domain_mask] = np.nan
    return accumulation
