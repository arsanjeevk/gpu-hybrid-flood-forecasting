"""Build temporally separated ANUGA-to-JAX residual-learning samples.

Samples are split into contiguous train, validation, and test blocks in time,
never by randomly shuffling timesteps. Consecutive flood states are strongly
autocorrelated, so a random sample split would put near-duplicates on both
sides of the evaluation boundary and produce still more optimistic metrics.
This single-event temporal holdout reduces direct leakage but is not equivalent
to validation on independent storm events. Spatial patches may be shuffled
within the training block.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from rasterio.features import rasterize
from rasterio.transform import from_origin

INPUT_CHANNELS = (
    "jax_depth_t",
    "jax_x_velocity_t",
    "jax_y_velocity_t",
    "rainfall_t",
    "terrain",
    "roughness",
)
TARGET_CHANNELS = (
    "depth_residual_t_plus_1",
    "x_velocity_residual_t_plus_1",
    "y_velocity_residual_t_plus_1",
)


@dataclass(frozen=True)
class ResidualDataset:
    """Compact physical-unit arrays used to construct CNN batches."""

    state_t: np.ndarray
    target_residual: np.ndarray
    raw_depth_t_plus_1: np.ndarray
    rainfall_t_mm_hr: np.ndarray
    terrain_standardized: np.ndarray
    roughness_standardized: np.ndarray
    domain_mask: np.ndarray
    loss_mask: np.ndarray
    input_time_s: np.ndarray
    target_time_s: np.ndarray
    x: np.ndarray
    y: np.ndarray
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    metadata: dict[str, Any]

    @property
    def spatial_shape(self) -> tuple[int, int]:
        """Return ``(height, width)``."""
        return self.terrain_standardized.shape

    @property
    def input_channels(self) -> int:
        """Return the number of materialized network input channels."""
        return len(INPUT_CHANNELS)

    def inputs(self, indices: np.ndarray | list[int]) -> np.ndarray:
        """Materialize state, forcing, and static channels for selected samples."""
        selected = np.asarray(indices, dtype=np.int64)
        state_mean = np.asarray(
            self.metadata["normalization"]["state_mean"],
            dtype=np.float32,
        )
        state_std = np.asarray(
            self.metadata["normalization"]["state_std"],
            dtype=np.float32,
        )
        state = (self.state_t[selected] - state_mean) / state_std
        state[:, ~self.domain_mask, :] = 0.0
        count, height, width, _ = state.shape
        rainfall_scale = float(self.metadata["normalization"]["rainfall_scale_mm_hr"])
        rainfall = np.broadcast_to(
            (self.rainfall_t_mm_hr[selected] / rainfall_scale)[:, None, None, None],
            (count, height, width, 1),
        )
        terrain = np.broadcast_to(
            self.terrain_standardized[None, :, :, None],
            (count, height, width, 1),
        )
        roughness = np.broadcast_to(
            self.roughness_standardized[None, :, :, None],
            (count, height, width, 1),
        )
        return np.concatenate((state, rainfall, terrain, roughness), axis=-1).astype(
            np.float32,
            copy=False,
        )


def _validate_ratios(train_fraction: float, val_fraction: float) -> None:
    if not 0.0 < train_fraction < 1.0 or not 0.0 < val_fraction < 1.0:
        raise ValueError("Train and validation fractions must each lie between zero and one.")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("Train and validation fractions must leave a non-empty test fraction.")


def temporal_split_indices(
    sample_count: int,
    *,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return disjoint contiguous indices for a development temporal holdout."""
    _validate_ratios(train_fraction, val_fraction)
    if sample_count < 3:
        raise ValueError("At least three transition samples are required for temporal splitting.")
    train_end = max(1, int(np.floor(sample_count * train_fraction)))
    val_end = max(train_end + 1, int(np.floor(sample_count * (train_fraction + val_fraction))))
    val_end = min(val_end, sample_count - 1)
    return (
        np.arange(0, train_end, dtype=np.int32),
        np.arange(train_end, val_end, dtype=np.int32),
        np.arange(val_end, sample_count, dtype=np.int32),
    )


def _standardize_static(values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float, float]:
    valid = values[mask]
    mean = float(valid.mean())
    std = float(valid.std())
    if not np.isfinite(std) or std < 1.0e-8:
        std = 1.0
    standardized = np.zeros_like(values, dtype=np.float32)
    standardized[mask] = (values[mask] - mean) / std
    return standardized, mean, std


def _rasterize_roughness(
    roughness_path: str | Path,
    *,
    x: np.ndarray,
    y: np.ndarray,
    crs: str,
    mask: np.ndarray,
) -> np.ndarray:
    roughness = gpd.read_file(roughness_path)
    if "roughness" not in roughness.columns:
        raise ValueError("Roughness layer must contain a 'roughness' attribute.")
    if roughness.crs is None:
        raise ValueError("Roughness layer has no CRS.")
    if str(roughness.crs) != crs:
        roughness = roughness.to_crs(crs)
    dx = float(np.median(np.diff(x)))
    dy = float(np.median(np.diff(y)))
    if not np.isclose(dx, dy):
        raise ValueError("Only square regular-grid cells are supported.")
    transform = from_origin(x[0] - dx / 2, y[-1] + dy / 2, dx, dy)
    north_up = rasterize(
        (
            (geometry, float(value))
            for geometry, value in zip(
                roughness.geometry,
                roughness["roughness"],
                strict=True,
            )
            if geometry is not None and not geometry.is_empty and np.isfinite(value)
        ),
        out_shape=mask.shape,
        transform=transform,
        fill=np.nan,
        dtype=np.float32,
    )
    values = np.flipud(north_up).copy()
    missing = mask & ~np.isfinite(values)
    if missing.any():
        raise ValueError(f"Roughness rasterization left {int(missing.sum())} domain cells empty.")
    values[~mask] = 0.0
    return values


def _rainfall_at_times(
    rainfall_path: str | Path,
    *,
    scenario: str,
    times_s: np.ndarray,
) -> np.ndarray:
    table = pd.read_parquet(rainfall_path)
    required = {"timestamp", "scenario", "rainfall_mm_hr", "units"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Rainfall table is missing columns: {sorted(missing)}")
    selected = table.loc[table["scenario"] == scenario].sort_values("timestamp")
    if selected.empty:
        raise ValueError(f"Rainfall scenario {scenario!r} was not found.")
    if set(selected["units"].dropna().astype(str)) != {"mm/hr"}:
        raise ValueError("Rainfall forcing must have explicit units of mm/hr.")
    timestamps = pd.to_datetime(selected["timestamp"])
    source_times = (timestamps - timestamps.iloc[0]).dt.total_seconds().to_numpy(np.float64)
    source_rates = selected["rainfall_mm_hr"].to_numpy(np.float64)
    if not np.isfinite(source_rates).all():
        raise ValueError("Rainfall forcing contains non-finite values.")
    return np.interp(times_s, source_times, source_rates).astype(np.float32)


def _aligned_anuga(anuga: xr.Dataset, jax_raw: xr.Dataset) -> xr.Dataset:
    required = {"depth", "x_velocity", "y_velocity", "elevation"}
    for name, dataset in (("ANUGA", anuga), ("JAX", jax_raw)):
        missing = required.difference(dataset.data_vars)
        if missing:
            raise ValueError(f"{name} dataset is missing variables: {sorted(missing)}")
    if str(anuga.attrs.get("crs")) != str(jax_raw.attrs.get("crs")):
        raise ValueError("ANUGA and JAX products must use the same CRS.")
    target_times = jax_raw.time.values
    if target_times[0] < anuga.time.values[0] or target_times[-1] > anuga.time.values[-1]:
        raise ValueError("JAX times extend beyond the ANUGA reference period.")
    if (
        np.array_equal(anuga.time.values, target_times)
        and np.array_equal(anuga.x.values, jax_raw.x.values)
        and np.array_equal(anuga.y.values, jax_raw.y.values)
    ):
        return anuga
    return anuga.interp(
        time=jax_raw.time,
        x=jax_raw.x,
        y=jax_raw.y,
        method="linear",
    )


def build_residual_dataset(
    anuga_path: str | Path,
    jax_path: str | Path,
    roughness_path: str | Path,
    rainfall_path: str | Path,
    *,
    rainfall_scenario: str,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    permanently_dry_threshold_m: float = 1.0e-4,
) -> ResidualDataset:
    """Build physical residual targets for transitions from time ``t`` to ``t+1``."""
    if permanently_dry_threshold_m < 0:
        raise ValueError("The permanently-dry threshold cannot be negative.")
    with xr.open_dataset(anuga_path) as anuga_source, xr.open_dataset(jax_path) as jax_source:
        anuga = _aligned_anuga(anuga_source, jax_source)
        times = np.asarray(jax_source.time.values, dtype=np.float64)
        if len(times) < 4 or not np.all(np.diff(times) > 0):
            raise ValueError("Solver time coordinates must contain at least four increasing times.")

        jax_state = np.stack(
            (
                jax_source["depth"].values,
                jax_source["x_velocity"].values,
                jax_source["y_velocity"].values,
            ),
            axis=-1,
        ).astype(np.float32)
        reference_state = np.stack(
            (
                anuga["depth"].values,
                anuga["x_velocity"].values,
                anuga["y_velocity"].values,
            ),
            axis=-1,
        ).astype(np.float32)
        terrain = np.asarray(jax_source["elevation"].values, dtype=np.float32)
        reference_terrain = np.asarray(anuga["elevation"].values, dtype=np.float32)
        x = np.asarray(jax_source.x.values, dtype=np.float64)
        y = np.asarray(jax_source.y.values, dtype=np.float64)
        crs = str(jax_source.attrs.get("crs"))

    domain_mask = np.isfinite(terrain) & np.isfinite(reference_terrain)
    finite_dynamic = np.isfinite(jax_state).all(axis=(0, 3)) & np.isfinite(reference_state).all(
        axis=(0, 3)
    )
    domain_mask &= finite_dynamic
    if not domain_mask.any():
        raise ValueError("ANUGA and JAX products have no common finite domain cells.")

    residual = reference_state[1:] - jax_state[1:]
    state_t = jax_state[:-1]
    target_count = len(times) - 1
    train, val, test = temporal_split_indices(
        target_count,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
    )
    reference_depth = reference_state[1:, :, :, 0]
    # Derive the activity/loss mask from training labels only. Using maximum
    # depth across the full series leaks validation/test inundation extent into
    # training even when the sample indices themselves are chronological.
    maximum_training_reference_depth = np.max(
        np.where(np.isfinite(reference_depth[train]), reference_depth[train], -np.inf),
        axis=0,
    )
    loss_mask = domain_mask & (maximum_training_reference_depth > permanently_dry_threshold_m)
    if not loss_mask.any():
        raise ValueError("No cells exceed the permanently-dry depth threshold.")

    state_t[:, ~domain_mask, :] = 0.0
    residual[:, ~domain_mask, :] = 0.0
    terrain[~domain_mask] = 0.0
    roughness = _rasterize_roughness(
        roughness_path,
        x=x,
        y=y,
        crs=crs,
        mask=domain_mask,
    )
    terrain_standardized, terrain_mean, terrain_std = _standardize_static(terrain, domain_mask)
    roughness_standardized, roughness_mean, roughness_std = _standardize_static(
        roughness,
        domain_mask,
    )
    rainfall = _rainfall_at_times(
        rainfall_path,
        scenario=rainfall_scenario,
        times_s=times[:-1],
    )
    rainfall_scale = max(float(np.max(np.abs(rainfall[train]))), 1.0)
    training_state = state_t[train][:, loss_mask, :]
    state_mean = training_state.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    state_std = training_state.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    state_std = np.maximum(state_std, np.float32(1.0e-6))
    training_target = residual[train][:, loss_mask, :]
    target_std = training_target.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    target_std = np.maximum(target_std, np.float32(1.0e-6))

    metadata: dict[str, Any] = {
        "description": "ANUGA minus raw-JAX one-step residual correction dataset",
        "crs": crs,
        "input_channels": list(INPUT_CHANNELS),
        "target_channels": list(TARGET_CHANNELS),
        "state_units": ["m", "m/s", "m/s"],
        "target_units": ["m", "m/s", "m/s"],
        "rainfall_units": "mm/hr",
        "sample_semantics": "state and rainfall at t predict ANUGA-JAX residual at t+1",
        "split_policy": (
            "disjoint contiguous chronological blocks; no temporal shuffling; "
            "single-event development holdout, not independent-event validation"
        ),
        "split_counts": {
            "train": int(len(train)),
            "validation": int(len(val)),
            "test": int(len(test)),
        },
        "permanently_dry_threshold_m": permanently_dry_threshold_m,
        "loss_mask_basis": "maximum reference depth in training time block only",
        "domain_cell_count": int(domain_mask.sum()),
        "loss_mask_cell_count": int(loss_mask.sum()),
        "normalization": {
            "terrain_mean_m": terrain_mean,
            "terrain_std_m": terrain_std,
            "roughness_mean": roughness_mean,
            "roughness_std": roughness_std,
            "rainfall_scale_mm_hr": rainfall_scale,
            "state_mean": state_mean.tolist(),
            "state_std": state_std.tolist(),
            "target_residual_std": target_std.tolist(),
        },
        "sources": {
            "anuga": str(Path(anuga_path).resolve()),
            "jax": str(Path(jax_path).resolve()),
            "roughness": str(Path(roughness_path).resolve()),
            "rainfall": str(Path(rainfall_path).resolve()),
            "rainfall_scenario": rainfall_scenario,
        },
    }
    return ResidualDataset(
        state_t=state_t,
        target_residual=residual,
        raw_depth_t_plus_1=jax_state[1:, :, :, 0],
        rainfall_t_mm_hr=rainfall,
        terrain_standardized=terrain_standardized,
        roughness_standardized=roughness_standardized,
        domain_mask=domain_mask,
        loss_mask=loss_mask,
        input_time_s=times[:-1],
        target_time_s=times[1:],
        x=x,
        y=y,
        train_indices=train,
        val_indices=val,
        test_indices=test,
        metadata=metadata,
    )


def save_residual_dataset(dataset: ResidualDataset, output_path: str | Path) -> Path:
    """Save the compact dataset as compressed NumPy arrays plus JSON metadata."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        state_t=dataset.state_t,
        target_residual=dataset.target_residual,
        raw_depth_t_plus_1=dataset.raw_depth_t_plus_1,
        rainfall_t_mm_hr=dataset.rainfall_t_mm_hr,
        terrain_standardized=dataset.terrain_standardized,
        roughness_standardized=dataset.roughness_standardized,
        domain_mask=dataset.domain_mask,
        loss_mask=dataset.loss_mask,
        input_time_s=dataset.input_time_s,
        target_time_s=dataset.target_time_s,
        x=dataset.x,
        y=dataset.y,
        train_indices=dataset.train_indices,
        val_indices=dataset.val_indices,
        test_indices=dataset.test_indices,
    )
    path.with_suffix(".json").write_text(
        json.dumps(dataset.metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def load_residual_dataset(path: str | Path) -> ResidualDataset:
    """Load a residual dataset without allowing pickled object arrays."""
    archive_path = Path(path)
    metadata_path = archive_path.with_suffix(".json")
    if not archive_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Dataset archive or metadata is missing for {archive_path}.")
    with np.load(archive_path, allow_pickle=False) as arrays:
        values = {name: arrays[name] for name in arrays.files}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return ResidualDataset(metadata=metadata, **values)
