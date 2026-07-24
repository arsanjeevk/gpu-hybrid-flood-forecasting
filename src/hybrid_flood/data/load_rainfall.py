"""Parse the raw rainfall workbook into a tidy, unit-aware time series."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import xarray as xr

RAINFALL_UNITS = "mm/hr"
TIME_COLUMN_CANDIDATES = ("timestamp", "datetime", "date_time", "time", "hour")


def _find_time_column(columns: pd.Index) -> Any:
    normalized = {str(column).strip().lower(): column for column in columns}
    for candidate in TIME_COLUMN_CANDIDATES:
        if candidate in normalized:
            return normalized[candidate]
    raise ValueError(
        "Rainfall workbook must contain a time column named one of "
        f"{list(TIME_COLUMN_CANDIDATES)}; found {list(columns)}."
    )


def _parse_timestamps(
    values: pd.Series,
    column_name: str,
    reference_start: str | pd.Timestamp,
) -> pd.DatetimeIndex:
    numeric = pd.to_numeric(values, errors="coerce")
    if column_name.strip().lower() == "hour" and numeric.notna().all():
        start = pd.Timestamp(reference_start)
        return pd.DatetimeIndex(start + pd.to_timedelta(numeric, unit="h"), name="timestamp")

    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        bad_rows = parsed[parsed.isna()].index.tolist()[:10]
        raise ValueError(f"Unparseable rainfall timestamps at workbook rows {bad_rows}.")
    return pd.DatetimeIndex(parsed, name="timestamp")


def load_rainfall(
    workbook_path: str | Path,
    *,
    sheet_name: str | int = 0,
    reference_start: str = "2000-01-01T00:00:00",
    frequency: str = "1h",
) -> pd.DataFrame:
    """Load rainfall scenarios and insert any missing timestamps as explicit rows."""
    path = Path(workbook_path)
    if not path.is_file():
        raise FileNotFoundError(f"Rainfall workbook does not exist: {path}")

    wide = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
    if wide.empty:
        raise ValueError(f"Rainfall workbook is empty: {path}")
    wide.columns = [str(column).strip() for column in wide.columns]
    time_column = _find_time_column(wide.columns)
    timestamps = _parse_timestamps(wide[time_column], str(time_column), reference_start)

    if timestamps.has_duplicates:
        duplicates = timestamps[timestamps.duplicated()].unique().astype(str).tolist()
        raise ValueError(f"Duplicate rainfall timestamps found: {duplicates[:10]}")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("Rainfall timestamps must be monotonically increasing.")

    rainfall_columns = [column for column in wide.columns if column != time_column]
    if not rainfall_columns:
        raise ValueError("Rainfall workbook contains no rainfall value columns.")
    values = wide[rainfall_columns].apply(pd.to_numeric, errors="coerce")
    non_numeric = wide[rainfall_columns].notna() & values.isna()
    if non_numeric.any().any():
        locations = [
            f"{column}[{row}]"
            for row, column in zip(*non_numeric.to_numpy().nonzero(), strict=False)
        ]
        raise ValueError(f"Non-numeric rainfall values found at {locations[:10]}.")
    if (values < 0).any().any():
        raise ValueError("Rainfall intensity cannot be negative.")

    values.index = timestamps
    expected_index = pd.date_range(
        start=timestamps.min(), end=timestamps.max(), freq=frequency, name="timestamp"
    )
    values = values.reindex(expected_index)
    missing_timestamp = ~values.index.isin(timestamps)

    tidy = (
        values.rename_axis(columns="scenario")
        .stack(future_stack=True)
        .rename("rainfall_mm_hr")
        .reset_index()
    )
    tidy["scenario"] = tidy["scenario"].astype(str)
    missing_lookup = pd.Series(missing_timestamp, index=expected_index)
    tidy["is_missing_timestamp"] = tidy["timestamp"].map(missing_lookup).astype(bool)
    tidy["units"] = RAINFALL_UNITS
    tidy.attrs.update(
        {
            "rainfall_units": RAINFALL_UNITS,
            "frequency": frequency,
            "reference_start": str(reference_start),
            "source": str(path),
        }
    )
    return tidy


def rainfall_to_xarray(rainfall: pd.DataFrame) -> xr.Dataset:
    """Convert tidy rainfall data to an xarray Dataset."""
    required = {"timestamp", "scenario", "rainfall_mm_hr"}
    missing = required.difference(rainfall.columns)
    if missing:
        raise ValueError(f"Rainfall table is missing columns: {sorted(missing)}")
    dataset = rainfall.set_index(["timestamp", "scenario"])[["rainfall_mm_hr"]].to_xarray()
    dataset["rainfall_mm_hr"].attrs["units"] = RAINFALL_UNITS
    dataset.attrs["description"] = "Cleaned urban rainfall forcing scenarios"
    return dataset


def save_clean_rainfall(rainfall: pd.DataFrame, output_path: str | Path) -> Path:
    """Persist tidy rainfall data as Parquet with units stored in the schema."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rainfall.to_parquet(path, index=False, engine="pyarrow")
    return path


def clean_rainfall_workbook(
    workbook_path: str | Path,
    output_path: str | Path,
    *,
    sheet_name: str | int = 0,
    reference_start: str = "2000-01-01T00:00:00",
    frequency: str = "1h",
) -> tuple[pd.DataFrame, xr.Dataset]:
    """Load, validate, save, and return rainfall in pandas and xarray forms."""
    rainfall = load_rainfall(
        workbook_path,
        sheet_name=sheet_name,
        reference_start=reference_start,
        frequency=frequency,
    )
    save_clean_rainfall(rainfall, output_path)
    return rainfall, rainfall_to_xarray(rainfall)
