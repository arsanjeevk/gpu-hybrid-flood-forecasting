"""Run and benchmark an ANUGA baseline simulation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import anuga
import netCDF4 as _netcdf4  # noqa: F401  # Load SWW backend before ANUGA evolves.

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SimulationResult:
    """Paths and diagnostics produced by one ANUGA run."""

    sww_path: Path
    metrics_path: Path
    run_name: str
    runtime_seconds: float
    step_metrics: list[dict[str, float]]
    maximum_absolute_mass_error_m3: float
    maximum_relative_mass_error: float
    final_mass_balance_error_m3: float


def configure_numerics(
    domain: anuga.Domain,
    *,
    flow_algorithm: str = "DE1",
    cfl: float = 0.9,
    minimum_storable_height_m: float = 0.001,
) -> None:
    """Apply explicit numerical settings used by the baseline."""
    domain.set_flow_algorithm(flow_algorithm)
    domain.set_cfl(cfl)
    domain.set_minimum_storable_height(minimum_storable_height_m)
    domain.set_quantities_to_be_stored(
        {
            "elevation": 1,
            "stage": 2,
            "xmomentum": 2,
            "ymomentum": 2,
        }
    )


def run_simulation(
    domain: anuga.Domain,
    output_directory: str | Path,
    *,
    duration_s: float,
    yieldstep_s: float,
    outputstep_s: float,
    flow_algorithm: str = "DE1",
    cfl: float = 0.9,
    minimum_storable_height_m: float = 0.001,
    run_timestamp: datetime | None = None,
) -> SimulationResult:
    """Evolve a configured domain and log exact ANUGA volume-balance terms."""
    if duration_s <= 0 or yieldstep_s <= 0 or outputstep_s <= 0:
        raise ValueError("Duration, yieldstep, and outputstep must be positive.")
    ratio = outputstep_s / yieldstep_s
    if not abs(ratio - round(ratio)) < 1.0e-9:
        raise ValueError("outputstep_s must be an integer multiple of yieldstep_s.")

    configure_numerics(
        domain,
        flow_algorithm=flow_algorithm,
        cfl=cfl,
        minimum_storable_height_m=minimum_storable_height_m,
    )
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = run_timestamp or datetime.now(UTC)
    timestamp_text = timestamp.strftime("%Y%m%dT%H%M%S%fZ")
    run_name = f"run_{timestamp_text}"
    domain.set_name(run_name)
    domain.set_datadir(str(output_dir))
    domain.set_store(True)

    initial_volume = float(domain.get_water_volume())
    metrics_path = output_dir / f"{run_name}_metrics.jsonl"
    step_metrics: list[dict[str, float]] = []
    run_start = perf_counter()
    previous_yield = run_start

    with metrics_path.open("w", encoding="utf-8") as metrics_stream:
        for simulation_time in domain.evolve(
            yieldstep=yieldstep_s,
            outputstep=outputstep_s,
            duration=duration_s,
        ):
            now = perf_counter()
            water_volume = float(domain.get_water_volume())
            boundary_flux = float(domain.get_boundary_flux_integral())
            forcing_volume = float(domain.get_fractional_step_volume_integral())
            mass_error = water_volume - boundary_flux - forcing_volume - initial_volume
            scale = max(
                abs(initial_volume) + abs(boundary_flux) + abs(forcing_volume),
                1.0,
            )
            metric = {
                "simulation_time_s": float(simulation_time),
                "wall_step_seconds": now - previous_yield,
                "wall_elapsed_seconds": now - run_start,
                "water_volume_m3": water_volume,
                "boundary_flux_integral_m3": boundary_flux,
                "forcing_volume_integral_m3": forcing_volume,
                "mass_balance_error_m3": mass_error,
                "relative_mass_balance_error": mass_error / scale,
            }
            step_metrics.append(metric)
            metrics_stream.write(json.dumps(metric, sort_keys=True) + "\n")
            metrics_stream.flush()
            LOGGER.info(
                "ANUGA t=%8.2fs wall=%7.3fs volume=%12.4fm3 mass_error=%+.6em3",
                simulation_time,
                metric["wall_step_seconds"],
                water_volume,
                mass_error,
            )
            previous_yield = now

    runtime = perf_counter() - run_start
    sww_path = output_dir / f"{run_name}.sww"
    if not sww_path.is_file():
        raise FileNotFoundError(f"ANUGA did not create the expected SWW output: {sww_path}")
    maximum_error = max(
        (abs(metric["mass_balance_error_m3"]) for metric in step_metrics),
        default=0.0,
    )
    maximum_relative_error = max(
        (abs(metric["relative_mass_balance_error"]) for metric in step_metrics),
        default=0.0,
    )
    final_mass_error = step_metrics[-1]["mass_balance_error_m3"] if step_metrics else 0.0
    LOGGER.info(
        "ANUGA run complete in %.3fs; maximum absolute mass error %.6e m3",
        runtime,
        maximum_error,
    )
    return SimulationResult(
        sww_path=sww_path,
        metrics_path=metrics_path,
        run_name=run_name,
        runtime_seconds=runtime,
        step_metrics=step_metrics,
        maximum_absolute_mass_error_m3=maximum_error,
        maximum_relative_mass_error=maximum_relative_error,
        final_mass_balance_error_m3=final_mass_error,
    )
