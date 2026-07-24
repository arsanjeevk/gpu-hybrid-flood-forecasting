"""Cross-solver consistency test on a controlled local-inflow basin."""

from __future__ import annotations

import anuga
import numpy as np
from affine import Affine
from pyproj import CRS

from hybrid_flood.jax_solver.shallow_water_2d import ShallowWaterSolver, StructuredGrid


def _triangular_square_mesh(
    rows: int,
    columns: int,
    spacing: float,
) -> tuple[list[list[float]], list[list[int]], dict[tuple[int, int], list[int]]]:
    coordinates = [
        [column * spacing, row * spacing]
        for row in range(rows + 1)
        for column in range(columns + 1)
    ]
    triangles: list[list[int]] = []
    cell_triangles: dict[tuple[int, int], list[int]] = {}
    for row in range(rows):
        for column in range(columns):
            lower_left = row * (columns + 1) + column
            lower_right = lower_left + 1
            upper_left = lower_left + columns + 1
            upper_right = upper_left + 1
            indices = [len(triangles), len(triangles) + 1]
            triangles.extend(
                [
                    [lower_left, lower_right, upper_right],
                    [lower_left, upper_right, upper_left],
                ]
            )
            cell_triangles[(row, column)] = indices
    return coordinates, triangles, cell_triangles


def _first_arrival(times: np.ndarray, values: np.ndarray, threshold: float) -> float:
    reached = np.flatnonzero(values >= threshold)
    return float(times[reached[0]]) if reached.size else float("inf")


def test_jax_and_anuga_toy_flood_peak_and_arrival_agree() -> None:
    rows = columns = 10
    spacing = 10.0
    duration_s = 20.0
    inflow_rate_m_s = 0.005
    inflow_cell = (5, 5)
    gauge_cell = (5, 7)
    output_times = np.arange(0.0, duration_s + 1.0, 1.0, dtype=np.float32)

    coordinates, triangles, cell_triangles = _triangular_square_mesh(
        rows,
        columns,
        spacing,
    )
    anuga_domain = anuga.Domain(coordinates, triangles)
    anuga_domain.set_store(False)
    anuga_domain.set_flow_algorithm("DE1")
    anuga_domain.set_cfl(0.4)
    anuga_domain.set_quantity("elevation", 0.0)
    anuga_domain.set_quantity("stage", 0.0)
    anuga_domain.set_quantity("friction", 0.0)
    anuga_domain.set_boundary({"exterior": anuga.Reflective_boundary(anuga_domain)})
    anuga.Rate_operator(
        anuga_domain,
        rate=inflow_rate_m_s,
        indices=cell_triangles[inflow_cell],
        label="single_cell_inflow",
    )
    anuga_inflow_depth = []
    anuga_gauge_depth = []
    anuga_times = []
    for time_s in anuga_domain.evolve(yieldstep=1.0, finaltime=duration_s):
        depth = (
            anuga_domain.quantities["stage"].centroid_values
            - anuga_domain.quantities["elevation"].centroid_values
        )
        anuga_inflow_depth.append(float(depth[cell_triangles[inflow_cell]].mean()))
        anuga_gauge_depth.append(float(depth[cell_triangles[gauge_cell]].mean()))
        anuga_times.append(float(time_s))

    bed = np.zeros((rows, columns), dtype=np.float32)
    rainfall_multiplier = np.zeros_like(bed)
    rainfall_multiplier[inflow_cell] = 1.0
    grid = StructuredGrid(
        bed=bed,
        manning_n=np.zeros_like(bed),
        domain_mask=np.ones_like(bed, dtype=bool),
        x=(np.arange(columns) + 0.5) * spacing,
        y=(np.arange(rows) + 0.5) * spacing,
        resolution_m=spacing,
        crs=CRS.from_epsg(32643),
        north_up_transform=Affine.identity(),
    )
    jax_solver = ShallowWaterSolver(
        grid,
        rainfall_times_s=np.array([0.0, duration_s], dtype=np.float32),
        rainfall_rates_m_s=np.array([inflow_rate_m_s, inflow_rate_m_s], dtype=np.float32),
        rainfall_multiplier=rainfall_multiplier,
        cfl=0.4,
        dry_tolerance_m=1.0e-5,
        max_dt_s=0.2,
    )
    jax_result = jax_solver.run(
        jax_solver.initial_state(),
        output_times,
        max_steps=2_000,
    )
    jax_depth = np.asarray(jax_result.states.h)
    jax_inflow_depth = jax_depth[:, inflow_cell[0], inflow_cell[1]]
    jax_gauge_depth = jax_depth[:, gauge_cell[0], gauge_cell[1]]

    # The triangular DE1 and structured HLL discretizations have different
    # numerical diffusion. These tolerances detect major implementation bugs
    # without requiring two distinct schemes to be pointwise identical.
    np.testing.assert_allclose(
        np.max(jax_inflow_depth),
        np.max(anuga_inflow_depth),
        rtol=0.15,
        atol=1.0e-3,
    )
    threshold_m = 1.0e-3
    anuga_arrival = _first_arrival(
        np.asarray(anuga_times),
        np.asarray(anuga_gauge_depth),
        threshold_m,
    )
    jax_arrival = _first_arrival(output_times, jax_gauge_depth, threshold_m)
    assert abs(jax_arrival - anuga_arrival) <= 3.0
