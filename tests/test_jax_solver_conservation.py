"""Conservation and lake-at-rest regression tests for the JAX solver."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from affine import Affine
from pyproj import CRS

from hybrid_flood.jax_solver.boundary_conditions import (
    all_open,
    all_reflective,
    apply_ghost_cells,
)
from hybrid_flood.jax_solver.shallow_water_2d import (
    ShallowWaterSolver,
    StructuredGrid,
    load_structured_grid,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_reflective_and_open_ghost_cells_have_correct_normal_momentum() -> None:
    h = np.ones((2, 2), dtype=np.float32)
    hu = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    hv = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    _, reflective_hu, reflective_hv = apply_ghost_cells(
        h,
        hu,
        hv,
        all_reflective().types,
    )
    _, open_hu, open_hv = apply_ghost_cells(h, hu, hv, all_open().types)
    np.testing.assert_allclose(np.asarray(reflective_hu)[1:-1, 0], -hu[:, 0])
    np.testing.assert_allclose(np.asarray(reflective_hu)[1:-1, -1], -hu[:, -1])
    np.testing.assert_allclose(np.asarray(reflective_hv)[0, 1:-1], -hv[0, :])
    np.testing.assert_allclose(np.asarray(reflective_hv)[-1, 1:-1], -hv[-1, :])
    np.testing.assert_allclose(np.asarray(open_hu)[1:-1, 0], hu[:, 0])
    np.testing.assert_allclose(np.asarray(open_hv)[0, 1:-1], hv[0, :])


def _solver_for_bed(
    bed: np.ndarray,
    *,
    resolution_m: float = 1.0,
    manning_n: float = 0.0,
    mask: np.ndarray | None = None,
) -> ShallowWaterSolver:
    if mask is None:
        mask = np.ones(bed.shape, dtype=bool)
    grid = StructuredGrid(
        bed=bed.astype(np.float32),
        manning_n=np.full(bed.shape, manning_n, dtype=np.float32),
        domain_mask=mask,
        x=(np.arange(bed.shape[1]) + 0.5) * resolution_m,
        y=(np.arange(bed.shape[0]) + 0.5) * resolution_m,
        resolution_m=resolution_m,
        crs=CRS.from_epsg(32643),
        north_up_transform=Affine.identity(),
    )
    return ShallowWaterSolver(
        grid,
        rainfall_times_s=np.array([0.0, 100.0], dtype=np.float32),
        rainfall_rates_m_s=np.array([0.0, 0.0], dtype=np.float32),
        cfl=0.4,
        dry_tolerance_m=1.0e-6,
        max_dt_s=0.1,
    )


def test_reflective_flat_basin_conserves_water_mass() -> None:
    bed = np.zeros((20, 20), dtype=np.float32)
    depth = np.full_like(bed, 0.1)
    depth[8:12, 8:12] = 0.2
    solver = _solver_for_bed(bed)
    result = solver.run(
        solver.initial_state(depth_m=depth),
        np.array([0.0, 1.0, 2.0], dtype=np.float32),
        max_steps=1_000,
    )
    volume = np.asarray(result.states.h).sum(axis=(1, 2))
    np.testing.assert_allclose(volume, volume[0], rtol=2.0e-7, atol=1.0e-5)


def test_lake_at_rest_remains_well_balanced() -> None:
    coordinate = np.linspace(-1.0, 1.0, 20)
    xx, yy = np.meshgrid(coordinate, coordinate)
    bed = (0.03 * np.exp(-5.0 * (xx**2 + yy**2))).astype(np.float32)
    water_surface_elevation = 0.2
    solver = _solver_for_bed(bed)
    result = solver.run(
        solver.initial_state(depth_m=water_surface_elevation - bed),
        np.array([0.0, 1.0, 2.0], dtype=np.float32),
        max_steps=1_000,
    )
    final_surface = np.asarray(result.states.h[-1]) + bed
    final_hu = np.asarray(result.states.hu[-1])
    final_hv = np.asarray(result.states.hv[-1])
    np.testing.assert_allclose(final_surface, water_surface_elevation, atol=2.0e-7)
    assert np.max(np.abs(final_hu)) < 1.0e-6
    assert np.max(np.abs(final_hv)) < 1.0e-6


def test_lake_at_rest_remains_balanced_at_irregular_mask_walls() -> None:
    """Inactive raster cells must behave as solid reflective boundaries."""
    bed = np.zeros((12, 12), dtype=np.float32)
    mask = np.ones_like(bed, dtype=bool)
    mask[4:8, 5:7] = False
    solver = _solver_for_bed(bed, mask=mask)
    result = solver.run(
        solver.initial_state(depth_m=0.1),
        np.array([0.0, 0.5], dtype=np.float32),
        max_steps=1_000,
    )
    final_depth = np.asarray(result.states.h[-1])
    final_hu = np.asarray(result.states.hu[-1])
    final_hv = np.asarray(result.states.hv[-1])
    np.testing.assert_allclose(final_depth[mask], 0.1, atol=2.0e-7)
    assert np.max(np.abs(final_hu[mask])) < 1.0e-6
    assert np.max(np.abs(final_hv[mask])) < 1.0e-6


def test_production_structured_grid_aligns_with_anuga_grid() -> None:
    grid = load_structured_grid(
        PROJECT_ROOT / "data/synthetic_dem/dem.tif",
        PROJECT_ROOT / "data/interim/roughness_epsg32643.gpkg",
        PROJECT_ROOT / "data/interim/domain_epsg32643.gpkg",
        resolution_m=50.0,
    )
    assert grid.bed.shape == (517, 551)
    assert grid.crs.to_epsg() == 32643
    assert np.isfinite(grid.bed[grid.domain_mask]).all()
    assert np.isfinite(grid.manning_n[grid.domain_mask]).all()
    assert grid.x[0] == 686_875.0
    assert grid.y[0] == 3_135_225.0
