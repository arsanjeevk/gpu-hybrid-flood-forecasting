"""Toy-basin mass-conservation regression test for ANUGA configuration."""

from __future__ import annotations

import anuga
import numpy as np

from hybrid_flood.anuga_solver.postprocess_sww import postprocess_sww
from hybrid_flood.anuga_solver.run_simulation import run_simulation


def test_flat_reflective_basin_single_inflow_conserves_mass(tmp_path) -> None:
    coordinates = [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]]
    triangles = [[0, 1, 2], [1, 3, 2]]
    domain = anuga.Domain(coordinates, triangles)
    domain.set_quantity("elevation", 0.0)
    domain.set_quantity("stage", 0.0)
    domain.set_quantity("friction", 0.03)
    domain.set_boundary({"exterior": anuga.Reflective_boundary(domain)})

    inflow_rate_m_s = 0.001
    anuga.Rate_operator(
        domain,
        rate=inflow_rate_m_s,
        indices=[0],
        label="single_triangle_inflow",
    )
    result = run_simulation(
        domain,
        tmp_path,
        duration_s=10.0,
        yieldstep_s=1.0,
        outputstep_s=1.0,
        flow_algorithm="DE1",
        cfl=0.5,
    )

    expected_volume = 50.0 * inflow_rate_m_s * 10.0
    water_volume = domain.get_water_volume()
    forcing_volume = domain.get_fractional_step_volume_integral()
    boundary_flux = domain.get_boundary_flux_integral()
    np.testing.assert_allclose(
        water_volume,
        expected_volume,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        forcing_volume,
        expected_volume,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    assert abs(boundary_flux) < 1.0e-12
    assert result.maximum_absolute_mass_error_m3 < 1.0e-10
    assert result.sww_path.is_file()
    assert result.metrics_path.is_file()

    dataset, report = postprocess_sww(
        result.sww_path,
        tmp_path / "toy_ground_truth.nc",
        grid_resolution_m=2.0,
        dry_tolerance_m=1.0e-6,
    )
    assert dataset["depth"].dims == ("time", "y", "x")
    assert dataset["velocity"].dims == ("time", "y", "x")
    assert report["time_steps"] == 11
    assert (tmp_path / "toy_ground_truth.nc").is_file()
