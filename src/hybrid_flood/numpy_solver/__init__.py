"""NumPy baseline for the structured shallow-water forecast."""

from hybrid_flood.numpy_solver.shallow_water_2d import (
    NumpyResult,
    NumpyShallowWaterSolver,
    NumpyState,
)

__all__ = ["NumpyResult", "NumpyShallowWaterSolver", "NumpyState"]
