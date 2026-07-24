# Architecture decision log

Record each important decision with its context, considered alternatives,
chosen approach, and consequences.

## Initial direction

- ANUGA provides the numerical reference solution, not observational ground
  truth. Both solvers share a synthetic, uncalibrated DEM and assumed
  parameters.
- A custom JAX finite-volume solver provides GPU-accelerated physics.
- A residual model corrects systematic discrepancies rather than replacing
  the governing solver.
