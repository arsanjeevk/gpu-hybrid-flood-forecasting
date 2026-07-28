# Report results status

The numerical text currently present in Sections 05--11 describes the earlier
one-hour development run. It is retained only as drafting history and is not
the final three-hour publication result.

Do not submit or cite `main.pdf` until all of the following are true:

1. `scripts/00_gpu_preflight.py` passes on the allocated A100 node.
2. Scripts 03 through 09 have been rerun in order with the frozen lockfile.
3. `scripts/10_validate_gpu_results.py` passes.
4. Sections 05--11 have been updated strictly from the new three-hour
   metadata, metrics, benchmark CSV, and generated figures.
5. `latexmk` completes without warnings after those updates.

ANUGA is a numerical reference, not observational ground truth. The synthetic
DEM, all-reflective boundary policy, absence of field calibration, and
three-hour window that excludes the rainfall peak must remain explicit
limitations in the final report.
