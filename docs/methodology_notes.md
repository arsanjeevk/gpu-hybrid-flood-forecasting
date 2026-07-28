# Methodology notes

Use this document for validated equations, numerical assumptions, dataset
transformations, experimental observations, and material that will later be
incorporated into the report.

## Current scientific status

- ANUGA is a numerical reference baseline, not ground truth. No observed flood
  depths, discharges, surveyed DEM, or calibration data are currently present.
- The 60-step chronological ML split is a single-event development holdout.
  It reduces direct temporal leakage but does not demonstrate generalization
  to independent storms.
- The loss/activity mask is constructed from the training time block only.
  Validation and test inundation extents are not used to configure training.
- Neural depth corrections preserve the water volume produced by each JAX
  physics interval; the network can redistribute water but cannot act as an
  unmodelled source or sink.
- Depth skill is reported both over the full common domain and over
  reference-wet samples (`ANUGA depth > 1e-4 m`) so dry-background cells
  cannot conceal poor inundation performance. Flood-extent CSI is reported
  separately at the configured event threshold.
- The source `BndTypeNo`/`ConstValue` schema could not be recovered. A final
  documented modelling decision therefore treats every domain boundary as
  reflective (closed/no-flow). This is a known limitation, not a pending
  implementation task; see `docs/decisions_log.md`.
- The configured three-hour run samples 8.82% of the 35-hour rainfall record,
  integrates 8.35497 mm under the linearly interpolated forcing, and excludes
  its peak at hour 34. It is a development experiment, not a complete
  return-period event simulation.

## JAX versus PyTorch benchmark protocol

- Both models contain 150,883 trainable parameters with matched convolutions,
  biases, skip connections, activations, downsampling, upsampling, and output
  head. JAX uses NHWC and PyTorch uses NCHW; layout conversion and device
  transfer occur before timing.
- Both frameworks receive numerically identical seeded float32 inputs.
- Forward and forward-backward-AdamW operations are measured separately.
  Every GPU interval is synchronized, compilation/first-iteration time is
  reported separately, and steady-state results contain mean and sample
  standard deviation over repeated measurements.
- Each batch size runs in a fresh subprocess because JAX exposes a
  process-level allocator peak without PyTorch's public peak-reset operation.
  This prevents a previous batch from contaminating JAX peak-memory results.
- Publication output requires a GPU visible to both frameworks. CPU execution
  is supported only as an explicitly enabled smoke test.
- JAX Perfetto/XProf and PyTorch Chrome traces are captured after warmup and
  aggregated into a common operation-duration CSV schema.
