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
- Boundary `BndTypeNo` semantics remain unverified, so all boundaries are
  intentionally reflective and results must be labelled provisional.
- The configured one-hour run samples only the start of the 35-hour rainfall
  record and excludes its peak. It is a pipeline demonstration, not yet a
  complete return-period event simulation.

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
