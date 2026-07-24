"""Fresh-process worker for one architecture-matched benchmark batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hybrid_flood.jax_solver.runtime import configure_jax_runtime

configure_jax_runtime()

from hybrid_flood.benchmark.jax_vs_pytorch import benchmark_frameworks  # noqa: E402
from hybrid_flood.benchmark.pytorch_residual_net import TorchResidualUNet  # noqa: E402
from hybrid_flood.ml.residual_net import ResidualUNet  # noqa: E402


def main() -> None:
    """Run one batch configuration and write records as JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    payload = json.loads(Path(arguments.payload).read_text(encoding="utf-8"))
    architecture = payload["architecture"]
    flax_model = ResidualUNet(
        depth=int(architecture["depth"]),
        channels=tuple(int(value) for value in architecture["channels"]),
        activation=str(architecture["activation"]),
        kernel_size=int(architecture["kernel_size"]),
        output_channels=int(architecture["output_channels"]),
    )

    def torch_factory() -> TorchResidualUNet:
        return TorchResidualUNet(
            input_channels=int(payload["input_shape"][2]),
            depth=int(architecture["depth"]),
            channels=tuple(int(value) for value in architecture["channels"]),
            activation=str(architecture["activation"]),
            kernel_size=int(architecture["kernel_size"]),
            output_channels=int(architecture["output_channels"]),
        )

    results = benchmark_frameworks(
        flax_model,
        torch_factory,
        batch_sizes=[int(payload["batch_size"])],
        input_shape=tuple(int(value) for value in payload["input_shape"]),
        warmup_iterations=int(payload["warmup_iterations"]),
        measured_iterations=int(payload["measured_iterations"]),
        require_gpu=bool(payload["require_gpu"]),
        allow_cpu=bool(payload["allow_cpu"]),
        torch_compile_enabled=bool(payload["torch_compile_enabled"]),
        torch_compile_mode=str(payload["torch_compile_mode"]),
        seed=int(payload["seed"]),
    )
    destination = Path(arguments.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    results.to_json(destination, orient="records", indent=2)


if __name__ == "__main__":
    main()
