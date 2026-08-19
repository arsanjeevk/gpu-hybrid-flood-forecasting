"""Train the shared V1/V2 residual model with PyTorch CUDA."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from hybrid_flood.ml.dataset import load_residual_dataset
from hybrid_flood.ml.torch_evaluate import evaluate_model
from hybrid_flood.ml.torch_model import PyTorchResidualUNet, count_torch_parameters
from hybrid_flood.ml.torch_train import train_model

LOGGER = logging.getLogger(__name__)


def _path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _model(cfg: DictConfig, input_channels: int) -> PyTorchResidualUNet:
    architecture = cfg.model.architecture
    return PyTorchResidualUNet(
        input_channels=input_channels,
        depth=int(architecture.depth),
        channels=tuple(int(value) for value in architecture.channels),
        activation=str(architecture.activation),
        kernel_size=int(architecture.kernel_size),
        output_channels=int(architecture.output_channels),
    )


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig) -> None:
    """Train once on V1 data; V1 and V2 reuse the identical checkpoint."""
    root = Path(get_original_cwd()).resolve()
    model_cfg = cfg.model
    if str(model_cfg.framework).lower() != "pytorch":
        raise ValueError("The active V1/V2 comparison requires framework=pytorch.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if bool(model_cfg.execution.require_gpu) and device.type != "cuda":
        raise RuntimeError("PyTorch CUDA training is required; no CUDA device is visible.")
    if bool(cfg.comparison.hardware.require_t4) and device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        if "T4" not in name.upper():
            raise RuntimeError(f"The active experiment requires an NVIDIA T4, found {name!r}.")
    dataset = load_residual_dataset(_path(root, model_cfg.dataset.output))
    expected = float(cfg.comparison.common.duration_s)
    if not np.isclose(float(dataset.target_time_s[-1]), expected):
        raise ValueError("The training dataset does not cover the configured three-hour horizon.")
    model = _model(cfg, dataset.input_channels)
    result = train_model(
        model,
        dataset,
        model_cfg.training,
        seed=int(cfg.project.seed),
        device=device,
        checkpoint_path=_path(root, model_cfg.outputs.checkpoint),
        history_path=_path(root, model_cfg.outputs.training_log),
    )
    _, metrics = evaluate_model(
        result.model,
        dataset,
        patch_size=int(model_cfg.training.patch_size),
        batch_size=int(model_cfg.training.batch_size),
        device=device,
        flood_threshold_m=float(model_cfg.evaluation.flood_threshold_m),
    )
    metrics_path = _path(root, model_cfg.outputs.test_metrics)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    metadata = {
        "generated_at": datetime.now(UTC).isoformat(),
        "configuration": OmegaConf.to_container(model_cfg, resolve=True),
        "framework": "pytorch",
        "shared_by_versions": ["V1", "V2"],
        "training_source": "V1 NumPy forecast residuals against ANUGA",
        "execution": {
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "gpu_accelerated": device.type == "cuda",
            "peak_memory_mb": (
                torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None
            ),
        },
        "parameter_count": count_torch_parameters(result.model),
        "best_epoch": result.best_epoch,
        "epochs_completed": result.epochs_completed,
        "best_validation_loss": result.best_validation_loss,
        "checkpoint": str(result.checkpoint_path),
        "training_log": str(result.history_path),
        "test_metrics": metrics,
    }
    metadata_path = _path(root, model_cfg.outputs.run_metadata)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    LOGGER.info(
        "PyTorch training complete on %s: epoch %d, test depth RMSE %.6g m",
        metadata["execution"]["device_name"],
        result.best_epoch,
        metrics["depth_rmse"],
    )


if __name__ == "__main__":
    main()
