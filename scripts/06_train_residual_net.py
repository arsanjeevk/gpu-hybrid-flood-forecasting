"""Train and evaluate the learned residual correction network."""

# ruff: noqa: E402

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from hybrid_flood.jax_solver.runtime import configure_jax_runtime

JAX_RUNTIME = configure_jax_runtime()

import jax
import numpy as np

from hybrid_flood.ml.dataset import load_residual_dataset
from hybrid_flood.ml.evaluate import evaluate_model
from hybrid_flood.ml.residual_net import model_from_config
from hybrid_flood.ml.train import train_model

LOGGER = logging.getLogger(__name__)


def _path(root: Path, configured: str) -> Path:
    path = Path(configured)
    return path if path.is_absolute() else root / path


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig) -> None:
    """Train with early stopping and evaluate the held-out time block."""
    root = Path(get_original_cwd()).resolve()
    model_cfg = cfg.model
    devices = jax.devices()
    gpu_devices = [device for device in devices if device.platform == "gpu"]
    if model_cfg.execution.require_gpu and not gpu_devices:
        raise RuntimeError(
            "GPU training was required, but JAX found no GPU device. "
            f"{JAX_RUNTIME['cuda_driver']['reason']}"
        )
    if gpu_devices:
        LOGGER.info("Training on JAX GPU device(s): %s", ", ".join(map(str, gpu_devices)))
    else:
        LOGGER.warning(
            "No JAX GPU is available; residual-network training will run on CPU. %s",
            JAX_RUNTIME["cuda_driver"]["reason"],
        )

    dataset = load_residual_dataset(_path(root, model_cfg.dataset.output))
    expected_duration_s = float(cfg.jax_solver.duration_s)
    if not np.isclose(
        float(dataset.target_time_s[-1]),
        expected_duration_s,
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise ValueError(
            "Residual dataset is stale or incomplete: it ends at "
            f"{float(dataset.target_time_s[-1])} s, expected {expected_duration_s} s."
        )
    model = model_from_config(model_cfg.architecture)
    result = train_model(
        model,
        dataset,
        model_cfg.training,
        seed=int(cfg.project.seed),
        checkpoint_path=_path(root, model_cfg.outputs.checkpoint),
        history_path=_path(root, model_cfg.outputs.training_log),
    )
    metrics = evaluate_model(
        model,
        result.state.params,
        dataset,
        patch_size=int(model_cfg.training.patch_size),
        batch_size=int(model_cfg.training.batch_size),
        flood_threshold_m=float(model_cfg.evaluation.flood_threshold_m),
        output_path=_path(root, model_cfg.outputs.test_metrics),
    )
    metadata = {
        "configuration": OmegaConf.to_container(model_cfg, resolve=True),
        "execution": {
            "devices": [str(device) for device in devices],
            "default_backend": jax.default_backend(),
            "gpu_accelerated": bool(gpu_devices),
            "runtime_probe": JAX_RUNTIME,
        },
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
        "Training complete: best epoch %d, validation loss %.6g, test depth RMSE %.6g m",
        result.best_epoch,
        result.best_validation_loss,
        metrics["depth_rmse"],
    )


if __name__ == "__main__":
    main()
