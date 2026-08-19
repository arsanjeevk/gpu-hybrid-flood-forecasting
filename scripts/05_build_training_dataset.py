"""Build aligned, chronologically split residual-learning samples."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from hybrid_flood.ml.dataset import build_residual_dataset, save_residual_dataset

LOGGER = logging.getLogger(__name__)


def _path(root: Path, configured: str) -> Path:
    path = Path(configured)
    return path if path.is_absolute() else root / path


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig) -> None:
    """Build and persist the Phase 5 residual dataset."""
    root = Path(get_original_cwd()).resolve()
    model_cfg = cfg.model
    LOGGER.info("Aligning ANUGA and V1 NumPy fields and constructing residual targets")
    dataset = build_residual_dataset(
        _path(root, model_cfg.inputs.anuga),
        _path(root, model_cfg.inputs.forecast),
        _path(root, model_cfg.inputs.roughness),
        _path(root, model_cfg.inputs.rainfall),
        rainfall_scenario=model_cfg.inputs.rainfall_scenario,
        train_fraction=model_cfg.dataset.train_fraction,
        val_fraction=model_cfg.dataset.validation_fraction,
        permanently_dry_threshold_m=model_cfg.dataset.permanently_dry_threshold_m,
    )
    expected_duration_s = float(cfg.comparison.common.duration_s)
    if not np.isclose(
        float(dataset.target_time_s[-1]),
        expected_duration_s,
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise ValueError(
            "Solver artifacts are stale or incomplete: the residual dataset ends at "
            f"{float(dataset.target_time_s[-1])} s, expected {expected_duration_s} s."
        )
    output = save_residual_dataset(dataset, _path(root, model_cfg.dataset.output))
    LOGGER.info(
        "Saved %d chronological transitions (%d/%d/%d train/val/test) to %s",
        len(dataset.input_time_s),
        len(dataset.train_indices),
        len(dataset.val_indices),
        len(dataset.test_indices),
        output,
    )


if __name__ == "__main__":
    main()
