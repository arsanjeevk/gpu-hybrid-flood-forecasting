"""Publication training and validation learning curves."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hybrid_flood.viz.style import DOUBLE_COLUMN_WIDTH, save_figure_pair


def plot_training_curves(
    training_log: str | Path | pd.DataFrame,
    output_base: str | Path,
) -> tuple[Path, Path]:
    """Plot total/data/physical losses and the learning-rate schedule."""
    history = (
        training_log.copy() if isinstance(training_log, pd.DataFrame) else pd.read_csv(training_log)
    )
    required = {
        "epoch",
        "train_total",
        "validation_total",
        "train_mse",
        "validation_mse",
        "train_mae",
        "validation_mae",
        "train_negative_depth_penalty",
        "validation_negative_depth_penalty",
        "learning_rate",
    }
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(f"Training log is missing columns: {sorted(missing)}")
    if history.empty or not np.isfinite(history[list(required)].to_numpy()).all():
        raise ValueError("Training history is empty or contains non-finite values.")

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(DOUBLE_COLUMN_WIDTH, 2.7),
        constrained_layout=True,
    )
    axes[0].plot(history.epoch, history.train_total, label="Training", color="#0072B2")
    axes[0].plot(
        history.epoch,
        history.validation_total,
        label="Validation",
        color="#D55E00",
    )
    best = int(history.validation_total.idxmin())
    axes[0].scatter(
        history.epoch.iloc[best],
        history.validation_total.iloc[best],
        color="black",
        marker="*",
        s=35,
        zorder=3,
        label=f"Best epoch ({int(history.epoch.iloc[best])})",
    )
    axes[0].set(
        title="Residual-network objective",
        xlabel="Epoch",
        ylabel="Scaled loss",
        yscale="log",
    )
    axes[0].grid(True)
    axes[0].legend(frameon=False)

    axes[1].plot(history.epoch, history.train_mse, label="Train MSE", color="#0072B2")
    axes[1].plot(
        history.epoch,
        history.validation_mse,
        label="Validation MSE",
        color="#D55E00",
    )
    axes[1].plot(
        history.epoch,
        history.train_mae,
        label="Train MAE",
        color="#56B4E9",
        linestyle="--",
    )
    axes[1].plot(
        history.epoch,
        history.validation_mae,
        label="Validation MAE",
        color="#E69F00",
        linestyle="--",
    )
    axes[1].set(
        title="Data-loss components",
        xlabel="Epoch",
        ylabel="Scaled component",
        yscale="log",
    )
    axes[1].grid(True)
    axes[1].legend(frameon=False, ncol=2)
    return save_figure_pair(figure, output_base)
