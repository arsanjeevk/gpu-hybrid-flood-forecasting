"""Shape, temporal-split, metric, and one-step learning sanity tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from hybrid_flood.ml.dataset import temporal_split_indices
from hybrid_flood.ml.evaluate import compute_test_metrics
from hybrid_flood.ml.losses import residual_correction_loss
from hybrid_flood.ml.residual_net import ResidualUNet
from hybrid_flood.ml.train import (
    create_train_state,
    load_checkpoint,
    save_checkpoint,
    training_step,
)


def test_residual_unet_preserves_odd_spatial_shape() -> None:
    """Encoder-decoder output must align exactly with the input grid."""
    model = ResidualUNet(
        depth=3,
        channels=(4, 8, 16),
        activation="relu",
        kernel_size=3,
        output_channels=3,
    )
    inputs = jnp.zeros((2, 17, 19, 6), dtype=jnp.float32)
    variables = model.init(jax.random.PRNGKey(0), inputs)

    output = model.apply(variables, inputs)

    assert output.shape == (2, 17, 19, 3)
    assert jnp.isfinite(output).all()


def test_single_training_step_reduces_tiny_batch_loss() -> None:
    """One optimizer step should move a zero-initialized head toward a simple target."""
    model = ResidualUNet(
        depth=2,
        channels=(4, 8),
        activation="relu",
        kernel_size=3,
        output_channels=3,
    )
    state = create_train_state(
        model,
        jax.random.PRNGKey(1),
        (1, 8, 8, 6),
        optax.adam(learning_rate=0.01),
    )
    batch = {
        "inputs": jnp.ones((1, 8, 8, 6), dtype=jnp.float32),
        "target": jnp.full((1, 8, 8, 3), 0.05, dtype=jnp.float32),
        "raw_depth": jnp.full((1, 8, 8), 0.2, dtype=jnp.float32),
        "mask": jnp.ones((1, 8, 8), dtype=bool),
    }
    weights = jnp.asarray((1.0, 0.0, 1.0), dtype=jnp.float32)
    initial_prediction = state.apply_fn({"params": state.params}, batch["inputs"])
    initial_loss = residual_correction_loss(
        initial_prediction,
        batch["target"],
        batch["raw_depth"],
        batch["mask"],
    ).total

    updated, _ = training_step(state, batch, weights, jnp.ones((3,), dtype=jnp.float32))
    updated_prediction = updated.apply_fn({"params": updated.params}, batch["inputs"])
    updated_loss = residual_correction_loss(
        updated_prediction,
        batch["target"],
        batch["raw_depth"],
        batch["mask"],
    ).total

    assert float(updated_loss) < float(initial_loss)


def test_temporal_splits_are_contiguous_and_disjoint() -> None:
    """Chronological partitions must never interleave adjacent states."""
    train, validation, test = temporal_split_indices(60)

    assert len(train) + len(validation) + len(test) == 60
    assert train[-1] + 1 == validation[0]
    assert validation[-1] + 1 == test[0]
    assert not set(train) & set(validation)
    assert not set(validation) & set(test)

    train, validation, test = temporal_split_indices(
        180,
        train_fraction=0.70,
        val_fraction=0.15,
    )
    assert (len(train), len(validation), len(test)) == (126, 27, 27)


def test_perfect_correction_has_unit_csi_and_zero_error() -> None:
    """Metric definitions should recognize an exact residual prediction."""
    target = np.zeros((1, 3, 3, 3), dtype=np.float32)
    target[..., 0] = 0.1
    raw_depth = np.zeros((1, 3, 3), dtype=np.float32)
    mask = np.ones((3, 3), dtype=bool)

    metrics = compute_test_metrics(
        target,
        target,
        raw_depth,
        mask,
        flood_threshold_m=0.05,
    )

    assert metrics["depth_rmse"] == 0.0
    assert metrics["mae_all_channels"] == 0.0
    assert metrics["critical_success_index"] == 1.0


def test_metrics_reject_nonfinite_values_inside_evaluation_domain() -> None:
    """Invalid solver cells must not silently turn publication metrics into NaN."""
    prediction = np.zeros((1, 2, 2, 3), dtype=np.float32)
    target = np.zeros_like(prediction)
    raw_depth = np.zeros((1, 2, 2), dtype=np.float32)
    mask = np.ones((2, 2), dtype=bool)
    prediction[0, 0, 0, 1] = np.nan

    with pytest.raises(ValueError, match="common finite"):
        compute_test_metrics(prediction, target, raw_depth, mask)


def test_checkpoint_round_trip(tmp_path) -> None:
    """Saved best parameters must be usable by later hybrid rollouts."""
    model = ResidualUNet(depth=1, channels=(4,), output_channels=3)
    state = create_train_state(
        model,
        jax.random.PRNGKey(2),
        (1, 8, 8, 6),
        optax.adam(1.0e-3),
    )
    path = tmp_path / "model.msgpack"

    save_checkpoint(state, path, metadata={"epoch": 1})
    restored = load_checkpoint(state.params, path)

    expected = state.apply_fn(
        {"params": state.params},
        jnp.ones((1, 8, 8, 6), dtype=jnp.float32),
    )
    actual = state.apply_fn(
        {"params": restored},
        jnp.ones((1, 8, 8, 6), dtype=jnp.float32),
    )
    np.testing.assert_allclose(actual, expected)
