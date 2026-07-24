"""Configurable Flax U-Net for two-dimensional residual correction."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp


def _activation(name: str) -> Callable[[jax.Array], jax.Array]:
    activations: dict[str, Callable[[jax.Array], jax.Array]] = {
        "relu": nn.relu,
        "gelu": nn.gelu,
        "silu": nn.silu,
        "tanh": jnp.tanh,
    }
    try:
        return activations[name.lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported activation {name!r}; choose one of {sorted(activations)}."
        ) from exc


class ConvBlock(nn.Module):
    """Two same-resolution convolutions used at each U-Net level."""

    channels: int
    kernel_size: int
    activation: str

    @nn.compact
    def __call__(self, inputs: jax.Array) -> jax.Array:
        activate = _activation(self.activation)
        x = nn.Conv(self.channels, (self.kernel_size, self.kernel_size), padding="SAME")(inputs)
        x = activate(x)
        x = nn.Conv(self.channels, (self.kernel_size, self.kernel_size), padding="SAME")(x)
        return activate(x)


class ResidualUNet(nn.Module):
    """U-Net-style encoder-decoder predicting depth and velocity residuals.

    Parameters are intentionally supplied from Hydra configuration rather than
    fixed in source.  Inputs and outputs use channels-last layout
    ``(batch, y, x, channels)``.
    """

    depth: int
    channels: Sequence[int]
    activation: str = "relu"
    kernel_size: int = 3
    output_channels: int = 3

    @nn.compact
    def __call__(self, inputs: jax.Array) -> jax.Array:
        if self.depth < 1:
            raise ValueError("U-Net depth must be positive.")
        if len(self.channels) < self.depth:
            raise ValueError("The channel list must contain at least one value per U-Net depth.")
        if self.kernel_size < 1 or self.kernel_size % 2 == 0:
            raise ValueError("The convolution kernel size must be a positive odd integer.")

        widths = tuple(int(value) for value in self.channels[: self.depth])
        x = inputs
        skips: list[jax.Array] = []
        for level, width in enumerate(widths):
            x = ConvBlock(width, self.kernel_size, self.activation, name=f"encoder_{level}")(x)
            skips.append(x)
            if level < self.depth - 1:
                x = nn.Conv(
                    widths[level + 1],
                    (2, 2),
                    strides=(2, 2),
                    padding="SAME",
                    name=f"downsample_{level}",
                )(x)

        for level in range(self.depth - 2, -1, -1):
            skip = skips[level]
            x = nn.ConvTranspose(
                widths[level],
                (2, 2),
                strides=(2, 2),
                padding="SAME",
                name=f"upsample_{level}",
            )(x)
            if x.shape[1:3] != skip.shape[1:3]:
                x = jax.image.resize(
                    x,
                    (x.shape[0], skip.shape[1], skip.shape[2], x.shape[-1]),
                    method="linear",
                )
            x = jnp.concatenate((x, skip), axis=-1)
            x = ConvBlock(
                widths[level],
                self.kernel_size,
                self.activation,
                name=f"decoder_{level}",
            )(x)

        return nn.Conv(
            self.output_channels,
            (1, 1),
            padding="SAME",
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
            name="residual_head",
        )(x)


def model_from_config(config: object) -> ResidualUNet:
    """Construct the network from a mapping- or attribute-style config."""

    def value(name: str):
        if isinstance(config, dict):
            return config[name]
        return getattr(config, name)

    return ResidualUNet(
        depth=int(value("depth")),
        channels=tuple(int(channel) for channel in value("channels")),
        activation=str(value("activation")),
        kernel_size=int(value("kernel_size")),
        output_channels=int(value("output_channels")),
    )
