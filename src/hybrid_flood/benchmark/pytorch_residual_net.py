"""Layer-for-layer PyTorch counterpart of the Flax residual U-Net."""

from __future__ import annotations

from collections.abc import Sequence
from math import ceil

import torch
import torch.nn.functional as functional
from torch import nn


def _activation(name: str) -> nn.Module:
    """Return the PyTorch activation matching ``ml.residual_net._activation``."""
    activations: dict[str, nn.Module] = {
        "relu": nn.ReLU(),
        # Flax/JAX GELU uses the tanh approximation by default.
        "gelu": nn.GELU(approximate="tanh"),
        "silu": nn.SiLU(),
        "tanh": nn.Tanh(),
    }
    try:
        return activations[name.lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported activation {name!r}; choose one of {sorted(activations)}."
        ) from exc


def _same_stride_two(inputs: torch.Tensor, kernel_size: int = 2) -> torch.Tensor:
    """Apply TensorFlow/Flax ``SAME`` padding for a stride-two convolution."""
    height, width = inputs.shape[-2:]
    output_height, output_width = ceil(height / 2), ceil(width / 2)
    pad_height = max((output_height - 1) * 2 + kernel_size - height, 0)
    pad_width = max((output_width - 1) * 2 + kernel_size - width, 0)
    return functional.pad(
        inputs,
        (
            pad_width // 2,
            pad_width - pad_width // 2,
            pad_height // 2,
            pad_height - pad_height // 2,
        ),
    )


class TorchConvBlock(nn.Module):
    """Two same-resolution convolutions matching the Flax ``ConvBlock``."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        activation: str,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv_1 = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size,
            padding=padding,
        )
        self.activation_1 = _activation(activation)
        self.conv_2 = nn.Conv2d(
            output_channels,
            output_channels,
            kernel_size,
            padding=padding,
        )
        self.activation_2 = _activation(activation)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply both convolutions and activations."""
        return self.activation_2(self.conv_2(self.activation_1(self.conv_1(inputs))))


class TorchResidualUNet(nn.Module):
    """PyTorch implementation matched to ``ml.residual_net.ResidualUNet``.

    Inputs and outputs use PyTorch-native ``(batch, channels, y, x)`` layout.
    Layout conversion is intentionally performed outside timed benchmark
    regions. Layer widths, kernels, biases, downsampling convolutions,
    transposed convolutions, skip concatenations, and the residual head match
    the Flax architecture one-for-one.
    """

    def __init__(
        self,
        *,
        input_channels: int,
        depth: int,
        channels: Sequence[int],
        activation: str = "relu",
        kernel_size: int = 3,
        output_channels: int = 3,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("U-Net depth must be positive.")
        if len(channels) < depth:
            raise ValueError("The channel list must contain at least one value per U-Net depth.")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("The convolution kernel size must be a positive odd integer.")

        self.widths = tuple(int(value) for value in channels[:depth])
        self.depth = depth
        encoders: list[nn.Module] = []
        downsample: list[nn.Module] = []
        current_channels = int(input_channels)
        for level, width in enumerate(self.widths):
            encoders.append(
                TorchConvBlock(
                    current_channels,
                    width,
                    kernel_size,
                    activation,
                )
            )
            current_channels = width
            if level < depth - 1:
                downsample.append(nn.Conv2d(width, self.widths[level + 1], kernel_size=2, stride=2))
                current_channels = self.widths[level + 1]
        self.encoders = nn.ModuleList(encoders)
        self.downsample = nn.ModuleList(downsample)

        upsample: list[nn.Module] = []
        decoders: list[nn.Module] = []
        for level in range(depth - 2, -1, -1):
            width = self.widths[level]
            upsample.append(nn.ConvTranspose2d(current_channels, width, kernel_size=2, stride=2))
            decoders.append(
                TorchConvBlock(
                    width * 2,
                    width,
                    kernel_size,
                    activation,
                )
            )
            current_channels = width
        self.upsample = nn.ModuleList(upsample)
        self.decoders = nn.ModuleList(decoders)
        self.residual_head = nn.Conv2d(current_channels, output_channels, kernel_size=1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Predict residual channels in NCHW layout."""
        x = inputs
        skips: list[torch.Tensor] = []
        for level, encoder in enumerate(self.encoders):
            x = encoder(x)
            skips.append(x)
            if level < self.depth - 1:
                x = self.downsample[level](_same_stride_two(x))

        for decoder_index, level in enumerate(range(self.depth - 2, -1, -1)):
            skip = skips[level]
            x = self.upsample[decoder_index](x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = functional.interpolate(
                    x,
                    size=skip.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            x = torch.cat((x, skip), dim=1)
            x = self.decoders[decoder_index](x)
        return self.residual_head(x)


def count_torch_parameters(model: nn.Module) -> int:
    """Count all trainable scalar parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
