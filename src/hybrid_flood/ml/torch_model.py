"""Active PyTorch residual network shared without modification by V1 and V2."""

from hybrid_flood.benchmark.pytorch_residual_net import (
    TorchResidualUNet,
    count_torch_parameters,
)

PyTorchResidualUNet = TorchResidualUNet

__all__ = ["PyTorchResidualUNet", "count_torch_parameters"]
