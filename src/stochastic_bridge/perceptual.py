from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class EfficientNetB0Features(nn.Module):
    """Frozen multi-scale EfficientNet-B0 features for perceptual losses.

    Inputs use the bridge model's ``[-1, 1]`` range. Spatial resolution is
    intentionally left unchanged: a 128x128 bridge crop produces feature maps
    at 64, 32, 16, 8, and 4 pixels for the default stages. Classification
    pooling and the linear head are not part of the forward path.
    """

    def __init__(
        self,
        *,
        pretrained: bool = True,
        stages: Sequence[int] = (1, 2, 3, 5, 7),
    ) -> None:
        super().__init__()
        if not stages:
            raise ValueError("at least one EfficientNet stage is required")
        if tuple(sorted(set(stages))) != tuple(stages):
            raise ValueError("stages must be unique and strictly increasing")
        if stages[0] < 0 or stages[-1] > 8:
            raise ValueError("EfficientNet-B0 stage indices must be in [0, 8]")

        self.stages = tuple(int(stage) for stage in stages)
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_b0(weights=weights)
        # Discard classification and every feature block deeper than the last
        # requested stage so unused 1280-channel weights consume no VRAM.
        blocks = list(backbone.features.children())[: self.stages[-1] + 1]
        self.features = nn.Sequential(*blocks)
        self.register_buffer(
            "image_mean",
            torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1),
            persistent=False,
        )
        self.requires_grad_(False)
        self.train(False)

    def train(self, mode: bool = True) -> EfficientNetB0Features:
        # Frozen BatchNorm statistics must not drift when a parent loss/model
        # is switched into training mode.
        super().train(False)
        return self

    def forward(self, image: Tensor) -> tuple[Tensor, ...]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("expected an NCHW RGB tensor")
        value = (image + 1.0) * 0.5
        value = (value - self.image_mean) / self.image_std
        outputs: list[Tensor] = []
        requested = set(self.stages)
        for index, block in enumerate(self.features):
            value = block(value)
            if index in requested:
                outputs.append(value)
            if index >= self.stages[-1]:
                break
        return tuple(outputs)
