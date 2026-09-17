from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class GoalDetailCorruptor(nn.Module):
    """Remove image-goal detail without using a second encoder.

    Blur, downsample/upsample, and rectangular masking are sampled independently
    per image. Inputs and outputs stay in [-1, 1].
    """

    def __init__(
        self,
        blur_probability: float = 0.35,
        downsample_probability: float = 0.50,
        mask_probability: float = 0.25,
        minimum_scale: float = 0.25,
        maximum_mask_fraction: float = 0.30,
    ) -> None:
        super().__init__()
        self.blur_probability = blur_probability
        self.downsample_probability = downsample_probability
        self.mask_probability = mask_probability
        self.minimum_scale = minimum_scale
        self.maximum_mask_fraction = maximum_mask_fraction

    def forward(self, images: Tensor) -> Tensor:
        outputs = []
        for image in images:
            item = image[None]
            if torch.rand((), device=image.device) < self.blur_probability:
                item = self._blur(item)
            if torch.rand((), device=image.device) < self.downsample_probability:
                item = self._downsample(item)
            if torch.rand((), device=image.device) < self.mask_probability:
                item = self._mask(item)
            outputs.append(item[0])
        return torch.stack(outputs).clamp(-1.0, 1.0)

    @staticmethod
    def _blur(image: Tensor) -> Tensor:
        channels = image.shape[1]
        sigma = float(torch.empty(1).uniform_(0.6, 1.8))
        radius = 2
        coordinates = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
        kernel_1d = torch.exp(-(coordinates**2) / (2.0 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel = kernel_2d.expand(channels, 1, -1, -1)
        return F.conv2d(image, kernel, padding=radius, groups=channels)

    def _downsample(self, image: Tensor) -> Tensor:
        height, width = image.shape[-2:]
        scale = float(torch.empty(1).uniform_(self.minimum_scale, 0.75))
        low_size = (max(2, round(height * scale)), max(2, round(width * scale)))
        low = F.interpolate(image, size=low_size, mode="bilinear", align_corners=False)
        return F.interpolate(low, size=(height, width), mode="bilinear", align_corners=False)

    def _mask(self, image: Tensor) -> Tensor:
        height, width = image.shape[-2:]
        max_h = max(1, round(height * self.maximum_mask_fraction))
        max_w = max(1, round(width * self.maximum_mask_fraction))
        mask_h = int(torch.randint(1, max_h + 1, ()).item())
        mask_w = int(torch.randint(1, max_w + 1, ()).item())
        top = int(torch.randint(0, height - mask_h + 1, ()).item())
        left = int(torch.randint(0, width - mask_w + 1, ()).item())
        output = image.clone()
        fill = image.mean(dim=(-2, -1), keepdim=True)
        output[:, :, top : top + mask_h, left : left + mask_w] = fill
        return output

