from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from geomloss import SamplesLoss
from torch import Tensor, nn


class MultiScaleCorrectionFeatures(nn.Module):
    """Parameter-free multi-scale features for image corrections.

    This keeps the architecture's image encoder shared while preventing the
    distance space itself from collapsing during early experiments.
    """

    def __init__(self, output_sizes: tuple[int, ...] = (4, 8, 16)) -> None:
        super().__init__()
        self.output_sizes = output_sizes

    def forward(self, images: Tensor) -> Tensor:
        features: list[Tensor] = []
        for size in self.output_sizes:
            pooled = F.adaptive_avg_pool2d(images, (size, size)).flatten(1)
            features.append(pooled / math.sqrt(pooled.shape[1]))
        return torch.cat(features, dim=1)


class SinkhornCorrectionCloudLoss(nn.Module):
    """Sinkhorn divergence between predicted and target correction clouds."""

    def __init__(
        self,
        blur: float = 0.05,
        scaling: float = 0.8,
        feature_extractor: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.features = feature_extractor or MultiScaleCorrectionFeatures()
        self.sinkhorn = SamplesLoss(
            loss="sinkhorn",
            p=2,
            blur=blur,
            scaling=scaling,
            debias=True,
            backend="tensorized",
        )

    def forward(
        self,
        predicted_cloud: Tensor,
        target_cloud: Tensor,
        current: Tensor,
    ) -> Tensor:
        """Compute loss for clouds shaped [B, M/N, C, H, W]."""
        if predicted_cloud.ndim != 5 or target_cloud.ndim != 5:
            raise ValueError("clouds must have shape [B, samples, C, H, W]")
        if predicted_cloud.shape[0] != target_cloud.shape[0]:
            raise ValueError("predicted and target batch sizes must match")
        if predicted_cloud.shape[2:] != target_cloud.shape[2:]:
            raise ValueError("predicted and target image shapes must match")
        if current.shape != predicted_cloud[:, 0].shape:
            raise ValueError("current must have shape [B, C, H, W]")

        current_features = self.features(current)
        predicted = _cloud_features(self.features, predicted_cloud) - current_features[:, None]
        target = _cloud_features(self.features, target_cloud) - current_features[:, None]
        loss = self.sinkhorn(predicted, target)
        return loss.mean()


class PairedCorrectionLoss(nn.Module):
    """Directly match samples that share the posterior reparameterization noise."""

    def __init__(self, feature_extractor: nn.Module | None = None) -> None:
        super().__init__()
        self.features = feature_extractor or MultiScaleCorrectionFeatures()

    def forward(self, predicted_cloud: Tensor, target_cloud: Tensor, current: Tensor) -> Tensor:
        if predicted_cloud.shape != target_cloud.shape:
            raise ValueError("paired loss requires equal predicted and target cloud shapes")
        current_features = self.features(current)
        predicted = _cloud_features(self.features, predicted_cloud) - current_features[:, None]
        target = _cloud_features(self.features, target_cloud) - current_features[:, None]
        return F.smooth_l1_loss(predicted, target)


class EnergyCorrectionCloudLoss(nn.Module):
    """Energy distance between implicit correction ensembles."""

    def __init__(self, feature_extractor: nn.Module | None = None) -> None:
        super().__init__()
        self.features = feature_extractor or MultiScaleCorrectionFeatures()

    def forward(self, predicted_cloud: Tensor, target_cloud: Tensor, current: Tensor) -> Tensor:
        current_features = self.features(current)
        predicted = _cloud_features(self.features, predicted_cloud) - current_features[:, None]
        target = _cloud_features(self.features, target_cloud) - current_features[:, None]
        cross = torch.cdist(predicted, target, p=2).mean(dim=(1, 2))
        predicted_self = torch.cdist(predicted, predicted, p=2).mean(dim=(1, 2))
        target_self = torch.cdist(target, target, p=2).mean(dim=(1, 2))
        return (2.0 * cross - predicted_self - target_self).mean()


def _cloud_features(features: nn.Module, cloud: Tensor) -> Tensor:
    if cloud.ndim != 5:
        raise ValueError("cloud must have shape [B, samples, C, H, W]")
    batch, samples = cloud.shape[:2]
    flat_cloud = cloud.reshape(batch * samples, *cloud.shape[2:])
    return features(flat_cloud).reshape(batch, samples, -1)


def build_cloud_loss(name: str, blur: float = 0.05) -> nn.Module:
    normalized = name.strip().lower()
    if normalized == "paired":
        return PairedCorrectionLoss()
    if normalized == "energy":
        return EnergyCorrectionCloudLoss()
    if normalized == "sinkhorn":
        return SinkhornCorrectionCloudLoss(blur=blur)
    raise ValueError(f"unknown loss '{name}'; choose paired, energy, or sinkhorn")
