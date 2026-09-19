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


class LaplacianCorrectionFeatures(nn.Module):
    """Full-band Laplacian-pyramid features with bounded total dimensionality."""

    def __init__(self, levels: int = 3) -> None:
        super().__init__()
        if levels < 1:
            raise ValueError("Laplacian features require at least one level")
        self.levels = levels

    def forward(self, images: Tensor) -> Tensor:
        features: list[Tensor] = []
        current = images
        for _ in range(self.levels):
            if min(current.shape[-2:]) < 2:
                break
            low = F.avg_pool2d(current, kernel_size=2, stride=2)
            reconstructed = F.interpolate(
                low, size=current.shape[-2:], mode="bilinear", align_corners=False
            )
            band = (current - reconstructed).flatten(1)
            features.append(band / math.sqrt(band.shape[1]))
            current = low
        low = current.flatten(1)
        features.append(low / math.sqrt(low.shape[1]))
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
        elementwise = F.smooth_l1_loss(predicted, target, reduction="none")
        if isinstance(self.features, MultiScaleCorrectionFeatures):
            # Every scale was already divided by sqrt(feature_dimension), so a
            # sum over feature coordinates is its normalized distance. Average
            # only over scales, samples, and batch. A second mean over the
            # concatenated feature axis would shrink the default loss by 336x
            # for the 4/8/16 feature pyramid.
            return elementwise.sum(dim=-1).mean() / len(self.features.output_sizes)
        # Preserve conventional elementwise averaging for custom extractors
        # whose feature normalization semantics are unknown.
        return elementwise.mean()


class PairedFullBandCloudLoss(nn.Module):
    """Match paired cloud samples in a complete, shift-preserving frequency pyramid.

    Unlike the pooled features used by :class:`PairedCorrectionLoss`, the
    undecimated bands below reconstruct the full-resolution error exactly.  A
    checkerboard or other high-frequency residual therefore cannot disappear
    through spatial averaging.  Samples retain their analytic posterior-noise
    pairing, so this is still a coupled cloud objective rather than an
    unordered set distance.
    """

    def __init__(
        self,
        levels: int = 3,
        charbonnier_epsilon: float = 1e-3,
        high_band_weight: float = 1.0,
        low_band_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if levels < 1:
            raise ValueError("full-band loss requires at least one level")
        if charbonnier_epsilon <= 0.0:
            raise ValueError("charbonnier_epsilon must be positive")
        if high_band_weight <= 0.0 or low_band_weight <= 0.0:
            raise ValueError("full-band weights must be positive")
        self.levels = levels
        self.charbonnier_epsilon = charbonnier_epsilon
        self.high_band_weight = high_band_weight
        self.low_band_weight = low_band_weight
        # Five-tap binomial filter used by an undecimated (a-trous) pyramid.
        # Dilation grows with level, but no spatial subsampling is performed.
        self.register_buffer(
            "kernel_1d",
            torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0,
            persistent=False,
        )

    def forward(self, predicted_cloud: Tensor, target_cloud: Tensor, current: Tensor) -> Tensor:
        if predicted_cloud.shape != target_cloud.shape:
            raise ValueError("paired full-band loss requires equal cloud shapes")
        if predicted_cloud.ndim != 5:
            raise ValueError("clouds must have shape [B, samples, C, H, W]")
        if current.shape != predicted_cloud[:, 0].shape:
            raise ValueError("current must have shape [B, C, H, W]")

        # Write this as correction matching to document the coupling semantics.
        # Algebraically, current cancels and leaves the paired full-image error.
        error = (predicted_cloud - current[:, None]) - (
            target_cloud - current[:, None]
        )
        detail = error.flatten(0, 1)
        total = detail.new_zeros(())
        total_weight = 0.0
        for level in range(self.levels):
            smooth = self._atrous_blur(detail, dilation=2**level)
            band = detail - smooth
            total = total + self.high_band_weight * self._charbonnier(band)
            total_weight += self.high_band_weight
            detail = smooth
        total = total + self.low_band_weight * self._charbonnier(detail)
        total_weight += self.low_band_weight
        return total / total_weight

    def _charbonnier(self, error: Tensor) -> Tensor:
        epsilon = self.charbonnier_epsilon
        return (torch.sqrt(error.float().square() + epsilon**2) - epsilon).mean()

    def _atrous_blur(self, image: Tensor, dilation: int) -> Tensor:
        channels = image.shape[1]
        kernel = self.kernel_1d.to(device=image.device, dtype=image.dtype)
        horizontal = kernel.reshape(1, 1, 1, 5).expand(channels, 1, 1, 5)
        vertical = kernel.reshape(1, 1, 5, 1).expand(channels, 1, 5, 1)
        padding = 2 * dilation
        # Replication padding is defined even for tiny smoke-test images where
        # the largest dilated kernel reaches beyond the opposite boundary.
        blurred = F.pad(image, (padding, padding, 0, 0), mode="replicate")
        blurred = F.conv2d(
            blurred, horizontal, groups=channels, dilation=(1, dilation)
        )
        blurred = F.pad(blurred, (0, 0, padding, padding), mode="replicate")
        return F.conv2d(
            blurred, vertical, groups=channels, dilation=(dilation, 1)
        )


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


class FullBandEnergyCorrectionCloudLoss(EnergyCorrectionCloudLoss):
    """Energy distance that preserves pixel-scale detail via a Laplacian pyramid."""

    def __init__(self, levels: int = 3) -> None:
        super().__init__(feature_extractor=LaplacianCorrectionFeatures(levels))


class SpatialNoiseCrossEntropyLoss(nn.Module):
    """Teach the sampler where target-cloud correction energy is concentrated.

    The predicted map remains unnormalized for sampling, so the cloud loss is
    free to learn its absolute scale.  Cross entropy sees only the normalized
    spatial allocation and therefore cannot impose a separate strength label.
    """

    def __init__(
        self,
        highpass: bool = True,
        kernel_size: int = 5,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("spatial CE kernel size must be a positive odd integer")
        if epsilon <= 0.0:
            raise ValueError("spatial CE epsilon must be positive")
        self.highpass = highpass
        self.kernel_size = kernel_size
        self.epsilon = epsilon

    def forward(
        self,
        predicted_energy: Tensor,
        target_cloud: Tensor,
        current: Tensor,
    ) -> Tensor:
        if predicted_energy.ndim == 4 and predicted_energy.shape[1] == 1:
            predicted_energy = predicted_energy[:, 0]
        if predicted_energy.ndim != 3:
            raise ValueError("predicted energy must have shape [B,H,W]")
        if target_cloud.ndim != 5:
            raise ValueError("target cloud must have shape [B,S,C,H,W]")
        if current.shape != target_cloud[:, 0].shape:
            raise ValueError("current must have shape [B,C,H,W]")
        if predicted_energy.shape != (
            current.shape[0],
            current.shape[2],
            current.shape[3],
        ):
            raise ValueError("predicted energy spatial shape must match current")

        correction = target_cloud - current[:, None]
        if self.highpass and self.kernel_size > 1:
            flat = correction.flatten(0, 1)
            smooth = F.avg_pool2d(
                flat,
                self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
            )
            correction = (flat - smooth).reshape_as(correction)
        target_energy = correction.float().square().mean(dim=(1, 2))
        target_total = target_energy.sum(dim=(-2, -1), keepdim=True)
        valid = target_total.flatten() > self.epsilon
        if not torch.any(valid):
            return predicted_energy.sum() * 0.0

        target_probability = target_energy[valid] / target_total[valid].clamp_min(
            self.epsilon
        )
        predicted = predicted_energy[valid].float().clamp_min(self.epsilon)
        predicted_probability = predicted / predicted.sum(
            dim=(-2, -1), keepdim=True
        ).clamp_min(self.epsilon)
        cross_entropy = -(
            target_probability
            * predicted_probability.clamp_min(self.epsilon).log()
        ).sum(dim=(-2, -1)).mean()
        # Subtract the target entropy. This is KL(q || p), which has exactly
        # the same gradient as CE(q, p) but removes the resolution-dependent
        # constant (log(HW) for a uniform target) from training curves.
        target_entropy = -(
            target_probability
            * target_probability.clamp_min(self.epsilon).log()
        ).sum(dim=(-2, -1)).mean()
        return cross_entropy - target_entropy


def _cloud_features(features: nn.Module, cloud: Tensor) -> Tensor:
    if cloud.ndim != 5:
        raise ValueError("cloud must have shape [B, samples, C, H, W]")
    batch, samples = cloud.shape[:2]
    flat_cloud = cloud.reshape(batch * samples, *cloud.shape[2:])
    return features(flat_cloud).reshape(batch, samples, -1)


def is_paired_cloud_loss(name: str) -> bool:
    return name.strip().lower() in {"paired", "paired_full_band"}


def build_cloud_loss(
    name: str,
    blur: float = 0.05,
    full_band_levels: int = 3,
    full_band_charbonnier_epsilon: float = 1e-3,
    full_band_high_weight: float = 1.0,
    full_band_low_weight: float = 1.0,
) -> nn.Module:
    normalized = name.strip().lower()
    if normalized == "paired":
        return PairedCorrectionLoss()
    if normalized == "paired_full_band":
        return PairedFullBandCloudLoss(
            levels=full_band_levels,
            charbonnier_epsilon=full_band_charbonnier_epsilon,
            high_band_weight=full_band_high_weight,
            low_band_weight=full_band_low_weight,
        )
    if normalized == "energy":
        return EnergyCorrectionCloudLoss()
    if normalized == "energy_full_band":
        return FullBandEnergyCorrectionCloudLoss(levels=full_band_levels)
    if normalized == "sinkhorn":
        return SinkhornCorrectionCloudLoss(blur=blur)
    raise ValueError(
        f"unknown loss '{name}'; choose paired, paired_full_band, energy, "
        "energy_full_band, or sinkhorn"
    )
