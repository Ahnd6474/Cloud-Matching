from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.transforms.functional import gaussian_blur

from .schedule import VPNoiseSchedule


SUPPORTED_CORRUPTIONS = (
    "white_gaussian",
    "local_gaussian",
    "edge_gaussian",
    "low_frequency_gaussian",
    "high_frequency_gaussian",
    "band_pass_gaussian",
    "channel_correlated_gaussian",
    "low_rank_gaussian",
    "anisotropic_gaussian",
    "signal_dependent_gaussian",
    "student_t",
    "laplace",
    "poisson",
    "speckle",
    "impulse",
    "blur",
    "downsample",
    "mask",
    "texture_suppress",
    "edge_erase",
)


class CorruptionMixture(nn.Module):
    """On-the-fly image corruption registry used to construct bridge states.

    Gaussian variants change spatial, spectral, directional, or channel
    covariance. Non-Gaussian and deterministic degradations use a marginal
    answer-cloud simulator instead of the analytic VP posterior.
    """

    def __init__(
        self,
        schedule: VPNoiseSchedule,
        weights: Mapping[str, float],
        spatial_floor: float = 0.08,
        student_t_df: float = 3.0,
    ) -> None:
        super().__init__()
        unknown = set(weights) - set(SUPPORTED_CORRUPTIONS)
        if unknown:
            raise ValueError(f"unknown corruption types: {sorted(unknown)}")
        positive = {name: float(weight) for name, weight in weights.items() if weight > 0}
        if not positive:
            raise ValueError("at least one corruption weight must be positive")
        if not 0.0 <= spatial_floor < 1.0:
            raise ValueError("spatial_floor must lie in [0, 1)")
        if student_t_df <= 2.0:
            raise ValueError("student_t_df must exceed 2 for finite variance")
        self.schedule = schedule
        self.names = tuple(positive)
        self.register_buffer(
            "probabilities",
            torch.tensor(list(positive.values()), dtype=torch.float32)
            / sum(positive.values()),
        )
        self.spatial_floor = spatial_floor
        self.student_t_df = student_t_df

    def sample_types(self, batch_size: int, device: torch.device) -> tuple[str, ...]:
        indices = torch.multinomial(
            self.probabilities.to(device), batch_size, replacement=True
        ).tolist()
        return tuple(self.names[index] for index in indices)

    def corrupt(
        self,
        clean: Tensor,
        levels: Tensor,
        corruption_types: Sequence[str] | None = None,
    ) -> Tensor:
        if clean.ndim != 4 or levels.shape != (clean.shape[0],):
            raise ValueError("expected clean [B,C,H,W] and levels [B]")
        if corruption_types is None:
            corruption_types = self.sample_types(clean.shape[0], clean.device)
        if len(corruption_types) != clean.shape[0]:
            raise ValueError("one corruption type is required per batch item")
        outputs = [
            self._corrupt_one(image[None], int(level.item()), name)
            for image, level, name in zip(clean, levels, corruption_types, strict=True)
        ]
        return torch.cat(outputs, dim=0).clamp(-1.0, 1.0)

    def sample_cloud(
        self,
        clean: Tensor,
        levels: Tensor,
        corruption_types: Sequence[str],
        samples: int,
    ) -> Tensor:
        clouds = [self.corrupt(clean, levels, corruption_types) for _ in range(samples)]
        return torch.stack(clouds, dim=1)

    def _corrupt_one(self, clean: Tensor, level: int, name: str) -> Tensor:
        if name not in SUPPORTED_CORRUPTIONS:
            raise ValueError(f"unsupported corruption: {name}")
        if level <= 0:
            return clean
        severity = min(max(level / self.schedule.steps, 0.0), 1.0)
        alpha_bar = self.schedule.alpha_bars[level].to(clean).reshape(1, 1, 1, 1)

        if name == "white_gaussian":
            return self._vp(clean, alpha_bar, torch.randn_like(clean))
        if name == "local_gaussian":
            mask = self._local_mask(clean)
            return self._spatial_vp(clean, alpha_bar, mask, torch.randn_like(clean))
        if name == "edge_gaussian":
            mask = self._edge_mask(clean)
            return self._spatial_vp(clean, alpha_bar, mask, torch.randn_like(clean))
        if name == "low_frequency_gaussian":
            noise = self._low_frequency_noise(clean, divisor=8)
            return self._vp(clean, alpha_bar, noise)
        if name == "high_frequency_gaussian":
            white = torch.randn_like(clean)
            noise = white - self._gaussian_blur(white, sigma=1.4)
            return self._vp(clean, alpha_bar, self._normalize(noise))
        if name == "band_pass_gaussian":
            white = torch.randn_like(clean)
            noise = self._gaussian_blur(white, 0.7) - self._gaussian_blur(white, 2.2)
            return self._vp(clean, alpha_bar, self._normalize(noise))
        if name == "channel_correlated_gaussian":
            return self._vp(clean, alpha_bar, self._channel_correlated_noise(clean))
        if name == "low_rank_gaussian":
            noise = self._low_frequency_noise(clean, divisor=16)
            return self._vp(clean, alpha_bar, noise)
        if name == "anisotropic_gaussian":
            return self._vp(clean, alpha_bar, self._anisotropic_noise(clean))
        if name == "signal_dependent_gaussian":
            scale = (0.15 + 0.85 * ((clean + 1.0) / 2.0).sqrt()).detach()
            noise = self._normalize(torch.randn_like(clean) * scale)
            return self._vp(clean, alpha_bar, noise)
        if name == "student_t":
            normal = torch.randn_like(clean)
            degrees = torch.tensor(
                self.student_t_df, device=clean.device, dtype=clean.dtype
            )
            chi_squared = torch.distributions.Chi2(degrees).sample(clean.shape)
            noise = normal / (chi_squared / self.student_t_df).sqrt().clamp_min(1e-5)
            noise = noise * math.sqrt((self.student_t_df - 2.0) / self.student_t_df)
            return self._vp(clean, alpha_bar, noise.clamp(-10.0, 10.0))
        if name == "laplace":
            scale = torch.tensor(1.0 / math.sqrt(2.0), device=clean.device, dtype=clean.dtype)
            noise = torch.distributions.Laplace(torch.zeros_like(clean), scale).sample()
            return self._vp(clean, alpha_bar, noise)
        if name == "poisson":
            return self._poisson(clean, severity)
        if name == "speckle":
            return self._speckle(clean, severity)
        if name == "impulse":
            return self._impulse(clean, severity)
        if name == "blur":
            return self._gaussian_blur(clean, sigma=0.2 + 3.0 * severity)
        if name == "downsample":
            return self._downsample(clean, severity)
        if name == "mask":
            return self._random_mask(clean, severity)
        if name == "texture_suppress":
            return self._texture_suppress(clean, severity)
        if name == "edge_erase":
            return self._edge_erase(clean, severity)
        raise AssertionError("unreachable")

    @staticmethod
    def _vp(clean: Tensor, alpha_bar: Tensor, noise: Tensor) -> Tensor:
        return alpha_bar.sqrt() * clean + (1.0 - alpha_bar).sqrt() * noise

    def _spatial_vp(
        self, clean: Tensor, alpha_bar: Tensor, mask: Tensor, noise: Tensor
    ) -> Tensor:
        strength = self.spatial_floor + (1.0 - self.spatial_floor) * mask
        spatial_alpha_bar = alpha_bar.pow(strength)
        return self._vp(clean, spatial_alpha_bar, noise)

    @staticmethod
    def _normalize(noise: Tensor) -> Tensor:
        dims = tuple(range(1, noise.ndim))
        centered = noise - noise.mean(dim=dims, keepdim=True)
        return centered / centered.square().mean(dim=dims, keepdim=True).sqrt().clamp_min(1e-6)

    @staticmethod
    def _local_mask(reference: Tensor) -> Tensor:
        _, _, height, width = reference.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=reference.device, dtype=reference.dtype),
            torch.linspace(-1.0, 1.0, width, device=reference.device, dtype=reference.dtype),
            indexing="ij",
        )
        center_x = torch.empty((), device=reference.device).uniform_(-0.6, 0.6)
        center_y = torch.empty((), device=reference.device).uniform_(-0.6, 0.6)
        radius_x = torch.empty((), device=reference.device).uniform_(0.15, 0.65)
        radius_y = torch.empty((), device=reference.device).uniform_(0.15, 0.65)
        distance = ((xx - center_x) / radius_x).square() + ((yy - center_y) / radius_y).square()
        return torch.exp(-1.5 * distance)[None, None]

    def _edge_mask(self, reference: Tensor) -> Tensor:
        gray = reference.mean(dim=1, keepdim=True)
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            device=reference.device,
            dtype=reference.dtype,
        ).reshape(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(-1, -2)
        gx = F.conv2d(gray, sobel_x, padding=1)
        gy = F.conv2d(gray, sobel_y, padding=1)
        magnitude = (gx.square() + gy.square() + 1e-8).sqrt()
        magnitude = magnitude / magnitude.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        magnitude = F.max_pool2d(magnitude, kernel_size=5, stride=1, padding=2)
        return self._gaussian_blur(magnitude, sigma=1.0).clamp(0.0, 1.0)

    def _texture_mask(self, reference: Tensor) -> Tensor:
        """Softly select high-frequency texture while excluding strong edges."""
        gray = reference.mean(dim=1, keepdim=True)
        high_frequency = (gray - self._gaussian_blur(gray, sigma=1.2)).abs()
        local_energy = self._gaussian_blur(high_frequency, sigma=1.0)
        scale = local_energy.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
        texture = (local_energy / (3.0 * scale)).clamp(0.0, 1.0)
        # Object boundaries and thin lines belong to the complementary edge
        # corruption; keeping them here makes this operator texture-specific.
        texture = texture * (1.0 - self._edge_mask(reference)).square()
        return self._gaussian_blur(texture, sigma=0.7).clamp(0.0, 1.0)

    def _texture_suppress(self, clean: Tensor, severity: float) -> Tensor:
        smooth = self._gaussian_blur(clean, sigma=0.5 + 2.5 * severity)
        mask = self._texture_mask(clean) * severity
        return clean * (1.0 - mask) + smooth * mask

    def _edge_erase(self, clean: Tensor, severity: float) -> Tensor:
        # Replace only a dilated soft band around lines/boundaries with a local
        # low-pass estimate. Flat areas and most stochastic texture survive.
        edge = self._edge_mask(clean)
        edge = F.max_pool2d(edge, kernel_size=3, stride=1, padding=1)
        mask = (edge * severity).clamp(0.0, 1.0)
        smooth = self._gaussian_blur(clean, sigma=0.8 + 3.2 * severity)
        return clean * (1.0 - mask) + smooth * mask

    def _low_frequency_noise(self, reference: Tensor, divisor: int) -> Tensor:
        height, width = reference.shape[-2:]
        low = torch.randn(
            reference.shape[0],
            reference.shape[1],
            max(2, height // divisor),
            max(2, width // divisor),
            device=reference.device,
            dtype=reference.dtype,
        )
        noise = F.interpolate(low, size=(height, width), mode="bicubic", align_corners=False)
        return self._normalize(noise)

    def _channel_correlated_noise(self, reference: Tensor) -> Tensor:
        if reference.shape[1] != 3:
            return self._normalize(torch.randn_like(reference))
        basis = torch.randn(
            reference.shape[0], 3, *reference.shape[-2:], device=reference.device, dtype=reference.dtype
        )
        matrices = torch.tensor(
            [
                [[1.0, 0.8, 0.8], [0.8, 1.0, 0.8], [0.8, 0.8, 1.0]],
                [[1.0, -0.6, 0.0], [-0.6, 1.0, 0.0], [0.0, 0.0, 0.4]],
                [[0.5, 0.2, -0.5], [0.2, 0.4, 0.2], [-0.5, 0.2, 0.5]],
            ],
            device=reference.device,
            dtype=reference.dtype,
        )
        matrix = matrices[int(torch.randint(0, len(matrices), ()).item())]
        mixed = torch.einsum("ij,bjhw->bihw", matrix, basis)
        return self._normalize(mixed)

    def _anisotropic_noise(self, reference: Tensor) -> Tensor:
        noise = torch.randn_like(reference)
        horizontal = bool(torch.randint(0, 2, ()).item())
        kernel_size = 9
        if horizontal:
            noise = F.avg_pool2d(
                noise, kernel_size=(1, kernel_size), stride=1, padding=(0, kernel_size // 2)
            )
        else:
            noise = F.avg_pool2d(
                noise, kernel_size=(kernel_size, 1), stride=1, padding=(kernel_size // 2, 0)
            )
        return self._normalize(noise)

    @staticmethod
    def _gaussian_blur(image: Tensor, sigma: float) -> Tensor:
        radius = max(1, min(6, math.ceil(3.0 * sigma)))
        kernel_size = 2 * radius + 1
        return gaussian_blur(image, [kernel_size, kernel_size], [sigma, sigma])

    @staticmethod
    def _poisson(clean: Tensor, severity: float) -> Tensor:
        image = (clean + 1.0) / 2.0
        peak = 1.0 + 63.0 * (1.0 - severity) ** 2
        noisy = torch.poisson(image.clamp_min(0.0) * peak) / peak
        return noisy.clamp(0.0, 1.0) * 2.0 - 1.0

    @staticmethod
    def _speckle(clean: Tensor, severity: float) -> Tensor:
        image = (clean + 1.0) / 2.0
        noisy = image * (1.0 + severity * torch.randn_like(image))
        return noisy.clamp(0.0, 1.0) * 2.0 - 1.0

    @staticmethod
    def _impulse(clean: Tensor, severity: float) -> Tensor:
        probability = min(0.65, 0.65 * severity)
        selector = torch.rand_like(clean[:, :1])
        salt = torch.where(torch.rand_like(clean[:, :1]) < 0.5, -torch.ones_like(clean[:, :1]), torch.ones_like(clean[:, :1]))
        return torch.where(selector < probability, salt.expand_as(clean), clean)

    @staticmethod
    def _downsample(clean: Tensor, severity: float) -> Tensor:
        height, width = clean.shape[-2:]
        scale = max(0.08, 1.0 - 0.92 * severity)
        low_size = (max(2, round(height * scale)), max(2, round(width * scale)))
        low = F.interpolate(clean, size=low_size, mode="bilinear", align_corners=False)
        return F.interpolate(low, size=(height, width), mode="bilinear", align_corners=False)

    @staticmethod
    def _random_mask(clean: Tensor, severity: float) -> Tensor:
        probability = min(0.85, 0.85 * severity)
        grid_h = max(2, clean.shape[-2] // 8)
        grid_w = max(2, clean.shape[-1] // 8)
        keep = (
            torch.rand(clean.shape[0], 1, grid_h, grid_w, device=clean.device) > probability
        ).to(clean.dtype)
        keep = F.interpolate(keep, size=clean.shape[-2:], mode="nearest")
        return clean * keep
