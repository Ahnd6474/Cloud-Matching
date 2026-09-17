from __future__ import annotations

import torch
from torch import Tensor, nn


def _batch_coeff(values: Tensor, levels: Tensor, ndim: int) -> Tensor:
    """Select one scalar per batch item and make it broadcastable."""
    selected = values.to(levels.device)[levels]
    return selected.reshape(levels.shape[0], *([1] * (ndim - 1)))


class VPNoiseSchedule(nn.Module):
    """Variance-preserving Gaussian corruption indexed from 0 to ``steps``.

    Level 0 is exactly clean. Levels 1..steps follow a linear beta schedule.
    The module also samples the arbitrary-skip posterior q(x_a | x_s, x_0),
    which combines every forward noise increment between answer level ``a``
    and input level ``s`` into one target transition distribution.
    """

    def __init__(
        self,
        steps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ) -> None:
        super().__init__()
        if steps < 1:
            raise ValueError("steps must be positive")
        if not 0.0 < beta_start <= beta_end < 1.0:
            raise ValueError("expected 0 < beta_start <= beta_end < 1")

        betas = torch.linspace(beta_start, beta_end, steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cat(
            [torch.ones(1, dtype=torch.float32), torch.cumprod(alphas, dim=0)]
        )
        self.steps = steps
        self.register_buffer("betas", betas)
        self.register_buffer("alpha_bars", alpha_bars)

    def q_sample(
        self,
        clean: Tensor,
        level: Tensor,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Draw x_level ~ q(x_level | x_0)."""
        self._validate_images_and_levels(clean, level)
        if noise is None:
            noise = torch.randn_like(clean)
        if noise.shape != clean.shape:
            raise ValueError("noise and clean must have identical shapes")

        alpha_bar = _batch_coeff(self.alpha_bars, level, clean.ndim)
        return alpha_bar.sqrt() * clean + (1.0 - alpha_bar).sqrt() * noise

    def posterior_mean_variance(
        self,
        clean: Tensor,
        later: Tensor,
        later_level: Tensor,
        earlier_level: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return q(x_a | x_s, x_0) for arbitrary levels 0 <= a <= s.

        This is the exact Gaussian posterior induced by the complete forward
        chain from answer level ``a`` to input level ``s``.
        """
        self._validate_images_and_levels(clean, later_level)
        self._validate_images_and_levels(later, earlier_level)
        if clean.shape != later.shape:
            raise ValueError("clean and later must have identical shapes")
        if torch.any(earlier_level > later_level):
            raise ValueError("earlier_level must be <= later_level")

        ab_a = _batch_coeff(self.alpha_bars, earlier_level, clean.ndim)
        ab_s = _batch_coeff(self.alpha_bars, later_level, clean.ndim)
        same_level = (earlier_level == later_level).reshape(
            earlier_level.shape[0], *([1] * (clean.ndim - 1))
        )

        denominator = (1.0 - ab_s).clamp_min(1e-12)
        alpha_s_given_a = (ab_s / ab_a.clamp_min(1e-12)).clamp(0.0, 1.0)

        clean_coeff = (
            ab_a.sqrt() * (1.0 - alpha_s_given_a) / denominator
        )
        later_coeff = (
            alpha_s_given_a.sqrt() * (1.0 - ab_a) / denominator
        )
        mean = clean_coeff * clean + later_coeff * later
        variance = (
            (1.0 - ab_a)
            * (1.0 - alpha_s_given_a)
            / denominator
        ).clamp_min(0.0)

        # q(x_s | x_s, x_0) is a point mass at x_s. Handling it explicitly
        # also covers the clean level where denominator would otherwise be 0.
        mean = torch.where(same_level, later, mean)
        variance = torch.where(same_level, torch.zeros_like(variance), variance)
        return mean, variance

    def sample_answer_cloud(
        self,
        clean: Tensor,
        current: Tensor,
        current_level: Tensor,
        answer_level: Tensor,
        samples: int,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Sample a [batch, samples, channels, height, width] target cloud."""
        if samples < 1:
            raise ValueError("samples must be positive")
        mean, variance = self.posterior_mean_variance(
            clean=clean,
            later=current,
            later_level=current_level,
            earlier_level=answer_level,
        )
        expected = (clean.shape[0], samples, *clean.shape[1:])
        if noise is None:
            noise = torch.randn(expected, device=clean.device, dtype=clean.dtype)
        if noise.shape != expected:
            raise ValueError(f"noise must have shape {expected}")
        return mean[:, None] + variance.sqrt()[:, None] * noise

    def _validate_images_and_levels(self, images: Tensor, levels: Tensor) -> None:
        if images.ndim != 4:
            raise ValueError("images must have shape [batch, channels, height, width]")
        if levels.ndim != 1 or levels.shape[0] != images.shape[0]:
            raise ValueError("levels must have shape [batch]")
        if levels.dtype != torch.long:
            raise TypeError("levels must use torch.long")
        if torch.any(levels < 0) or torch.any(levels > self.steps):
            raise ValueError(f"levels must lie in [0, {self.steps}]")
