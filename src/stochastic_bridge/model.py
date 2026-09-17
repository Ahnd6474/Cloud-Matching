from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x) + self.skip(x)


class SharedPyramidEncoder(nn.Module):
    """One encoder instance used for both current and goal images."""

    def __init__(self, in_channels: int, base_channels: int) -> None:
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.blocks = nn.ModuleList(
            [
                ConvBlock(in_channels, channels[0]),
                ConvBlock(channels[0], channels[1]),
                ConvBlock(channels[1], channels[2]),
                ConvBlock(channels[2], channels[3]),
            ]
        )
        self.downsamples = nn.ModuleList(
            [
                nn.Conv2d(channels[0], channels[0], 4, stride=2, padding=1),
                nn.Conv2d(channels[1], channels[1], 4, stride=2, padding=1),
                nn.Conv2d(channels[2], channels[2], 4, stride=2, padding=1),
            ]
        )
        self.channels = channels

    def forward(self, image: Tensor) -> list[Tensor]:
        features: list[Tensor] = []
        x = image
        for index, block in enumerate(self.blocks):
            x = block(x)
            features.append(x)
            if index < len(self.downsamples):
                x = self.downsamples[index](x)
        return features


class SharedViTPyramidEncoder(nn.Module):
    """ViT encoder whose token grid is projected into decoder pyramid features."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        heads: int,
        depth: int,
        patch_size: int = 8,
    ) -> None:
        super().__init__()
        if patch_size != 8:
            raise ValueError("the current three-stage decoder requires vit_patch_size=8")
        dim = base_channels * 8
        if dim % heads:
            raise ValueError("base_channels * 8 must be divisible by heads")
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(
            in_channels, dim, kernel_size=patch_size, stride=patch_size
        )
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=4 * dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=depth, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(dim)
        self.to_quarter = nn.Sequential(
            nn.ConvTranspose2d(dim, base_channels * 4, 4, stride=2, padding=1),
            ConvBlock(base_channels * 4, base_channels * 4),
        )
        self.to_half = nn.Sequential(
            nn.ConvTranspose2d(
                base_channels * 4, base_channels * 2, 4, stride=2, padding=1
            ),
            ConvBlock(base_channels * 2, base_channels * 2),
        )
        self.to_full = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),
            ConvBlock(base_channels, base_channels),
        )
        self.channels = [base_channels, base_channels * 2, base_channels * 4, dim]

    def forward(self, image: Tensor) -> list[Tensor]:
        bottleneck = self.patch_embed(image)
        height, width = bottleneck.shape[-2:]
        tokens = bottleneck.flatten(2).transpose(1, 2)
        position = _sinusoidal_2d_position(
            height, width, tokens.shape[-1], image.device, image.dtype
        )
        tokens = self.norm(self.transformer(tokens + position))
        bottleneck = tokens.transpose(1, 2).reshape(
            image.shape[0], tokens.shape[-1], height, width
        )
        quarter = self.to_quarter(bottleneck)
        half = self.to_half(quarter)
        full = self.to_full(half)
        return [full, half, quarter, bottleneck]


def _sinusoidal_2d_position(
    height: int, width: int, dim: int, device: torch.device, dtype: torch.dtype
) -> Tensor:
    if dim % 4:
        raise ValueError("attention dimension must be divisible by 4")
    quarter = dim // 4
    frequency = torch.exp(
        -math.log(10_000.0)
        * torch.arange(quarter, device=device, dtype=torch.float32)
        / max(quarter - 1, 1)
    )
    y = torch.arange(height, device=device, dtype=torch.float32)[:, None] * frequency
    x = torch.arange(width, device=device, dtype=torch.float32)[:, None] * frequency
    y_embedding = torch.cat([y.sin(), y.cos()], dim=1)[:, None].expand(-1, width, -1)
    x_embedding = torch.cat([x.sin(), x.cos()], dim=1)[None].expand(height, -1, -1)
    position = torch.cat([y_embedding, x_embedding], dim=-1)
    return position.reshape(1, height * width, dim).to(dtype=dtype)


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, current: Tensor, goal: Tensor) -> Tensor:
        goal_norm = self.context_norm(goal)
        attended, _ = self.attention(
            self.query_norm(current), goal_norm, goal_norm, need_weights=False
        )
        current = current + attended
        return current + self.ff(current)


class DecoderStage(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
        self.block = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class StochasticImageBridge(nn.Module):
    """Shared encoder, cross-attention, spatial noise, and residual decoder."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        heads: int = 8,
        attention_depth: int = 2,
        max_residual: float = 1.0,
        encoder_type: str = "cnn",
        vit_depth: int = 4,
        vit_patch_size: int = 8,
        image_size: int | None = None,
        **legacy: object,
    ) -> None:
        super().__init__()
        del image_size
        if "dim" in legacy:
            base_channels = max(8, int(legacy.pop("dim")) // 4)
        legacy.pop("patch_size", None)
        legacy.pop("noise_dim", None)
        if legacy:
            raise TypeError(f"unexpected model arguments: {sorted(legacy)}")

        bottleneck_channels = base_channels * 8
        if bottleneck_channels % heads:
            raise ValueError("base_channels * 8 must be divisible by heads")
        if bottleneck_channels % 4:
            raise ValueError("base_channels * 8 must be divisible by 4")

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.max_residual = max_residual
        normalized_encoder = encoder_type.strip().lower()
        if normalized_encoder == "cnn":
            self.encoder: nn.Module = SharedPyramidEncoder(in_channels, base_channels)
        elif normalized_encoder == "vit":
            self.encoder = SharedViTPyramidEncoder(
                in_channels=in_channels,
                base_channels=base_channels,
                heads=heads,
                depth=vit_depth,
                patch_size=vit_patch_size,
            )
        else:
            raise ValueError("encoder_type must be 'cnn' or 'vit'")
        self.encoder_type = normalized_encoder
        self.attention = nn.ModuleList(
            [CrossAttentionBlock(bottleneck_channels, heads) for _ in range(attention_depth)]
        )
        self.noise_projection = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(bottleneck_channels, bottleneck_channels, 1),
        )
        channels = self.encoder.channels  # type: ignore[attr-defined]
        self.decoder = nn.ModuleList(
            [
                DecoderStage(channels[3], channels[2], channels[2]),
                DecoderStage(channels[2], channels[1], channels[1]),
                DecoderStage(channels[1], channels[0], channels[0]),
            ]
        )
        self.decoder_noise = nn.ModuleList(
            [
                nn.Conv2d(in_channels, channels[2], 3, padding=1),
                nn.Conv2d(in_channels, channels[1], 3, padding=1),
                nn.Conv2d(in_channels, channels[0], 3, padding=1),
            ]
        )
        self.output = nn.Conv2d(channels[0], 2 * in_channels, 3, padding=1)

    def encode_condition(self, current: Tensor, goal: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Compute deterministic context once before drawing cloud samples."""
        self._validate_images(current, goal)
        # One larger shared-encoder call has better accelerator utilization and
        # fewer kernel launches than two independent calls with identical weights.
        batch = current.shape[0]
        combined_features = self.encoder(torch.cat([current, goal], dim=0))
        current_features = [feature[:batch] for feature in combined_features]
        goal_features = [feature[batch:] for feature in combined_features]
        current_bottleneck = current_features[-1]
        goal_bottleneck = goal_features[-1]
        height, width = current_bottleneck.shape[-2:]
        current_tokens = current_bottleneck.flatten(2).transpose(1, 2)
        goal_tokens = goal_bottleneck.flatten(2).transpose(1, 2)
        position = _sinusoidal_2d_position(
            height,
            width,
            current_tokens.shape[-1],
            current.device,
            current.dtype,
        )
        current_tokens = current_tokens + position
        goal_tokens = goal_tokens + position
        for block in self.attention:
            current_tokens = block(current_tokens, goal_tokens)
        fused = current_tokens.transpose(1, 2).reshape_as(current_bottleneck)
        return fused, current_features[:-1]

    def forward(
        self,
        current: Tensor,
        goal: Tensor,
        samples: int = 4,
        noise: Tensor | None = None,
    ) -> Tensor:
        fused, current_skips = self.encode_condition(current, goal)
        batch, _, height, width = current.shape
        if samples < 1:
            raise ValueError("samples must be positive")
        if noise is None:
            noise = torch.randn(
                batch,
                samples,
                self.in_channels,
                height,
                width,
                device=current.device,
                dtype=current.dtype,
            )
        expected = (batch, samples, self.in_channels, height, width)
        if noise.shape != expected:
            raise ValueError(f"noise must have shape {expected}, got {tuple(noise.shape)}")

        flat_noise = noise.reshape(batch * samples, self.in_channels, height, width)
        noise_low = F.interpolate(
            flat_noise, size=fused.shape[-2:], mode="bilinear", align_corners=False
        )
        noise_features = self.noise_projection(noise_low)
        x = fused[:, None].expand(-1, samples, -1, -1, -1).reshape_as(noise_features)
        x = x + noise_features

        expanded_skips = [
            feature[:, None]
            .expand(-1, samples, -1, -1, -1)
            .reshape(batch * samples, *feature.shape[1:])
            for feature in current_skips
        ]
        for stage, noise_injection, skip in zip(
            self.decoder, self.decoder_noise, reversed(expanded_skips), strict=True
        ):
            x = stage(x, skip)
            noise_at_scale = F.interpolate(
                flat_noise, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
            x = x + noise_injection(noise_at_scale)

        residual_logits, gate_logits = self.output(x).chunk(2, dim=1)
        residual = self.max_residual * torch.tanh(residual_logits)
        gate = torch.sigmoid(gate_logits)
        flat_current = (
            current[:, None]
            .expand(-1, samples, -1, -1, -1)
            .reshape(batch * samples, *current.shape[1:])
        )
        next_state = (flat_current + gate * residual).clamp(-1.0, 1.0)
        return next_state.reshape(batch, samples, *current.shape[1:])

    def _validate_images(self, current: Tensor, goal: Tensor) -> None:
        if current.ndim != 4 or current.shape != goal.shape:
            raise ValueError("current and goal must have identical [B,C,H,W] shapes")
        if current.shape[1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} image channels")
        if current.shape[-2] % 8 or current.shape[-1] % 8:
            raise ValueError("image height and width must be divisible by 8")
