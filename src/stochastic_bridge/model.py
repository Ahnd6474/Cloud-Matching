from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


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


class ImplicitCoordinateDecoder(nn.Module):
    """Decode every output coordinate with a shared MLP, without learned upsampling.

    The fused low-resolution condition is sampled continuously at the output
    grid.  Raw current pixels, full-resolution encoder detail, Fourier
    coordinates, and the spatial latent are then mapped to residual/gate logits
    by the same pointwise MLP.  There is no transposed convolution or patch
    unprojection that can assign different kernels to fixed pixel phases.
    """

    def __init__(
        self,
        condition_channels: int,
        detail_channels: int,
        image_channels: int,
        hidden_dim: int,
        depth: int,
        fourier_bands: int,
        chunk_size: int,
    ) -> None:
        super().__init__()
        if hidden_dim < 1 or depth < 1:
            raise ValueError("implicit hidden_dim and depth must be positive")
        if fourier_bands < 0:
            raise ValueError("implicit_fourier_bands cannot be negative")
        if chunk_size < 1:
            raise ValueError("implicit_chunk_size must be positive")
        self.image_channels = image_channels
        self.fourier_bands = fourier_bands
        self.chunk_size = chunk_size
        coordinate_channels = 2 + 4 * fourier_bands
        input_dim = (
            condition_channels
            + detail_channels
            + image_channels  # current RGB
            + image_channels  # spatial Gaussian latent
            + coordinate_channels
        )
        layers: list[nn.Module] = [
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
        ]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        output = nn.Linear(hidden_dim, 2 * image_channels)
        nn.init.normal_(output.weight, std=1e-3)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        fused: Tensor,
        detail: Tensor,
        current: Tensor,
        flat_noise: Tensor,
        samples: int,
    ) -> tuple[Tensor, Tensor]:
        batch, _, height, width = current.shape
        fused_at_pixels = F.interpolate(
            fused, size=(height, width), mode="bilinear", align_corners=False
        )
        condition = torch.cat([fused_at_pixels, detail, current], dim=1)
        condition = (
            condition[:, None]
            .expand(-1, samples, -1, -1, -1)
            .reshape(batch * samples, condition.shape[1], height, width)
        )
        coordinates = _fourier_coordinate_grid(
            height,
            width,
            self.fourier_bands,
            device=current.device,
            dtype=current.dtype,
        ).expand(batch * samples, -1, -1, -1)
        queries = torch.cat([condition, flat_noise, coordinates], dim=1)
        queries = queries.permute(0, 2, 3, 1).reshape(-1, queries.shape[1])
        decoded = torch.cat(
            [
                self.mlp(queries[start : start + self.chunk_size])
                for start in range(0, queries.shape[0], self.chunk_size)
            ],
            dim=0,
        )
        decoded = decoded.reshape(
            batch * samples, height, width, 2 * self.image_channels
        ).permute(0, 3, 1, 2)
        return decoded.chunk(2, dim=1)


def _fourier_coordinate_grid(
    height: int,
    width: int,
    bands: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=torch.float32)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    coordinates = torch.stack([xx, yy], dim=-1)
    features = [coordinates]
    if bands:
        frequencies = torch.pi * 2.0 ** torch.arange(
            bands, device=device, dtype=torch.float32
        )
        angles = coordinates[..., None] * frequencies
        features.extend([angles.sin().flatten(-2), angles.cos().flatten(-2)])
    encoded = torch.cat(features, dim=-1)
    return encoded.permute(2, 0, 1)[None].to(dtype=dtype)


def _partition_windows(
    tokens: Tensor,
    window_size: int,
    shift: int,
) -> tuple[Tensor, Tensor, tuple[int, int, int, int, int]]:
    """Partition a channels-last image without cyclic boundary wrapping."""
    batch, height, width, dim = tokens.shape
    top = shift
    left = shift
    padded_height = math.ceil((height + top) / window_size) * window_size
    padded_width = math.ceil((width + left) / window_size) * window_size
    bottom = padded_height - height - top
    right = padded_width - width - left
    channels_first = tokens.permute(0, 3, 1, 2)
    padded = F.pad(channels_first, (left, right, top, bottom))
    padded = padded.permute(0, 2, 3, 1)
    windows = (
        padded.reshape(
            batch,
            padded_height // window_size,
            window_size,
            padded_width // window_size,
            window_size,
            dim,
        )
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(-1, window_size * window_size, dim)
    )

    valid = torch.ones(
        (batch, 1, height, width), device=tokens.device, dtype=torch.bool
    )
    valid = F.pad(valid, (left, right, top, bottom), value=False)
    valid_windows = (
        valid.reshape(
            batch,
            1,
            padded_height // window_size,
            window_size,
            padded_width // window_size,
            window_size,
        )
        .permute(0, 2, 4, 3, 5, 1)
        .reshape(-1, window_size * window_size)
    )
    metadata = (height, width, padded_height, padded_width, shift)
    return windows, ~valid_windows, metadata


def _reverse_windows(
    windows: Tensor,
    window_size: int,
    metadata: tuple[int, int, int, int, int],
    batch: int,
) -> Tensor:
    height, width, padded_height, padded_width, shift = metadata
    dim = windows.shape[-1]
    padded = (
        windows.reshape(
            batch,
            padded_height // window_size,
            padded_width // window_size,
            window_size,
            window_size,
            dim,
        )
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(batch, padded_height, padded_width, dim)
    )
    return padded[:, shift : shift + height, shift : shift + width]


class FactorizedAttention2d(nn.Module):
    """Local-window or axial attention over a full-resolution token grid."""

    def __init__(
        self,
        dim: int,
        heads: int,
        kind: str,
        window_size: int = 8,
        shift: bool = False,
    ) -> None:
        super().__init__()
        if kind not in {"local", "row", "column"}:
            raise ValueError("attention kind must be local, row, or column")
        if window_size < 1:
            raise ValueError("fullres_window_size must be positive")
        self.kind = kind
        self.window_size = window_size
        self.shift = window_size // 2 if shift and window_size > 1 else 0
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, query: Tensor, context: Tensor | None = None) -> Tensor:
        context = query if context is None else context
        if query.shape != context.shape or query.ndim != 4:
            raise ValueError("attention tensors must share [B,H,W,D] shape")
        batch, height, width, dim = query.shape
        if self.kind == "row":
            q = query.reshape(batch * height, width, dim)
            kv = context.reshape(batch * height, width, dim)
            result, _ = self.attention(q, kv, kv, need_weights=False)
            return result.reshape(batch, height, width, dim)
        if self.kind == "column":
            q = query.permute(0, 2, 1, 3).reshape(batch * width, height, dim)
            kv = context.permute(0, 2, 1, 3).reshape(batch * width, height, dim)
            result, _ = self.attention(q, kv, kv, need_weights=False)
            return result.reshape(batch, width, height, dim).permute(0, 2, 1, 3)

        q_windows, padding_mask, metadata = _partition_windows(
            query, self.window_size, self.shift
        )
        kv_windows, _, context_metadata = _partition_windows(
            context, self.window_size, self.shift
        )
        if metadata != context_metadata:
            raise RuntimeError("query and context window layouts differ")
        result, _ = self.attention(
            q_windows,
            kv_windows,
            kv_windows,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        return _reverse_windows(result, self.window_size, metadata, batch)


class FullResolutionCrossBlock(nn.Module):
    """Fuse current and dream-goal pixels with local and axial cross attention."""

    def __init__(
        self,
        dim: int,
        heads: int,
        window_size: int,
        ffn_ratio: float,
        shifted: bool,
    ) -> None:
        super().__init__()
        hidden = max(dim, round(dim * ffn_ratio))
        self.query_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(3)])
        self.context_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(3)])
        self.attentions = nn.ModuleList(
            [
                FactorizedAttention2d(
                    dim, heads, "local", window_size=window_size, shift=shifted
                ),
                FactorizedAttention2d(dim, heads, "row"),
                FactorizedAttention2d(dim, heads, "column"),
            ]
        )
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, current: Tensor, goal: Tensor) -> Tensor:
        for query_norm, context_norm, attention in zip(
            self.query_norms,
            self.context_norms,
            self.attentions,
            strict=True,
        ):
            current = current + attention(query_norm(current), context_norm(goal))
        return current + self.ff(current)


class FullResolutionMixerBlock(nn.Module):
    """One local, row-axial, or column-axial transformer block."""

    def __init__(
        self,
        dim: int,
        heads: int,
        kind: str,
        window_size: int,
        ffn_ratio: float,
        shifted: bool,
    ) -> None:
        super().__init__()
        hidden = max(dim, round(dim * ffn_ratio))
        self.norm = nn.LayerNorm(dim)
        self.attention = FactorizedAttention2d(
            dim,
            heads,
            kind,
            window_size=window_size,
            shift=shifted,
        )
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        tokens = tokens + self.attention(self.norm(tokens))
        return tokens + self.ff(tokens)


class FullResolutionAxialCore(nn.Module):
    """Pixel-token bridge with dense cross fusion and factorized self attention."""

    def __init__(
        self,
        in_channels: int,
        dim: int,
        heads: int,
        depth: int,
        cross_depth: int,
        ffn_ratio: float,
        window_size: int,
        max_residual: float,
        noise_energy_min: float,
        noise_energy_max: float,
        noise_energy_init: float,
        gradient_checkpointing: bool,
    ) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("fullres_dim must be divisible by heads")
        if dim % 4:
            raise ValueError("fullres_dim must be divisible by 4")
        if depth < 1 or cross_depth < 1:
            raise ValueError("full-resolution depths must be positive")
        if ffn_ratio <= 0.0:
            raise ValueError("fullres_ffn_ratio must be positive")
        self.in_channels = in_channels
        self.dim = dim
        self.max_residual = max_residual
        self.noise_energy_min = noise_energy_min
        self.noise_energy_max = noise_energy_max
        self.gradient_checkpointing = gradient_checkpointing
        self.pixel_embed = nn.Linear(in_channels, dim)
        self.cross_blocks = nn.ModuleList(
            [
                FullResolutionCrossBlock(
                    dim,
                    heads,
                    window_size,
                    ffn_ratio,
                    shifted=bool(index % 2),
                )
                for index in range(cross_depth)
            ]
        )
        pattern = ("local", "row", "local", "column")
        self.blocks = nn.ModuleList(
            [
                FullResolutionMixerBlock(
                    dim,
                    heads,
                    pattern[index % len(pattern)],
                    window_size,
                    ffn_ratio,
                    shifted=(pattern[index % len(pattern)] == "local" and index % 4 == 2),
                )
                for index in range(depth)
            ]
        )
        self.energy_norm = nn.LayerNorm(dim)
        self.energy_head = nn.Linear(dim, 1)
        fraction = (noise_energy_init - noise_energy_min) / (
            noise_energy_max - noise_energy_min
        )
        energy_bias = math.log(fraction / (1.0 - fraction))
        nn.init.zeros_(self.energy_head.weight)
        nn.init.constant_(self.energy_head.bias, energy_bias)
        projection = torch.randn(dim, in_channels)
        projection = projection / projection.norm(dim=1, keepdim=True).clamp_min(1e-8)
        self.register_buffer("rgb_noise_projection", projection)
        self.output_norm = nn.LayerNorm(dim)
        self.output_head = nn.Linear(dim, in_channels)
        nn.init.normal_(self.output_head.weight, std=1e-3)
        nn.init.zeros_(self.output_head.bias)

    def encode_condition(self, current: Tensor, goal: Tensor) -> Tensor:
        batch, _, height, width = current.shape
        current_tokens = self.pixel_embed(current.permute(0, 2, 3, 1))
        goal_tokens = self.pixel_embed(goal.permute(0, 2, 3, 1))
        position = _sinusoidal_2d_position(
            height, width, self.dim, current.device, current.dtype
        ).reshape(1, height, width, self.dim)
        current_tokens = current_tokens + position
        goal_tokens = goal_tokens + position
        for block in self.cross_blocks:
            if self.gradient_checkpointing and self.training:
                current_tokens = checkpoint(
                    block, current_tokens, goal_tokens, use_reentrant=False
                )
            else:
                current_tokens = block(current_tokens, goal_tokens)
        if current_tokens.shape != (batch, height, width, self.dim):
            raise RuntimeError("full-resolution condition changed spatial shape")
        return current_tokens

    def spatial_energy(self, condition: Tensor) -> Tensor:
        unit_energy = torch.sigmoid(self.energy_head(self.energy_norm(condition)))
        energy = self.noise_energy_min + (
            self.noise_energy_max - self.noise_energy_min
        ) * unit_energy
        return energy.squeeze(-1)

    def _latent_noise(
        self,
        condition: Tensor,
        samples: int,
        noise: Tensor | None,
    ) -> Tensor:
        batch, height, width, dim = condition.shape
        if noise is None:
            return torch.randn(
                batch,
                samples,
                height,
                width,
                dim,
                device=condition.device,
                dtype=condition.dtype,
            )
        if noise.ndim != 5 or noise.shape[:2] != (batch, samples):
            raise ValueError("noise must have shape [B,samples,C,H,W]")
        if noise.shape[-2:] != (height, width):
            raise ValueError("noise spatial shape must match the input images")
        channels = noise.shape[2]
        channels_last = noise.permute(0, 1, 3, 4, 2)
        if channels == dim:
            return channels_last
        if channels == self.in_channels:
            return F.linear(channels_last, self.rgb_noise_projection)
        raise ValueError(
            f"noise channels must be {self.in_channels} or {dim}, got {channels}"
        )

    def decode(
        self,
        condition: Tensor,
        current: Tensor,
        samples: int,
        noise: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        batch, height, width, dim = condition.shape
        energy = self.spatial_energy(condition)
        latent = self._latent_noise(condition, samples, noise)
        # ``energy`` is total expected feature-vector energy at a pixel.
        latent = latent * (energy / dim).sqrt()[:, None, :, :, None]
        tokens = condition[:, None].expand(-1, samples, -1, -1, -1) + latent
        tokens = tokens.reshape(batch * samples, height, width, dim)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                tokens = checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)
        residual = self.max_residual * torch.tanh(
            self.output_head(self.output_norm(tokens))
        )
        residual = residual.permute(0, 3, 1, 2)
        flat_current = (
            current[:, None]
            .expand(-1, samples, -1, -1, -1)
            .reshape(batch * samples, *current.shape[1:])
        )
        cloud = (flat_current + residual).clamp(-1.0, 1.0)
        return cloud.reshape(batch, samples, *current.shape[1:]), energy


class StochasticImageBridge(nn.Module):
    """Shared encoder, cross-attention, spatial noise, and residual decoder."""

    def __init__(
        self,
        architecture: str = "pyramid",
        in_channels: int = 3,
        base_channels: int = 32,
        heads: int = 8,
        attention_depth: int = 2,
        max_residual: float = 1.0,
        noise_variance_min: float = 1e-4,
        noise_variance_max: float = 1.0,
        noise_variance_init: float = 0.1,
        encoder_type: str = "cnn",
        vit_depth: int = 4,
        vit_patch_size: int = 8,
        decoder_type: str = "conv",
        implicit_hidden_dim: int = 128,
        implicit_depth: int = 3,
        implicit_fourier_bands: int = 6,
        implicit_chunk_size: int = 65_536,
        fullres_dim: int = 320,
        fullres_depth: int = 12,
        fullres_cross_depth: int = 2,
        fullres_ffn_ratio: float = 2.0,
        fullres_window_size: int = 8,
        fullres_gradient_checkpointing: bool = True,
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

        normalized_architecture = architecture.strip().lower()
        if normalized_architecture not in {"pyramid", "fullres_axial"}:
            raise ValueError("architecture must be 'pyramid' or 'fullres_axial'")
        bottleneck_channels = base_channels * 8
        if normalized_architecture == "pyramid" and bottleneck_channels % heads:
            raise ValueError("base_channels * 8 must be divisible by heads")
        if normalized_architecture == "pyramid" and bottleneck_channels % 4:
            raise ValueError("base_channels * 8 must be divisible by 4")
        if not 0.0 <= noise_variance_min < noise_variance_max:
            raise ValueError("noise_variance_min must be non-negative and below max")
        if not noise_variance_min < noise_variance_init < noise_variance_max:
            raise ValueError("noise_variance_init must lie strictly between min and max")

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.max_residual = max_residual
        self.noise_variance_min = noise_variance_min
        self.noise_variance_max = noise_variance_max
        self.architecture = normalized_architecture
        if self.architecture == "fullres_axial":
            self.fullres = FullResolutionAxialCore(
                in_channels=in_channels,
                dim=fullres_dim,
                heads=heads,
                depth=fullres_depth,
                cross_depth=fullres_cross_depth,
                ffn_ratio=fullres_ffn_ratio,
                window_size=fullres_window_size,
                max_residual=max_residual,
                noise_energy_min=noise_variance_min,
                noise_energy_max=noise_variance_max,
                noise_energy_init=noise_variance_init,
                gradient_checkpointing=fullres_gradient_checkpointing,
            )
            self.encoder_type = "fullres_axial"
            self.decoder_type = "linear_pixel"
            return
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
        variance_hidden = max(bottleneck_channels // 4, 8)
        self.noise_variance_head = nn.Sequential(
            nn.LayerNorm(bottleneck_channels),
            nn.Linear(bottleneck_channels, variance_hidden),
            nn.SiLU(),
            nn.Linear(variance_hidden, 1),
        )
        variance_fraction = (noise_variance_init - noise_variance_min) / (
            noise_variance_max - noise_variance_min
        )
        variance_bias = math.log(variance_fraction / (1.0 - variance_fraction))
        final_variance_layer = self.noise_variance_head[-1]
        assert isinstance(final_variance_layer, nn.Linear)
        nn.init.zeros_(final_variance_layer.weight)
        nn.init.constant_(final_variance_layer.bias, variance_bias)
        channels = self.encoder.channels  # type: ignore[attr-defined]
        normalized_decoder = decoder_type.strip().lower()
        if normalized_decoder == "conv":
            self.noise_projection = nn.Sequential(
                nn.Conv2d(in_channels, bottleneck_channels, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(bottleneck_channels, bottleneck_channels, 1),
            )
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
        elif normalized_decoder == "implicit":
            self.implicit_decoder = ImplicitCoordinateDecoder(
                condition_channels=channels[3],
                detail_channels=channels[0],
                image_channels=in_channels,
                hidden_dim=implicit_hidden_dim,
                depth=implicit_depth,
                fourier_bands=implicit_fourier_bands,
                chunk_size=implicit_chunk_size,
            )
        else:
            raise ValueError("decoder_type must be 'conv' or 'implicit'")
        self.decoder_type = normalized_decoder

    def encode_condition(self, current: Tensor, goal: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Compute deterministic context once before drawing cloud samples."""
        self._validate_images(current, goal)
        if self.architecture == "fullres_axial":
            condition = self.fullres.encode_condition(current, goal)
            return condition.permute(0, 3, 1, 2), []
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

    def noise_variance_from_condition(self, fused: Tensor) -> Tensor:
        """Predict one conditional latent variance per input image."""
        if fused.ndim != 4:
            raise ValueError("fused condition must have shape [B,C,H,W]")
        if self.architecture == "fullres_axial":
            condition = fused.permute(0, 2, 3, 1)
            return self.fullres.spatial_energy(condition).mean(
                dim=(-2, -1), keepdim=False
            )[:, None]
        pooled = fused.mean(dim=(-2, -1))
        unit_variance = torch.sigmoid(self.noise_variance_head(pooled))
        return self.noise_variance_min + (
            self.noise_variance_max - self.noise_variance_min
        ) * unit_variance

    def predict_noise_variance(self, current: Tensor, goal: Tensor) -> Tensor:
        """Encode a condition and return its learned scalar latent variance [B]."""
        fused, _ = self.encode_condition(current, goal)
        return self.noise_variance_from_condition(fused).squeeze(-1)

    def noise_energy_from_condition(self, fused: Tensor) -> Tensor:
        """Return the learned full-resolution noise-energy map [B,H,W]."""
        if self.architecture != "fullres_axial":
            raise RuntimeError("spatial noise energy is available only for fullres_axial")
        if fused.ndim != 4:
            raise ValueError("fused condition must have shape [B,C,H,W]")
        return self.fullres.spatial_energy(fused.permute(0, 2, 3, 1))

    def predict_noise_energy(self, current: Tensor, goal: Tensor) -> Tensor:
        """Encode current/goal and return spatial noise energy [B,H,W]."""
        fused, _ = self.encode_condition(current, goal)
        return self.noise_energy_from_condition(fused)

    def forward(
        self,
        current: Tensor,
        goal: Tensor,
        samples: int = 4,
        noise: Tensor | None = None,
        return_noise_variance: bool = False,
        return_noise_energy: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        fused, current_skips = self.encode_condition(current, goal)
        if samples < 1:
            raise ValueError("samples must be positive")
        if self.architecture == "fullres_axial":
            condition = fused.permute(0, 2, 3, 1)
            cloud, spatial_energy = self.fullres.decode(
                condition=condition,
                current=current,
                samples=samples,
                noise=noise,
            )
            noise_variance = spatial_energy.mean(dim=(-2, -1))
            if return_noise_energy:
                return cloud, noise_variance, spatial_energy
            if return_noise_variance:
                return cloud, noise_variance
            return cloud
        noise_variance = self.noise_variance_from_condition(fused)
        batch, _, height, width = current.shape
        if return_noise_energy:
            raise RuntimeError("spatial noise energy is available only for fullres_axial")
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

        # lambda is a learned variance, so its square root scales the standard
        # Gaussian supplied by the caller. Cloud matching is its only target.
        noise_scale = noise_variance.sqrt().reshape(batch, 1, 1, 1, 1)
        scaled_noise = noise * noise_scale
        flat_noise = scaled_noise.reshape(
            batch * samples, self.in_channels, height, width
        )
        if self.decoder_type == "conv":
            noise_low = F.interpolate(
                flat_noise, size=fused.shape[-2:], mode="bilinear", align_corners=False
            )
            noise_features = self.noise_projection(noise_low)
            x = fused[:, None].expand(-1, samples, -1, -1, -1).reshape_as(
                noise_features
            )
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
        else:
            residual_logits, gate_logits = self.implicit_decoder(
                fused=fused,
                detail=current_skips[0],
                current=current,
                flat_noise=flat_noise,
                samples=samples,
            )
        residual = self.max_residual * torch.tanh(residual_logits)
        gate = torch.sigmoid(gate_logits)
        flat_current = (
            current[:, None]
            .expand(-1, samples, -1, -1, -1)
            .reshape(batch * samples, *current.shape[1:])
        )
        next_state = (flat_current + gate * residual).clamp(-1.0, 1.0)
        cloud = next_state.reshape(batch, samples, *current.shape[1:])
        if return_noise_variance:
            return cloud, noise_variance.squeeze(-1)
        return cloud

    def _validate_images(self, current: Tensor, goal: Tensor) -> None:
        if current.ndim != 4 or current.shape != goal.shape:
            raise ValueError("current and goal must have identical [B,C,H,W] shapes")
        if current.shape[1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} image channels")
        if self.architecture == "pyramid" and (
            current.shape[-2] % 8 or current.shape[-1] % 8
        ):
            raise ValueError("image height and width must be divisible by 8")
