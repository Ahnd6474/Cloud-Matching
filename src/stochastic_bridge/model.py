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
) -> tuple[Tensor, Tensor | None, tuple[int, int, int, int, int]]:
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

    padding_mask = None
    if any((left, right, top, bottom)):
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
        padding_mask = ~valid_windows
    metadata = (height, width, padded_height, padded_width, shift)
    return windows, padding_mask, metadata


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
        self_attention = context is None
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
        if self_attention:
            kv_windows, context_metadata = q_windows, metadata
        else:
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


class MultiscaleCvTAttention2d(nn.Module):
    """CvT-style attention with native-grid Q and convolutionally pooled K/V.

    The residual/query stream never leaves the input grid.  Depthwise spatial
    convolutions provide local inductive bias while reducing the context to a
    compact multiscale token memory before the learned Q/K/V projections.  This
    is intentionally a full-resolution CvT variant, not an exact reproduction
    of the stage hierarchy from the original CvT paper.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        kernel_sizes: list[int] | tuple[int, ...],
        output_sizes: list[int] | tuple[int, ...],
    ) -> None:
        super().__init__()
        kernels = tuple(int(value) for value in kernel_sizes)
        sizes = tuple(int(value) for value in output_sizes)
        if not kernels or len(kernels) != len(sizes):
            raise ValueError(
                "pooled attention requires equally sized non-empty kernel/output lists"
            )
        if any(kernel < 1 or kernel % 2 == 0 for kernel in kernels):
            raise ValueError("pooled attention kernels must be positive odd integers")
        if any(size < 1 for size in sizes):
            raise ValueError("pooled attention output sizes must be positive")

        self.dim = dim
        self.kernel_sizes = kernels
        self.output_sizes = sizes
        self.context_convolutions = nn.ModuleList(
            [
                nn.Conv2d(
                    dim,
                    dim,
                    kernel_size=kernel,
                    padding=kernel // 2,
                    groups=dim,
                    bias=False,
                )
                for kernel in kernels
            ]
        )
        # A normalized box filter is a stable multiscale starting point while
        # leaving every depthwise kernel trainable.
        for kernel, convolution in zip(
            self.kernel_sizes, self.context_convolutions, strict=True
        ):
            nn.init.constant_(convolution.weight, 1.0 / (kernel * kernel))
        self.scale_embeddings = nn.Parameter(torch.zeros(len(kernels), dim))
        nn.init.normal_(self.scale_embeddings, std=0.02)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)

    def pooled_token_count(self, height: int, width: int) -> int:
        return sum(
            min(size, height) * min(size, width) for size in self.output_sizes
        )

    def pool_context(self, context: Tensor) -> Tensor:
        if context.ndim != 4 or context.shape[-1] != self.dim:
            raise ValueError("context must have shape [B,H,W,D]")
        _, height, width, _ = context.shape
        channels_first = context.permute(0, 3, 1, 2)
        pooled_tokens: list[Tensor] = []
        for index, (convolution, output_size) in enumerate(
            zip(self.context_convolutions, self.output_sizes, strict=True)
        ):
            pooled_height = min(output_size, height)
            pooled_width = min(output_size, width)
            # Convolution and spatial reduction happen together. Running 3x3,
            # 5x5 and 7x7 kernels densely before pooling would erase the compute
            # saving that reduced K/V attention is meant to provide.
            stride_height = max(1, height // pooled_height)
            stride_width = max(1, width // pooled_width)
            features = F.conv2d(
                channels_first,
                convolution.weight,
                bias=None,
                stride=(stride_height, stride_width),
                padding=convolution.padding,
                groups=self.dim,
            )
            pooled = F.adaptive_avg_pool2d(
                features, (pooled_height, pooled_width)
            )
            tokens = pooled.flatten(2).transpose(1, 2)
            pooled_tokens.append(tokens + self.scale_embeddings[index])
        return torch.cat(pooled_tokens, dim=1)

    def forward(self, query: Tensor, context: Tensor | None = None) -> Tensor:
        context = query if context is None else context
        if query.ndim != 4 or query.shape[-1] != self.dim:
            raise ValueError("query must have shape [B,H,W,D]")
        if context.ndim != 4 or context.shape[0] != query.shape[0]:
            raise ValueError("context must have shape [B,H,W,D]")
        batch, height, width, dim = query.shape
        query_tokens = query.reshape(batch, height * width, dim)
        context_tokens = self.pool_context(context)
        result, _ = self.attention(
            query_tokens,
            context_tokens,
            context_tokens,
            need_weights=False,
        )
        return result.reshape(batch, height, width, dim)


# Backward-compatible import name for older analysis scripts.  New code should
# use the CvT name; module/parameter paths are unchanged for old checkpoints.
MultiscalePooledAttention2d = MultiscaleCvTAttention2d


class FullResolutionCrossBlock(nn.Module):
    """Fuse image conditions with CvT or legacy factorized cross attention."""

    def __init__(
        self,
        dim: int,
        heads: int,
        window_size: int,
        ffn_ratio: float,
        shifted: bool,
        gate_init: float,
        attention_type: str = "factorized",
        pooled_kernel_sizes: list[int] | tuple[int, ...] = (3, 5, 7),
        pooled_output_sizes: list[int] | tuple[int, ...] = (8, 4, 2),
    ) -> None:
        super().__init__()
        if not 0.0 < gate_init <= 1.0:
            raise ValueError("fullres_cross_gate_init must lie in (0, 1]")
        hidden = max(dim, round(dim * ffn_ratio))
        if attention_type == "factorized":
            attentions: list[nn.Module] = [
                FactorizedAttention2d(
                    dim, heads, "local", window_size=window_size, shift=shifted
                ),
                FactorizedAttention2d(dim, heads, "row"),
                FactorizedAttention2d(dim, heads, "column"),
            ]
        elif attention_type == "cvt":
            attentions = [
                MultiscaleCvTAttention2d(
                    dim,
                    heads,
                    pooled_kernel_sizes,
                    pooled_output_sizes,
                )
            ]
        else:
            raise ValueError(
                "fullres_attention_type must be 'factorized' or 'cvt'"
            )
        self.query_norms = nn.ModuleList(
            [nn.LayerNorm(dim) for _ in attentions]
        )
        self.context_norms = nn.ModuleList(
            [nn.LayerNorm(dim) for _ in attentions]
        )
        self.attentions = nn.ModuleList(attentions)
        self.gate_projections = nn.ModuleList(
            [nn.Linear(dim, 1) for _ in attentions]
        )
        gate_logit = 12.0 if gate_init == 1.0 else math.log(
            gate_init / (1.0 - gate_init)
        )
        for projection in self.gate_projections:
            nn.init.zeros_(projection.weight)
            nn.init.constant_(projection.bias, gate_logit)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, current: Tensor, goal: Tensor) -> Tensor:
        for query_norm, context_norm, attention, gate_projection in zip(
            self.query_norms,
            self.context_norms,
            self.attentions,
            self.gate_projections,
            strict=True,
        ):
            update = attention(query_norm(current), context_norm(goal))
            gate = torch.sigmoid(gate_projection(update))
            current = current + gate * update
        return current + self.ff(current)


class LearnedTextEncoder(nn.Module):
    """End-to-end caption encoder used without a pretrained CLIP dependency."""

    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        dim: int,
        heads: int,
        depth: int,
        ffn_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if vocab_size < 4 or max_length < 3:
            raise ValueError("text vocabulary/max length are too small")
        if depth < 1 or dim % heads:
            raise ValueError("text depth must be positive and dim divisible by heads")
        if ffn_ratio <= 0.0 or not 0.0 <= dropout < 1.0:
            raise ValueError("invalid text FFN ratio or dropout")
        self.max_length = max_length
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.position = nn.Parameter(torch.zeros(1, max_length, dim))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=max(dim, round(dim * ffn_ratio)),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=depth,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, token_ids: Tensor, token_mask: Tensor | None = None) -> Tensor:
        if token_ids.ndim != 2 or token_ids.shape[1] > self.max_length:
            raise ValueError("token_ids must have shape [B,L] within text_max_length")
        if token_ids.dtype != torch.long:
            raise ValueError("token_ids must use torch.long")
        if token_mask is None:
            token_mask = token_ids.ne(0)
        if token_mask.shape != token_ids.shape:
            raise ValueError("token_mask must match token_ids")
        tokens = self.embedding(token_ids) + self.position[:, : token_ids.shape[1]]
        tokens = self.transformer(tokens, src_key_padding_mask=~token_mask.bool())
        return self.output_norm(tokens)


class FullResolutionTextCrossBlock(nn.Module):
    """Full-resolution image queries attending directly to caption tokens."""

    def __init__(
        self,
        dim: int,
        heads: int,
        ffn_ratio: float,
        gate_init: float,
    ) -> None:
        super().__init__()
        if not 0.0 < gate_init <= 1.0:
            raise ValueError("fullres_cross_gate_init must lie in (0, 1]")
        hidden = max(dim, round(dim * ffn_ratio))
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.gate_projection = nn.Linear(dim, 1)
        gate_logit = 12.0 if gate_init == 1.0 else math.log(
            gate_init / (1.0 - gate_init)
        )
        nn.init.zeros_(self.gate_projection.weight)
        nn.init.constant_(self.gate_projection.bias, gate_logit)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(
        self,
        current: Tensor,
        text: Tensor,
        text_mask: Tensor | None = None,
    ) -> Tensor:
        if current.ndim != 4 or text.ndim != 3:
            raise ValueError("text cross attention expects image [B,H,W,D] and text [B,L,D]")
        batch, height, width, dim = current.shape
        query = self.query_norm(current).reshape(batch, height * width, dim)
        context = self.context_norm(text)
        key_padding_mask = None if text_mask is None else ~text_mask.bool()
        update, _ = self.attention(
            query,
            context,
            context,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        update = update.reshape(batch, height, width, dim)
        gate = torch.sigmoid(self.gate_projection(update))
        current = current + gate * update
        return current + self.ff(current)


class FullResolutionMixerBlock(nn.Module):
    """One CvT or legacy local/axial transformer block followed by an FFN."""

    def __init__(
        self,
        dim: int,
        heads: int,
        kind: str,
        window_size: int,
        ffn_ratio: float,
        shifted: bool,
        attention_type: str = "factorized",
        pooled_kernel_sizes: list[int] | tuple[int, ...] = (3, 5, 7),
        pooled_output_sizes: list[int] | tuple[int, ...] = (8, 4, 2),
    ) -> None:
        super().__init__()
        hidden = max(dim, round(dim * ffn_ratio))
        self.norm = nn.LayerNorm(dim)
        if attention_type == "factorized":
            self.attention: nn.Module = FactorizedAttention2d(
                dim,
                heads,
                kind,
                window_size=window_size,
                shift=shifted,
            )
        elif attention_type == "cvt":
            self.attention = MultiscaleCvTAttention2d(
                dim,
                heads,
                pooled_kernel_sizes,
                pooled_output_sizes,
            )
        else:
            raise ValueError(
                "fullres_attention_type must be 'factorized' or 'cvt'"
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


class FullResolutionRandomAttention(nn.Module):
    """Let full-resolution queries select from an independent random KV memory."""

    def __init__(
        self,
        dim: int,
        heads: int,
        random_dim: int,
        temperature: float,
        gate_init: float,
    ) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("random-attention dimension must be divisible by heads")
        if random_dim < 1:
            raise ValueError("random_dim must be positive")
        if temperature <= 0.0:
            raise ValueError("random-attention temperature must be positive")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.random_dim = random_dim
        self.temperature = temperature
        self.query_norm = nn.LayerNorm(dim)
        self.query_projection = nn.Linear(dim, dim, bias=False)
        # K and V deliberately use separate learned projections of the same Z.
        self.key_projection = nn.Linear(random_dim, dim, bias=False)
        self.value_projection = nn.Linear(random_dim, dim, bias=False)
        self.key_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.value_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.output_projection = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, query: Tensor, random_tokens: Tensor) -> Tensor:
        if query.ndim != 4:
            raise ValueError("query must have shape [B,H,W,C]")
        if random_tokens.ndim != 3:
            raise ValueError("random_tokens must have shape [B,M,R]")
        batch, height, width, dim = query.shape
        if dim != self.dim:
            raise ValueError(f"query feature dimension must be {self.dim}")
        if random_tokens.shape[0] != batch:
            raise ValueError("query and random memory batch sizes must match")
        if random_tokens.shape[-1] != self.random_dim:
            raise ValueError(
                f"random token dimension must be {self.random_dim}"
            )

        query_flat = self.query_projection(self.query_norm(query)).reshape(
            batch, height * width, self.heads, self.head_dim
        )
        key = self.key_norm(self.key_projection(random_tokens)).reshape(
            batch, random_tokens.shape[1], self.heads, self.head_dim
        )
        value = self.value_norm(self.value_projection(random_tokens)).reshape(
            batch, random_tokens.shape[1], self.heads, self.head_dim
        )
        query_flat = query_flat.permute(0, 2, 1, 3) / self.temperature
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        attended = F.scaled_dot_product_attention(
            query_flat,
            key,
            value,
            dropout_p=0.0,
            is_causal=False,
        )
        attended = (
            attended.permute(0, 2, 1, 3)
            .reshape(batch, height, width, dim)
        )
        return query + self.gate * self.output_projection(attended)


class FullResolutionConvEncoder(nn.Module):
    """Encode an image on its native grid with shared local convolutional context."""

    def __init__(self, in_channels: int, dim: int) -> None:
        super().__init__()
        self.input_projection = nn.Conv2d(
            in_channels,
            dim,
            kernel_size=3,
            padding=1,
        )
        self.norm = nn.LayerNorm(dim)
        self.local_refinement = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
        )
        self.local_gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, image: Tensor) -> Tensor:
        tokens = self.input_projection(image).permute(0, 2, 3, 1)
        normalized = self.norm(tokens).permute(0, 3, 1, 2)
        update = self.local_refinement(normalized).permute(0, 2, 3, 1)
        return tokens + self.local_gate * update


class FullResolutionCvTEncoder(nn.Module):
    """Shared native-grid CvT stem followed by multiscale reduced-KV attention."""

    def __init__(
        self,
        in_channels: int,
        dim: int,
        heads: int,
        pooled_kernel_sizes: list[int] | tuple[int, ...],
        pooled_output_sizes: list[int] | tuple[int, ...],
    ) -> None:
        super().__init__()
        self.stem = FullResolutionConvEncoder(in_channels, dim)
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = MultiscaleCvTAttention2d(
            dim,
            heads,
            pooled_kernel_sizes,
            pooled_output_sizes,
        )
        self.attention_gate = nn.Parameter(torch.tensor(0.1))

    @property
    def input_projection(self) -> nn.Conv2d:
        return self.stem.input_projection

    def forward(self, image: Tensor) -> Tensor:
        tokens = self.stem(image)
        update = self.attention(self.attention_norm(tokens))
        return tokens + self.attention_gate * update


# Compatibility for code that imported the pre-CvT class name directly.
FullResolutionConvolutionalAttentionEncoder = FullResolutionCvTEncoder


class FullResolutionAxialCore(nn.Module):
    """Pixel-token bridge with CvT or legacy factorized spatial attention."""

    def __init__(
        self,
        in_channels: int,
        dim: int,
        heads: int,
        depth: int,
        cross_depth: int,
        cross_gate_init: float,
        ffn_ratio: float,
        window_size: int,
        max_residual: float,
        noise_energy_min: float,
        noise_energy_max: float,
        noise_energy_init: float,
        noise_energy_parameterization: str,
        noise_amplitude_safety_max: float,
        gradient_checkpointing: bool,
        encoder_type: str,
        attention_type: str,
        pooled_kernel_sizes: list[int] | tuple[int, ...],
        pooled_output_sizes: list[int] | tuple[int, ...],
        random_attention: bool,
        random_slots: int,
        random_dim: int,
        random_temperature: float,
        random_gate1_init: float,
        random_positions: list[int] | tuple[int, ...],
        goal_condition: str,
        text_vocab_size: int,
        text_max_length: int,
        text_depth: int,
        text_heads: int,
        text_ffn_ratio: float,
        text_dropout: float,
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
        if random_attention and depth < 1:
            raise ValueError("fullres random attention requires at least one refinement block")
        if random_slots < 1:
            raise ValueError("fullres_random_slots must be positive")
        if random_dim < 1:
            raise ValueError("fullres_random_dim must be positive")
        if random_temperature <= 0.0:
            raise ValueError("fullres_random_temperature must be positive")
        self.in_channels = in_channels
        self.dim = dim
        self.max_residual = max_residual
        self.noise_energy_min = noise_energy_min
        self.noise_energy_max = noise_energy_max
        self.noise_energy_parameterization = noise_energy_parameterization
        self.noise_amplitude_safety_max = noise_amplitude_safety_max
        self.gradient_checkpointing = gradient_checkpointing
        normalized_attention = attention_type.strip().lower()
        if normalized_attention in {
            "cvt",
            "pooled",
            "multiscale_pooled",
            "pooled_multiscale",
        }:
            normalized_attention = "cvt"
        if normalized_attention not in {"factorized", "cvt"}:
            raise ValueError(
                "fullres_attention_type must be 'factorized' or 'cvt'"
            )
        self.attention_type = normalized_attention
        self.random_attention_enabled = random_attention
        self.random_slots = random_slots
        self.random_dim = random_dim
        if random_positions:
            if len(random_positions) not in {1, 2}:
                raise ValueError("fullres_random_positions must contain one index")
            # Older checkpoints stored the now-removed second injection point.
            positions = (int(random_positions[0]),)
        else:
            positions = (0,)
        if positions[0] < 0 or positions[0] >= depth:
            raise ValueError(
                "fullres_random_positions must be within [0, fullres_depth)"
            )
        self.random_positions = positions
        normalized_goal_condition = goal_condition.strip().lower()
        if normalized_goal_condition not in {"image", "text"}:
            raise ValueError("fullres_goal_condition must be 'image' or 'text'")
        self.goal_condition = normalized_goal_condition
        normalized_encoder = encoder_type.strip().lower()
        if normalized_encoder == "linear":
            self.pixel_embed: nn.Module = nn.Linear(in_channels, dim)
        elif normalized_encoder == "cnn":
            self.image_encoder: nn.Module = FullResolutionConvEncoder(in_channels, dim)
        elif normalized_encoder in {"cvt", "conv_attention", "pooled_attention"}:
            self.image_encoder = FullResolutionCvTEncoder(
                in_channels,
                dim,
                heads,
                pooled_kernel_sizes,
                pooled_output_sizes,
            )
            normalized_encoder = "cvt"
        else:
            raise ValueError(
                "fullres_encoder_type must be 'linear', 'cnn', or 'cvt'"
            )
        self.encoder_type = normalized_encoder
        if self.goal_condition == "text":
            self.text_encoder = LearnedTextEncoder(
                text_vocab_size,
                text_max_length,
                dim,
                text_heads,
                text_depth,
                text_ffn_ratio,
                text_dropout,
            )
            self.cross_blocks = nn.ModuleList(
                [
                    FullResolutionTextCrossBlock(
                        dim,
                        heads,
                        ffn_ratio,
                        cross_gate_init,
                    )
                    for _ in range(cross_depth)
                ]
            )
        else:
            self.cross_blocks = nn.ModuleList(
                [
                    FullResolutionCrossBlock(
                        dim,
                        heads,
                        window_size,
                        ffn_ratio,
                        shifted=bool(index % 2),
                        gate_init=cross_gate_init,
                        attention_type=self.attention_type,
                        pooled_kernel_sizes=pooled_kernel_sizes,
                        pooled_output_sizes=pooled_output_sizes,
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
                    attention_type=self.attention_type,
                    pooled_kernel_sizes=pooled_kernel_sizes,
                    pooled_output_sizes=pooled_output_sizes,
                )
                for index in range(depth)
            ]
        )
        self.random_attentions = nn.ModuleList()
        if self.random_attention_enabled:
            self.random_attentions.append(
                FullResolutionRandomAttention(
                    dim,
                    heads,
                    random_dim,
                    random_temperature,
                    random_gate1_init,
                )
            )
        self.energy_norm = nn.LayerNorm(dim)
        self.energy_head = nn.Linear(dim, 1)
        self.reset_energy_head(noise_energy_init)
        projection = torch.randn(dim, in_channels)
        projection = projection / projection.norm(dim=1, keepdim=True).clamp_min(1e-8)
        self.register_buffer("rgb_noise_projection", projection)
        self.output_norm = nn.LayerNorm(dim)
        self.output_head = nn.Linear(dim, in_channels)
        nn.init.normal_(self.output_head.weight, std=1e-3)
        nn.init.zeros_(self.output_head.bias)

    def reset_energy_head(self, initial_energy: float) -> None:
        """Reset only the sampler scale when changing its parameterization."""
        if initial_energy <= self.noise_energy_min:
            raise ValueError("initial energy must be strictly above its minimum")
        if self.noise_energy_parameterization == "fixed_unit":
            # This head is deliberately disconnected in fixed-unit mode.
            nn.init.zeros_(self.energy_head.weight)
            nn.init.zeros_(self.energy_head.bias)
            return
        if self.noise_energy_parameterization == "bounded":
            if initial_energy >= self.noise_energy_max:
                raise ValueError("bounded initial energy must be below its maximum")
            fraction = (initial_energy - self.noise_energy_min) / (
                self.noise_energy_max - self.noise_energy_min
            )
            energy_bias = math.log(fraction / (1.0 - fraction))
        else:
            initial_amplitude = math.sqrt(initial_energy - self.noise_energy_min)
            if initial_amplitude >= self.noise_amplitude_safety_max:
                raise ValueError("initial amplitude must be below its safety limit")
            # Inverse softplus makes the initial total energy exact while
            # leaving the learned operating scale without a sigmoid ceiling.
            energy_bias = math.log(math.expm1(initial_amplitude))
        nn.init.zeros_(self.energy_head.weight)
        nn.init.constant_(self.energy_head.bias, energy_bias)

    def encode_condition(
        self,
        current: Tensor,
        goal: Tensor,
        goal_mask: Tensor | None = None,
    ) -> Tensor:
        batch, _, height, width = current.shape
        if self.encoder_type == "linear":
            current_tokens = self.pixel_embed(current.permute(0, 2, 3, 1))
            if self.goal_condition == "image":
                goal_tokens = self.pixel_embed(goal.permute(0, 2, 3, 1))
        else:
            if self.goal_condition == "image":
                # One shared batched call reduces kernel-launch overhead and keeps
                # current/goal normalization behavior exactly symmetric.
                combined = self.image_encoder(torch.cat([current, goal], dim=0))
                current_tokens = combined[:batch]
                goal_tokens = combined[batch:]
            else:
                current_tokens = self.image_encoder(current)
        position = _sinusoidal_2d_position(
            height, width, self.dim, current.device, current.dtype
        ).reshape(1, height, width, self.dim)
        current_tokens = current_tokens + position
        if self.goal_condition == "text":
            goal_tokens = self.text_encoder(goal, goal_mask).to(current_tokens.dtype)
            for block in self.cross_blocks:
                if self.gradient_checkpointing and self.training:
                    current_tokens = checkpoint(
                        block,
                        current_tokens,
                        goal_tokens,
                        goal_mask,
                        use_reentrant=False,
                    )
                else:
                    current_tokens = block(current_tokens, goal_tokens, goal_mask)
        else:
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
        if self.noise_energy_parameterization == "fixed_unit":
            # `_latent_noise` has unit variance in each feature coordinate.
            # decode divides it by sqrt(dim), so E||e||^2 = 1 per pixel.
            # Actual diversity is calibrated in output space by the loss.
            return condition.new_ones(condition.shape[:-1])
        raw = self.energy_head(self.energy_norm(condition))
        if self.noise_energy_parameterization == "bounded":
            unit_energy = torch.sigmoid(raw)
            energy = self.noise_energy_min + (
                self.noise_energy_max - self.noise_energy_min
            ) * unit_energy
        else:
            # Learn noise amplitude directly.  The clamp is only a distant
            # numerical guard for AMP; it is not the trainable operating range.
            amplitude = F.softplus(raw.float()).clamp_max(
                self.noise_amplitude_safety_max
            )
            energy = (self.noise_energy_min + amplitude.square()).to(raw.dtype)
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

    def _random_memories(
        self,
        condition: Tensor,
        samples: int,
        noise: Tensor | None,
    ) -> Tensor:
        """Create one Z memory, deterministically from supplied noise."""
        batch = condition.shape[0]
        needed = self.random_slots * self.random_dim
        if noise is None:
            memory = torch.randn(
                batch,
                samples,
                self.random_slots,
                self.random_dim,
                device=condition.device,
                dtype=condition.dtype,
            )
        else:
            if noise.ndim != 5 or noise.shape[:2] != (batch, samples):
                raise ValueError("noise must have shape [B,samples,C,H,W]")
            flat = noise.flatten(2)
            if flat.shape[-1] < needed:
                repeats = math.ceil(needed / flat.shape[-1])
                flat = flat.repeat(1, 1, repeats)
            memory = flat[..., :needed].reshape(
                batch,
                samples,
                self.random_slots,
                self.random_dim,
            )
        return memory.reshape(
            batch * samples,
            self.random_slots,
            self.random_dim,
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
        expanded_condition = condition[:, None].expand(
            -1, samples, -1, -1, -1
        )
        if self.random_attention_enabled:
            random_memory = self._random_memories(condition, samples, noise)
            tokens = expanded_condition.reshape(
                batch * samples, height, width, dim
            )
            random_memories = (random_memory,)
        else:
            latent = self._latent_noise(condition, samples, noise)
            # ``energy`` is total expected feature-vector energy at a pixel.
            latent = latent * (energy / dim).sqrt()[:, None, :, :, None]
            tokens = (expanded_condition + latent).reshape(
                batch * samples, height, width, dim
            )
            random_memories = ()

        random_index = 0
        for index in range(len(self.blocks) + 1):
            while (
                self.random_attention_enabled
                and random_index < len(self.random_positions)
                and self.random_positions[random_index] == index
            ):
                random_layer = self.random_attentions[random_index]
                random_memory = random_memories[random_index]
                if self.gradient_checkpointing and self.training:
                    tokens = checkpoint(
                        random_layer,
                        tokens,
                        random_memory,
                        use_reentrant=False,
                    )
                else:
                    tokens = random_layer(tokens, random_memory)
                random_index += 1
            if index == len(self.blocks):
                break
            block = self.blocks[index]
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
        noise_energy_parameterization: str = "bounded",
        noise_amplitude_safety_max: float = 8.0,
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
        fullres_cross_gate_init: float = 1.0,
        fullres_ffn_ratio: float = 2.0,
        fullres_window_size: int = 8,
        fullres_gradient_checkpointing: bool = True,
        fullres_encoder_type: str = "linear",
        fullres_attention_type: str = "factorized",
        fullres_pooled_kernel_sizes: list[int] | tuple[int, ...] = (3, 5, 7),
        fullres_pooled_output_sizes: list[int] | tuple[int, ...] = (8, 4, 2),
        fullres_random_attention: bool = False,
        fullres_random_slots: int = 64,
        fullres_random_dim: int = 64,
        fullres_random_temperature: float = 1.0,
        fullres_random_gate1_init: float = 0.02,
        fullres_random_gate2_init: float | None = None,
        fullres_random_positions: list[int] | tuple[int, ...] = (),
        fullres_goal_condition: str = "image",
        text_vocab_size: int = 16384,
        text_max_length: int = 48,
        text_depth: int = 4,
        text_heads: int = 8,
        text_ffn_ratio: float = 4.0,
        text_dropout: float = 0.1,
        image_size: int | None = None,
        **legacy: object,
    ) -> None:
        super().__init__()
        del image_size
        del fullres_random_gate2_init  # Accepted only for old checkpoint configs.
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
        normalized_energy_parameterization = (
            noise_energy_parameterization.strip().lower()
        )
        if normalized_energy_parameterization not in {
            "bounded",
            "fixed_unit",
            "softplus_amplitude",
        }:
            raise ValueError(
                "noise_energy_parameterization must be 'bounded', "
                "'fixed_unit', or 'softplus_amplitude'"
            )
        if (
            normalized_architecture != "fullres_axial"
            and normalized_energy_parameterization != "bounded"
        ):
            raise ValueError(
                "non-bounded noise energy modes are available only for fullres_axial"
            )
        if not 0.0 <= noise_variance_min < noise_variance_max:
            raise ValueError("noise_variance_min must be non-negative and below max")
        if noise_variance_init <= noise_variance_min:
            raise ValueError("noise_variance_init must be strictly above min")
        if (
            normalized_energy_parameterization == "bounded"
            and noise_variance_init >= noise_variance_max
        ):
            raise ValueError("bounded noise_variance_init must be strictly below max")
        if noise_amplitude_safety_max <= 0.0:
            raise ValueError("noise_amplitude_safety_max must be positive")
        if (
            normalized_energy_parameterization == "softplus_amplitude"
            and noise_variance_init - noise_variance_min
            >= noise_amplitude_safety_max**2
        ):
            raise ValueError(
                "initial noise energy must be below the amplitude safety limit"
            )

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.max_residual = max_residual
        self.noise_variance_min = noise_variance_min
        self.noise_variance_max = noise_variance_max
        self.noise_energy_parameterization = normalized_energy_parameterization
        self.noise_amplitude_safety_max = noise_amplitude_safety_max
        self.architecture = normalized_architecture
        if self.architecture == "fullres_axial":
            self.fullres = FullResolutionAxialCore(
                in_channels=in_channels,
                dim=fullres_dim,
                heads=heads,
                depth=fullres_depth,
                cross_depth=fullres_cross_depth,
                cross_gate_init=fullres_cross_gate_init,
                ffn_ratio=fullres_ffn_ratio,
                window_size=fullres_window_size,
                max_residual=max_residual,
                noise_energy_min=noise_variance_min,
                noise_energy_max=noise_variance_max,
                noise_energy_init=noise_variance_init,
                noise_energy_parameterization=normalized_energy_parameterization,
                noise_amplitude_safety_max=noise_amplitude_safety_max,
                gradient_checkpointing=fullres_gradient_checkpointing,
                encoder_type=fullres_encoder_type,
                attention_type=fullres_attention_type,
                pooled_kernel_sizes=fullres_pooled_kernel_sizes,
                pooled_output_sizes=fullres_pooled_output_sizes,
                random_attention=fullres_random_attention,
                random_slots=fullres_random_slots,
                random_dim=fullres_random_dim,
                random_temperature=fullres_random_temperature,
                random_gate1_init=fullres_random_gate1_init,
                random_positions=fullres_random_positions,
                goal_condition=fullres_goal_condition,
                text_vocab_size=text_vocab_size,
                text_max_length=text_max_length,
                text_depth=text_depth,
                text_heads=text_heads,
                text_ffn_ratio=text_ffn_ratio,
                text_dropout=text_dropout,
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

    def load_state_dict(
        self,
        state_dict: dict[str, Tensor],
        strict: bool = True,
        assign: bool = False,
    ):
        """Load checkpoints across the single-random and CNN-encoder migration."""
        compatible = state_dict.copy()
        for name in tuple(compatible):
            if name.startswith("fullres.random_attentions.1."):
                compatible.pop(name)

        if (
            self.architecture == "fullres_axial"
            and self.fullres.encoder_type == "cvt"
        ):
            legacy_stem_prefix = "fullres.image_encoder."
            stem_names = (
                "input_projection.",
                "norm.",
                "local_refinement.",
                "local_gate",
            )
            for name in tuple(compatible):
                suffix = name.removeprefix(legacy_stem_prefix)
                if name.startswith(legacy_stem_prefix) and suffix.startswith(stem_names):
                    compatible[
                        f"fullres.image_encoder.stem.{suffix}"
                    ] = compatible.pop(name)

        if (
            self.architecture == "fullres_axial"
            and self.fullres.encoder_type in {"cnn", "cvt"}
            and "fullres.pixel_embed.weight" in compatible
        ):
            projection_prefix = (
                "fullres.image_encoder.stem.input_projection"
                if self.fullres.encoder_type == "cvt"
                else "fullres.image_encoder.input_projection"
            )
            if f"{projection_prefix}.weight" in compatible:
                compatible.pop("fullres.pixel_embed.weight")
                compatible.pop("fullres.pixel_embed.bias")
            else:
                linear_weight = compatible.pop("fullres.pixel_embed.weight")
                linear_bias = compatible.pop("fullres.pixel_embed.bias")
                target = self.fullres.image_encoder.input_projection.weight
                kernel = torch.zeros_like(target)
                center_y = kernel.shape[-2] // 2
                center_x = kernel.shape[-1] // 2
                kernel[:, :, center_y, center_x] = linear_weight.to(kernel)
                compatible[f"{projection_prefix}.weight"] = kernel
                compatible[f"{projection_prefix}.bias"] = linear_bias

        return super().load_state_dict(compatible, strict=strict, assign=assign)

    def encode_condition(
        self,
        current: Tensor,
        goal: Tensor,
        goal_mask: Tensor | None = None,
    ) -> tuple[Tensor, list[Tensor]]:
        """Compute deterministic context once before drawing cloud samples."""
        self._validate_condition(current, goal, goal_mask)
        if self.architecture == "fullres_axial":
            condition = self.fullres.encode_condition(current, goal, goal_mask)
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

    def predict_noise_variance(
        self,
        current: Tensor,
        goal: Tensor,
        goal_mask: Tensor | None = None,
    ) -> Tensor:
        """Encode a condition and return its learned scalar latent variance [B]."""
        fused, _ = self.encode_condition(current, goal, goal_mask)
        return self.noise_variance_from_condition(fused).squeeze(-1)

    def noise_energy_from_condition(self, fused: Tensor) -> Tensor:
        """Return the learned full-resolution noise-energy map [B,H,W]."""
        if self.architecture != "fullres_axial":
            raise RuntimeError("spatial noise energy is available only for fullres_axial")
        if fused.ndim != 4:
            raise ValueError("fused condition must have shape [B,C,H,W]")
        return self.fullres.spatial_energy(fused.permute(0, 2, 3, 1))

    def predict_noise_energy(
        self,
        current: Tensor,
        goal: Tensor,
        goal_mask: Tensor | None = None,
    ) -> Tensor:
        """Encode current/goal and return spatial noise energy [B,H,W]."""
        fused, _ = self.encode_condition(current, goal, goal_mask)
        return self.noise_energy_from_condition(fused)

    def forward(
        self,
        current: Tensor,
        goal: Tensor,
        samples: int = 4,
        noise: Tensor | None = None,
        return_noise_variance: bool = False,
        return_noise_energy: bool = False,
        goal_mask: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        fused, current_skips = self.encode_condition(current, goal, goal_mask)
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

    def _validate_condition(
        self,
        current: Tensor,
        goal: Tensor,
        goal_mask: Tensor | None,
    ) -> None:
        if current.ndim != 4:
            raise ValueError("current must have shape [B,C,H,W]")
        text_goal = (
            self.architecture == "fullres_axial"
            and self.fullres.goal_condition == "text"
        )
        if text_goal:
            if goal.ndim != 2 or goal.shape[0] != current.shape[0]:
                raise ValueError("text goal must have shape [B,L]")
            if goal.dtype != torch.long:
                raise ValueError("text goal token IDs must use torch.long")
            if goal_mask is not None and goal_mask.shape != goal.shape:
                raise ValueError("goal_mask must match text goal")
        elif current.shape != goal.shape:
            raise ValueError("current and goal must have identical [B,C,H,W] shapes")
        if current.shape[1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} image channels")
        if self.architecture == "pyramid" and (
            current.shape[-2] % 8 or current.shape[-1] % 8
        ):
            raise ValueError("image height and width must be divisible by 8")
