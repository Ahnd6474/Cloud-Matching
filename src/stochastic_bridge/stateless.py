from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor


_UINT32_MASK = (1 << 32) - 1
_UINT32_SCALE = float(1 << 32)


def _pcg_uniform(seeds: Tensor, counters: Tensor) -> Tensor:
    """Map uint32 seed/counter pairs to deterministic open-interval uniforms."""
    value = counters[None, :] ^ (seeds[:, None] & _UINT32_MASK)
    state = (value * 747_796_405 + 2_891_336_453) & _UINT32_MASK
    shift = (state >> 28) + 4
    word = (((state >> shift) ^ state) * 277_803_737) & _UINT32_MASK
    word = ((word >> 22) ^ word) & _UINT32_MASK
    return (word.float() + 0.5) / _UINT32_SCALE


def stateless_normal(
    seeds: Tensor,
    sample_shape: Sequence[int],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Generate one reproducible standard-normal tensor per uint32 seed.

    The generator is independent of global RNG state and batch ordering. A
    PCG-style 32-bit hash produces uniform pairs, followed by Box-Muller. This
    lets prepared datasets persist one int64 seed instead of a full FP16 noise
    cloud while reconstructing all samples together on the training device.
    """
    if not sample_shape or any(int(size) < 1 for size in sample_shape):
        raise ValueError("sample_shape must contain positive dimensions")
    target_device = torch.device(device) if device is not None else seeds.device
    seed_values = seeds.to(device=target_device, dtype=torch.int64).reshape(-1)
    count = math.prod(int(size) for size in sample_shape)
    counters = torch.arange(count, device=target_device, dtype=torch.int64) * 2
    first = _pcg_uniform(seed_values, counters)
    second = _pcg_uniform(seed_values, counters + 1)
    radius = torch.sqrt(-2.0 * torch.log(first))
    values = radius * torch.cos((2.0 * math.pi) * second)
    return values.reshape(seed_values.shape[0], *sample_shape).to(dtype=dtype)
