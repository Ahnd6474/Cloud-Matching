from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .noise import CorruptionMixture
from .schedule import VPNoiseSchedule
from .stateless import stateless_normal


@dataclass(frozen=True)
class BridgeBatch:
    clean: Tensor
    current: Tensor
    goal: Tensor
    target_cloud: Tensor
    target_noise: Tensor | None
    target_noise_seed: Tensor | None
    current_level: Tensor
    goal_level: Tensor
    answer_level: Tensor
    corruption_types: tuple[str, ...] | None = None


def sample_level_triplet(
    batch_size: int,
    max_level: int,
    answer_jump: int = 10,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
    clean_answer_probability: float = 0.0,
    goal_from_clean: bool = False,
    answer_from_current: bool = False,
    adaptive_answer_jump: bool = False,
    near_clean_threshold: int = 20,
    near_clean_answer_jump: int = 2,
    near_clean_answer_jump_min: int = 0,
    mid_clean_threshold: int = 50,
    mid_clean_answer_jump: int = 5,
    mid_clean_answer_jump_min: int = 0,
    near_clean_probability: float = 0.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Sample current ``s``, goal ``r``, and answer ``a`` levels.

    ``s`` is the current/input corruption level, ``r`` is the image-goal
    corruption level, and ``a`` is the answer level requested by the user.
    With ``goal_from_clean``, ``r=0`` and ``s`` is uniform over ``[1, T]``.
    With ``answer_from_current``, ``a=max(s-answer_jump, 0)``; otherwise the
    legacy bridge rule ``a=max(r-answer_jump, 0)`` is retained.
    ``clean_answer_probability`` adds direct arbitrary-state-to-x_0 examples.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if max_level < 1:
        raise ValueError("max_level must be positive")
    if answer_jump < 1:
        raise ValueError("answer_jump must be positive")
    if not 0.0 <= clean_answer_probability <= 1.0:
        raise ValueError("clean_answer_probability must lie in [0, 1]")
    if not 0.0 <= near_clean_probability <= 1.0:
        raise ValueError("near_clean_probability must lie in [0, 1]")
    if near_clean_threshold < 1 or mid_clean_threshold < near_clean_threshold:
        raise ValueError("adaptive answer thresholds must be ordered and positive")
    if min(near_clean_answer_jump, mid_clean_answer_jump) < 1:
        raise ValueError("adaptive answer jumps must be positive")
    near_jump_min = near_clean_answer_jump_min or near_clean_answer_jump
    mid_jump_min = mid_clean_answer_jump_min or mid_clean_answer_jump
    if not 1 <= near_jump_min <= near_clean_answer_jump:
        raise ValueError("near-clean jump range is invalid")
    if not 1 <= mid_jump_min <= mid_clean_answer_jump:
        raise ValueError("mid-clean jump range is invalid")

    if goal_from_clean:
        goal_level = torch.zeros(batch_size, device=device, dtype=torch.long)
    else:
        goal_level = torch.randint(
            0,
            max_level,
            (batch_size,),
            device=device,
            generator=generator,
        )
    # Uniformly sample one integer from [goal+1, max_level] per item.
    span = max_level - goal_level
    offset = torch.floor(
        torch.rand(batch_size, device=device, generator=generator) * span
    ).long()
    current_level = goal_level + 1 + offset
    if goal_from_clean and near_clean_probability > 0.0:
        near_max = min(near_clean_threshold, max_level)
        near_level = torch.randint(
            1,
            near_max + 1,
            (batch_size,),
            device=device,
            generator=generator,
        )
        choose_near = (
            torch.rand(batch_size, device=device, generator=generator)
            < near_clean_probability
        )
        current_level = torch.where(choose_near, near_level, current_level)
    answer_base = current_level if answer_from_current else goal_level
    effective_jump = torch.full_like(answer_base, answer_jump)
    if adaptive_answer_jump and answer_from_current:
        mid_jump = torch.randint(
            mid_jump_min,
            mid_clean_answer_jump + 1,
            (batch_size,),
            device=device,
            generator=generator,
        )
        near_jump = torch.randint(
            near_jump_min,
            near_clean_answer_jump + 1,
            (batch_size,),
            device=device,
            generator=generator,
        )
        effective_jump = torch.where(
            current_level <= mid_clean_threshold,
            mid_jump,
            effective_jump,
        )
        effective_jump = torch.where(
            current_level <= near_clean_threshold,
            near_jump,
            effective_jump,
        )
    answer_level = (answer_base - effective_jump).clamp_min(0)
    if clean_answer_probability > 0.0:
        clean_answer = (
            torch.rand(batch_size, device=device, generator=generator)
            < clean_answer_probability
        )
        answer_level = torch.where(
            clean_answer, torch.zeros_like(answer_level), answer_level
        )
    return current_level, goal_level, answer_level


def build_bridge_batch(
    clean: Tensor,
    schedule: VPNoiseSchedule,
    target_samples: int = 8,
    answer_jump: int = 10,
    generator: torch.Generator | None = None,
    goal_corruptor: nn.Module | None = None,
    corruption_mixture: CorruptionMixture | None = None,
    clean_answer_probability: float = 0.0,
    endpoint_corruption_mixture: CorruptionMixture | None = None,
    endpoint_corruption_probability: float = 0.0,
    goal_from_clean: bool = False,
    answer_from_current: bool = False,
    adaptive_answer_jump: bool = False,
    near_clean_threshold: int = 20,
    near_clean_answer_jump: int = 2,
    near_clean_answer_jump_min: int = 0,
    mid_clean_threshold: int = 50,
    mid_clean_answer_jump: int = 5,
    mid_clean_answer_jump_min: int = 0,
    near_clean_probability: float = 0.0,
) -> BridgeBatch:
    """Construct current, image-goal, and cumulative answer cloud.

    With ``corruption_mixture=None`` this uses the exact arbitrary-skip VP
    posterior. Structured endpoint corruptions may replace the input states
    only when the answer is x_0, whose posterior is a clean point mass.
    Otherwise, one corruption family is sampled per item and an independent
    marginal answer cloud is simulated at the answer level.
    """
    if corruption_mixture is not None and endpoint_corruption_mixture is not None:
        raise ValueError(
            "endpoint corruption requires the analytic VP path; do not combine "
            "it with the general corruption mixture"
        )
    current_level, goal_level, answer_level = sample_level_triplet(
        batch_size=clean.shape[0],
        max_level=schedule.steps,
        answer_jump=answer_jump,
        device=clean.device,
        generator=generator,
        clean_answer_probability=clean_answer_probability,
        goal_from_clean=goal_from_clean,
        answer_from_current=answer_from_current,
        adaptive_answer_jump=adaptive_answer_jump,
        near_clean_threshold=near_clean_threshold,
        near_clean_answer_jump=near_clean_answer_jump,
        near_clean_answer_jump_min=near_clean_answer_jump_min,
        mid_clean_threshold=mid_clean_threshold,
        mid_clean_answer_jump=mid_clean_answer_jump,
        mid_clean_answer_jump_min=mid_clean_answer_jump_min,
        near_clean_probability=near_clean_probability,
    )
    if not 0.0 <= endpoint_corruption_probability <= 1.0:
        raise ValueError("endpoint_corruption_probability must lie in [0, 1]")
    corruption_types: tuple[str, ...] | None = None
    target_noise_seed = torch.randint(
        0,
        1 << 32,
        (clean.shape[0],),
        device=clean.device,
        dtype=torch.int64,
        generator=generator,
    )
    target_noise = stateless_normal(
        target_noise_seed,
        (target_samples, *clean.shape[1:]),
        device=clean.device,
        dtype=clean.dtype,
    )
    if corruption_mixture is None:
        current_noise = torch.randn(
            clean.shape, device=clean.device, dtype=clean.dtype, generator=generator
        )
        goal_noise = torch.randn(
            clean.shape, device=clean.device, dtype=clean.dtype, generator=generator
        )
        current = schedule.q_sample(clean, current_level, current_noise)
        goal = schedule.q_sample(clean, goal_level, goal_noise)
        target_cloud = schedule.sample_answer_cloud(
            clean=clean,
            current=current,
            current_level=current_level,
            answer_level=answer_level,
            samples=target_samples,
            noise=target_noise,
        )
        # Non-VP structured degradations are valid with the paired objective at
        # the clean endpoint: q(x_0 | ...) is exactly the clean point mass, so
        # no synthetic intermediate posterior needs to be assumed.
        if endpoint_corruption_mixture is not None:
            selected = (answer_level == 0) & (
                torch.rand(
                    clean.shape[0], device=clean.device, generator=generator
                )
                < endpoint_corruption_probability
            )
            selected_indices = selected.nonzero(as_tuple=False).flatten()
            labels = ["white_gaussian"] * clean.shape[0]
            if selected_indices.numel() > 0:
                subset_types = endpoint_corruption_mixture.sample_types(
                    int(selected_indices.numel()), clean.device
                )
                subset_clean = clean[selected_indices]
                structured_current = endpoint_corruption_mixture.corrupt(
                    subset_clean, current_level[selected_indices], subset_types
                )
                structured_goal = endpoint_corruption_mixture.corrupt(
                    subset_clean, goal_level[selected_indices], subset_types
                )
                current = current.clone()
                goal = goal.clone()
                current[selected_indices] = structured_current
                goal[selected_indices] = structured_goal
                for index, name in zip(
                    selected_indices.tolist(), subset_types, strict=True
                ):
                    labels[index] = name
            corruption_types = tuple(labels)
    else:
        corruption_types = corruption_mixture.sample_types(clean.shape[0], clean.device)
        current = corruption_mixture.corrupt(clean, current_level, corruption_types)
        goal = corruption_mixture.corrupt(clean, goal_level, corruption_types)
        target_cloud = corruption_mixture.sample_cloud(
            clean, answer_level, corruption_types, target_samples
        )
    if goal_corruptor is not None:
        goal = goal_corruptor(goal)
    return BridgeBatch(
        clean=clean,
        current=current,
        goal=goal,
        target_cloud=target_cloud,
        target_noise=target_noise,
        target_noise_seed=target_noise_seed,
        current_level=current_level,
        goal_level=goal_level,
        answer_level=answer_level,
        corruption_types=corruption_types,
    )
