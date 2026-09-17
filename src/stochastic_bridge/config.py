from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    root: str = ""
    prepared_root: str = "data/prepared"
    prepared_cache_shards: int = 2
    synthetic: bool = False
    synthetic_length: int = 256
    image_size: int = 64
    batch_size: int = 8
    workers: int = 0
    prefetch_factor: int = 2
    random_crop: bool = True
    horizontal_flip: bool = True


@dataclass
class PrepareConfig:
    variants_per_image: int = 4
    shard_size: int = 64
    batch_size: int = 8
    workers: int = 0
    seed: int = 17


@dataclass
class ScheduleConfig:
    steps: int = 100
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    answer_jump: int = 10


@dataclass
class GoalCorruptionConfig:
    enabled: bool = True
    blur_probability: float = 0.35
    downsample_probability: float = 0.50
    mask_probability: float = 0.25
    minimum_scale: float = 0.25
    maximum_mask_fraction: float = 0.30


@dataclass
class CorruptionConfig:
    """Mixture used for current/goal states and answer-cloud simulation."""

    enabled: bool = True
    spatial_floor: float = 0.08
    student_t_df: float = 3.0
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "white_gaussian": 40.0,
            "local_gaussian": 12.0,
            "edge_gaussian": 8.0,
            "low_frequency_gaussian": 6.0,
            "high_frequency_gaussian": 6.0,
            "band_pass_gaussian": 4.0,
            "channel_correlated_gaussian": 4.0,
            "low_rank_gaussian": 4.0,
            "anisotropic_gaussian": 4.0,
            "signal_dependent_gaussian": 4.0,
            "student_t": 2.0,
            "laplace": 2.0,
            "poisson": 2.0,
            "speckle": 2.0,
            "impulse": 2.0,
            "blur": 3.0,
            "downsample": 3.0,
            "mask": 3.0,
        }
    )


@dataclass
class ModelConfig:
    in_channels: int = 3
    base_channels: int = 32
    heads: int = 8
    attention_depth: int = 2
    max_residual: float = 1.0
    encoder_type: str = "cnn"
    vit_depth: int = 4
    vit_patch_size: int = 8


@dataclass
class LossConfig:
    name: str = "energy"
    samples: int = 4
    sinkhorn_blur: float = 0.05


@dataclass
class TrainConfig:
    epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    amp: bool = True
    amp_dtype: str = "float16"
    channels_last: bool = True
    tf32: bool = True
    fused_optimizer: bool = True
    compile: bool = False
    compile_mode: str = "reduce-overhead"
    seed: int = 17
    log_every: int = 20
    preview_every: int = 200
    save_every: int = 1000
    output_dir: str = "runs/default"
    resume: str = ""


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    prepare: PrepareConfig = field(default_factory=PrepareConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    corruption: CorruptionConfig = field(default_factory=CorruptionConfig)
    goal_corruption: GoalCorruptionConfig = field(default_factory=GoalCorruptionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return config_from_dict(raw)


def config_from_dict(raw: dict[str, Any]) -> ExperimentConfig:
    return ExperimentConfig(
        data=DataConfig(**raw.get("data", {})),
        prepare=PrepareConfig(**raw.get("prepare", {})),
        schedule=ScheduleConfig(**raw.get("schedule", {})),
        corruption=CorruptionConfig(**raw.get("corruption", {})),
        goal_corruption=GoalCorruptionConfig(**raw.get("goal_corruption", {})),
        model=ModelConfig(**raw.get("model", {})),
        loss=LossConfig(**raw.get("loss", {})),
        train=TrainConfig(**raw.get("train", {})),
    )
