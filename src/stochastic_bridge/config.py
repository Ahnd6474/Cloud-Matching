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
    crop_mode: str = "resized"
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
    # In addition to ordinary short bridge transitions, explicitly train a
    # substantial fraction of arbitrary noisy states to terminate at x_0.
    clean_answer_probability: float = 0.0


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
class EndpointCorruptionConfig:
    """Structured degradations used only when the requested answer is clean.

    Restricting these non-VP corruptions to x_0 targets keeps the target a
    point mass.  They can therefore be mixed with analytically paired VP
    transitions without inventing an invalid intermediate posterior.
    """

    enabled: bool = False
    probability: float = 0.0
    spatial_floor: float = 0.08
    student_t_df: float = 3.0
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "local_gaussian": 2.0,
            "edge_gaussian": 2.0,
            "high_frequency_gaussian": 1.0,
            "texture_suppress": 3.0,
            "edge_erase": 3.0,
            "blur": 1.0,
            "downsample": 1.0,
            "mask": 1.0,
        }
    )


@dataclass
class ModelConfig:
    architecture: str = "pyramid"
    in_channels: int = 3
    base_channels: int = 32
    heads: int = 8
    attention_depth: int = 2
    max_residual: float = 1.0
    noise_variance_min: float = 1e-4
    noise_variance_max: float = 1.0
    noise_variance_init: float = 0.1
    noise_energy_parameterization: str = "bounded"
    noise_amplitude_safety_max: float = 8.0
    encoder_type: str = "cnn"
    vit_depth: int = 4
    vit_patch_size: int = 8
    decoder_type: str = "conv"
    implicit_hidden_dim: int = 128
    implicit_depth: int = 3
    implicit_fourier_bands: int = 6
    implicit_chunk_size: int = 65536
    fullres_dim: int = 320
    fullres_depth: int = 12
    fullres_cross_depth: int = 2
    fullres_ffn_ratio: float = 2.0
    fullres_window_size: int = 8
    fullres_gradient_checkpointing: bool = True


@dataclass
class LossConfig:
    name: str = "energy"
    samples: int = 4
    sinkhorn_blur: float = 0.05
    full_band_levels: int = 3
    full_band_charbonnier_epsilon: float = 1e-3
    full_band_high_weight: float = 1.0
    full_band_low_weight: float = 1.0
    paired_mean_weight: float = 1.0
    paired_deviation_weight: float = 1.0
    paired_variance_weight: float = 0.1
    spatial_ce_weight: float = 0.0
    spatial_ce_highpass: bool = True
    spatial_ce_kernel_size: int = 5


@dataclass
class TrainConfig:
    epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    amp: bool = True
    amp_dtype: str = "float16"
    amp_init_scale: float = 1024.0
    amp_growth_interval: int = 2000
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
    endpoint_corruption: EndpointCorruptionConfig = field(
        default_factory=EndpointCorruptionConfig
    )
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
        endpoint_corruption=EndpointCorruptionConfig(
            **raw.get("endpoint_corruption", {})
        ),
        goal_corruption=GoalCorruptionConfig(**raw.get("goal_corruption", {})),
        model=ModelConfig(**raw.get("model", {})),
        loss=LossConfig(**raw.get("loss", {})),
        train=TrainConfig(**raw.get("train", {})),
    )
