from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm.auto import tqdm

from stochastic_bridge.config import ExperimentConfig, load_config
from stochastic_bridge.data import BridgeBatch, build_bridge_batch
from stochastic_bridge.datasets import CocoCaptionDataset
from stochastic_bridge.losses import (
    PairedMeanDeviationFullBandLoss,
    PairedPerceptualCloudLoss,
    SpatialNoiseCrossEntropyLoss,
    build_cloud_loss,
    is_paired_cloud_loss,
)
from stochastic_bridge.model import StochasticImageBridge
from stochastic_bridge.noise import CorruptionMixture
from stochastic_bridge.perceptual import EfficientNetB0Features
from stochastic_bridge.schedule import VPNoiseSchedule
from stochastic_bridge.text import CocoWordTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="COCO caption-conditioned DDP training")
    parser.add_argument("--config", default="configs/coco_text_fullres.yaml")
    parser.add_argument("--coco-root", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--resume", default="")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--validation-images", type=int, default=4)
    return parser.parse_args()


def distributed_setup(requested_device: str) -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = torch.cuda.is_available() and requested_device != "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if world_size > 1:
        if use_cuda:
            torch.cuda.set_device(local_rank)
        init_method = os.environ.get("CLOUD_MATCHING_DIST_INIT_METHOD", "env://")
        kwargs: dict[str, Any] = {}
        if use_cuda:
            kwargs["device_id"] = torch.device(f"cuda:{local_rank}")
        if init_method != "env://":
            kwargs.update(rank=rank, world_size=world_size)
        dist.init_process_group(
            backend="nccl" if use_cuda else "gloo",
            init_method=init_method,
            **kwargs,
        )
    return rank, local_rank, world_size, torch.device(
        f"cuda:{local_rank}" if use_cuda else "cpu"
    )


def seed_everything(seed: int, rank: int) -> None:
    seed += rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_bridge_components(
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[VPNoiseSchedule, CorruptionMixture | None, CorruptionMixture | None]:
    schedule = VPNoiseSchedule(
        config.schedule.steps,
        config.schedule.beta_start,
        config.schedule.beta_end,
    ).to(device)
    corruption = config.corruption
    mixture = (
        CorruptionMixture(
            schedule,
            corruption.weights,
            corruption.spatial_floor,
            corruption.student_t_df,
        ).to(device)
        if corruption.enabled
        else None
    )
    endpoint = config.endpoint_corruption
    endpoint_mixture = (
        CorruptionMixture(
            schedule,
            endpoint.weights,
            endpoint.spatial_floor,
            endpoint.student_t_df,
        ).to(device)
        if endpoint.enabled and endpoint.probability > 0.0
        else None
    )
    return schedule, mixture, endpoint_mixture


def make_bridge_batch(
    clean: Tensor,
    config: ExperimentConfig,
    schedule: VPNoiseSchedule,
    mixture: CorruptionMixture | None,
    endpoint_mixture: CorruptionMixture | None,
    generator: torch.Generator | None = None,
) -> BridgeBatch:
    schedule_config = config.schedule
    return build_bridge_batch(
        clean,
        schedule,
        target_samples=config.loss.samples,
        answer_jump=schedule_config.answer_jump,
        generator=generator,
        corruption_mixture=mixture,
        clean_answer_probability=schedule_config.clean_answer_probability,
        goal_from_clean=schedule_config.goal_from_clean,
        answer_from_current=schedule_config.answer_from_current,
        adaptive_answer_jump=schedule_config.adaptive_answer_jump,
        near_clean_threshold=schedule_config.near_clean_threshold,
        near_clean_answer_jump=schedule_config.near_clean_answer_jump,
        near_clean_answer_jump_min=schedule_config.near_clean_answer_jump_min,
        mid_clean_threshold=schedule_config.mid_clean_threshold,
        mid_clean_answer_jump=schedule_config.mid_clean_answer_jump,
        mid_clean_answer_jump_min=schedule_config.mid_clean_answer_jump_min,
        near_clean_probability=schedule_config.near_clean_probability,
        endpoint_corruption_mixture=endpoint_mixture,
        endpoint_corruption_probability=config.endpoint_corruption.probability,
    )


def encode_captions(
    tokenizer: CocoWordTokenizer,
    captions: list[str],
    device: torch.device,
    dropout: float = 0.0,
) -> tuple[Tensor, Tensor, list[str]]:
    conditioned = [
        "" if dropout > 0.0 and random.random() < dropout else caption
        for caption in captions
    ]
    token_ids, token_mask = tokenizer.batch_encode(conditioned)
    return (
        token_ids.to(device, non_blocking=True),
        token_mask.to(device, non_blocking=True),
        conditioned,
    )


def cloud_components(
    criterion: nn.Module,
    predicted: Tensor,
    target: Tensor,
    current: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if isinstance(criterion, PairedMeanDeviationFullBandLoss):
        return criterion.components(predicted, target, current)
    total = criterion(predicted, target, current)
    zero = total.new_zeros(())
    return total, zero, zero, zero


def psnr(predicted: Tensor, target: Tensor) -> Tensor:
    mse = (predicted - target).square().flatten(1).mean(1).clamp_min(1e-10)
    return 10.0 * torch.log10(4.0 / mse)


def diversity(cloud: Tensor) -> Tensor:
    return cloud.detach().float().std(dim=1, unbiased=False).mean(dim=(1, 2, 3))


def optimizer_groups(
    model: nn.Module,
    base_lr: float,
    text_multiplier: float,
    weight_decay: float,
) -> list[dict[str, Any]]:
    if text_multiplier <= 0.0:
        raise ValueError("text learning-rate multiplier must be positive")
    regular: list[nn.Parameter] = []
    gates: list[nn.Parameter] = []
    text: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if ".text_encoder." in name:
            text.append(parameter)
        elif (
            (".random_attentions." in name and name.endswith(".gate"))
            or ".gate_projection" in name
        ):
            gates.append(parameter)
        else:
            regular.append(parameter)
    groups: list[dict[str, Any]] = [
        {"params": regular, "lr": base_lr, "weight_decay": weight_decay},
        {"params": text, "lr": base_lr * text_multiplier, "weight_decay": weight_decay},
    ]
    if gates:
        groups.append({"params": gates, "lr": base_lr, "weight_decay": 0.0})
    return groups


def display(images: Tensor) -> Tensor:
    return (images.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) / 2.0


@torch.no_grad()
def validate(
    model: StochasticImageBridge,
    loader: DataLoader,
    tokenizer: CocoWordTokenizer,
    config: ExperimentConfig,
    schedule: VPNoiseSchedule,
    mixture: CorruptionMixture | None,
    endpoint_mixture: CorruptionMixture | None,
    criterion: nn.Module,
    perceptual: PairedPerceptualCloudLoss | None,
    spatial: SpatialNoiseCrossEntropyLoss | None,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    max_batches: int,
    writer: SummaryWriter,
    epoch: int,
    validation_images: int,
) -> dict[str, float]:
    model.eval()
    totals = torch.zeros(9, device=device)
    generator = torch.Generator(device=device).manual_seed(config.train.seed + 10_000)
    first_logged = False
    for batch_index, raw in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        clean = raw["clean"].to(device, non_blocking=True)
        bridge = make_bridge_batch(
            clean, config, schedule, mixture, endpoint_mixture, generator
        )
        captions = list(raw["caption"])
        token_ids, token_mask, _ = encode_captions(tokenizer, captions, device)
        paired = is_paired_cloud_loss(config.loss.name)
        fixed_noise = bridge.target_noise
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            output = model(
                bridge.current,
                token_ids,
                samples=config.loss.samples,
                noise=fixed_noise if (paired or fixed_noise is not None) else None,
                return_noise_variance=True,
                return_noise_energy=spatial is not None,
                goal_mask=token_mask,
            )
            assert isinstance(output, tuple)
            predicted, variance = output[:2]
            cloud_loss, _, _, _ = cloud_components(
                criterion, predicted, bridge.target_cloud, bridge.current
            )
            perceptual_loss = cloud_loss.new_zeros(())
            if perceptual is not None:
                perceptual_loss = perceptual(predicted, bridge.target_cloud)
            spatial_loss = cloud_loss.new_zeros(())
            if spatial is not None:
                spatial_loss = spatial(output[2], bridge.target_cloud, bridge.current)
            loss = (
                cloud_loss
                + config.loss.perceptual_weight * perceptual_loss
                + config.loss.spatial_ce_weight * spatial_loss
            )
        count = clean.shape[0]
        predicted_mean = predicted.float().mean(1)
        totals += torch.stack(
            [
                loss.float() * count,
                psnr(bridge.current.float(), clean.float()).sum(),
                psnr(predicted_mean, clean.float()).sum(),
                diversity(predicted).sum(),
                diversity(bridge.target_cloud).sum(),
                variance.float().sum(),
                torch.tensor(float(count), device=device),
                cloud_loss.float() * count,
                perceptual_loss.float() * count,
            ]
        )
        if not first_logged and validation_images > 0:
            first_logged = True
            count_preview = min(validation_images, count)
            shuffled_ids = token_ids.roll(1, 0)
            shuffled_mask = token_mask.roll(1, 0)
            shuffled = model(
                bridge.current,
                shuffled_ids,
                samples=config.loss.samples,
                noise=fixed_noise,
                goal_mask=shuffled_mask,
            )
            if isinstance(shuffled, tuple):
                shuffled = shuffled[0]
            sensitivity = (
                predicted.float().mean(1) - shuffled.float().mean(1)
            ).abs().mean()
            writer.add_scalar("text/shuffled_caption_output_l1", sensitivity.item(), epoch)
            columns = torch.stack(
                [
                    display(clean[:count_preview]),
                    display(bridge.current[:count_preview]),
                    display(bridge.target_cloud[:count_preview].mean(1)),
                    display(predicted[:count_preview].mean(1)),
                    display(predicted[:count_preview, 0]),
                ],
                dim=1,
            ).flatten(0, 1)
            writer.add_image(
                "validation/comparison",
                make_grid(columns, nrow=5, padding=2),
                epoch,
            )
            writer.add_text(
                "validation/captions",
                "  \n".join(
                    f"{index}: {caption}"
                    for index, caption in enumerate(captions[:count_preview])
                ),
                epoch,
            )
    denominator = max(totals[6].item(), 1.0)
    model.train()
    return {
        "val_loss": totals[0].item() / denominator,
        "val_current_psnr": totals[1].item() / denominator,
        "val_output_psnr": totals[2].item() / denominator,
        "val_predicted_diversity": totals[3].item() / denominator,
        "val_target_diversity": totals[4].item() / denominator,
        "val_noise_variance": totals[5].item() / denominator,
        "val_cloud_loss": totals[7].item() / denominator,
        "val_perceptual_loss": totals[8].item() / denominator,
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    history: list[dict[str, float]],
    config: ExperimentConfig,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "history": history,
            "config": config.to_dict(),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size, device = distributed_setup(args.device)
    is_main = rank == 0
    config = load_config(args.config)
    if not config.text.enabled or config.model.fullres_goal_condition != "text":
        raise ValueError("COCO text training requires text.enabled and a text goal")
    if args.epochs is not None:
        config.train.epochs = args.epochs
    seed_everything(config.train.seed, rank)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = config.train.tf32
        torch.backends.cudnn.allow_tf32 = config.train.tf32
    torch.set_float32_matmul_precision("high")

    coco_root = Path(args.coco_root or config.data.root).expanduser().resolve()
    vocab_path = Path(config.text.vocab_path)
    if not vocab_path.is_absolute():
        vocab_path = (Path.cwd() / vocab_path).resolve()
    tokenizer = CocoWordTokenizer.load(vocab_path)
    # Frequency filtering can yield slightly fewer entries than the requested
    # cap. The saved vocabulary is authoritative for the embedding table.
    config.text.vocab_size = len(tokenizer)
    config.model.text_vocab_size = len(tokenizer)
    if tokenizer.max_length != config.model.text_max_length:
        raise ValueError("tokenizer/model max-length mismatch")

    train_dataset = CocoCaptionDataset(
        coco_root / "train2017",
        coco_root / "annotations" / "captions_train2017.json",
        config.data.image_size,
        random_caption=True,
        horizontal_flip=config.data.horizontal_flip,
    )
    val_dataset = CocoCaptionDataset(
        coco_root / "val2017",
        coco_root / "annotations" / "captions_val2017.json",
        config.data.image_size,
        random_caption=False,
        horizontal_flip=False,
    )
    sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=config.train.seed,
        drop_last=True,
    )
    loader_kwargs: dict[str, Any] = {
        "num_workers": config.data.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": config.data.workers > 0,
    }
    if config.data.workers > 0:
        loader_kwargs["prefetch_factor"] = config.data.prefetch_factor
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        sampler=sampler,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )
        if is_main
        else None
    )

    output = Path(args.output or config.train.output_dir).expanduser().resolve()
    if is_main:
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.json").write_text(
            json.dumps(config.to_dict(), indent=2), encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "train_images": len(train_dataset),
                    "val_images": len(val_dataset),
                    "world_size": world_size,
                    "per_gpu_batch": config.data.batch_size,
                    "global_batch": config.data.batch_size * world_size,
                    "steps_per_epoch": len(train_loader),
                    "vocabulary": len(tokenizer),
                },
                indent=2,
            )
        )
    if world_size > 1:
        dist.barrier()

    model = StochasticImageBridge(**config.model.__dict__).to(device)
    train_model: nn.Module = model
    if world_size > 1:
        train_model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            static_graph=True,
        )
    if config.train.compile:
        train_model = torch.compile(
            train_model,
            mode=config.train.compile_mode,
            fullgraph=False,
        )

    schedule, mixture, endpoint_mixture = build_bridge_components(config, device)
    paired = is_paired_cloud_loss(config.loss.name)
    if paired and mixture is not None:
        raise ValueError("paired cloud losses require corruption.enabled=false")
    criterion = build_cloud_loss(
        config.loss.name,
        blur=config.loss.sinkhorn_blur,
        full_band_levels=config.loss.full_band_levels,
        full_band_charbonnier_epsilon=config.loss.full_band_charbonnier_epsilon,
        full_band_high_weight=config.loss.full_band_high_weight,
        full_band_low_weight=config.loss.full_band_low_weight,
        paired_mean_weight=config.loss.paired_mean_weight,
        paired_deviation_weight=config.loss.paired_deviation_weight,
        paired_variance_weight=config.loss.paired_variance_weight,
    ).to(device)
    perceptual = (
        PairedPerceptualCloudLoss(
            EfficientNetB0Features(
                pretrained=config.loss.perceptual_pretrained,
                stages=config.loss.perceptual_stages,
            ),
            mean_weight=config.loss.perceptual_mean_weight,
            deviation_weight=config.loss.perceptual_deviation_weight,
        ).to(device)
        if config.loss.perceptual_weight > 0.0
        else None
    )
    spatial = (
        SpatialNoiseCrossEntropyLoss(
            highpass=config.loss.spatial_ce_highpass,
            kernel_size=config.loss.spatial_ce_kernel_size,
        ).to(device)
        if config.loss.spatial_ce_weight > 0.0
        else None
    )

    optimizer = torch.optim.AdamW(
        optimizer_groups(
            model,
            config.train.learning_rate,
            config.text.learning_rate_multiplier,
            config.train.weight_decay,
        ),
        lr=config.train.learning_rate,
        fused=config.train.fused_optimizer and device.type == "cuda",
    )
    if config.train.lr_schedule == "cosine":
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(config.train.epochs, 1)
            )
        )
    elif config.train.lr_schedule == "fixed":
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    else:
        raise ValueError("train.lr_schedule must be 'cosine' or 'fixed'")
    amp_enabled = config.train.amp and device.type == "cuda"
    amp_dtype = torch.float16 if config.train.amp_dtype in {"float16", "fp16"} else torch.bfloat16
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled and amp_dtype == torch.float16,
        init_scale=config.train.amp_init_scale,
        growth_interval=config.train.amp_growth_interval,
    )

    history: list[dict[str, float]] = []
    start_epoch = 0
    best_val = math.inf
    resume = args.resume or config.train.resume
    if resume:
        state = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        lr_scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state.get("scaler", {}))
        history = state.get("history", [])
        start_epoch = int(state.get("epoch", 0))
        best_val = min((row["val_loss"] for row in history), default=math.inf)

    writer = SummaryWriter(output / "tensorboard") if is_main else None
    if writer is not None:
        writer.add_text(
            "run/validation_comparison_columns",
            "clean | current | target_mean | prediction_mean | prediction_sample_0",
            0,
        )

    for epoch in range(start_epoch, config.train.epochs):
        sampler.set_epoch(epoch)
        train_model.train()
        totals = torch.zeros(8, device=device)
        progress = tqdm(
            train_loader,
            desc=f"epoch {epoch + 1}/{config.train.epochs}",
            disable=not is_main,
        )
        for batch_index, raw in enumerate(progress):
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
            optimizer.zero_grad(set_to_none=True)
            clean = raw["clean"].to(device, non_blocking=True)
            bridge = make_bridge_batch(
                clean, config, schedule, mixture, endpoint_mixture
            )
            token_ids, token_mask, _ = encode_captions(
                tokenizer,
                list(raw["caption"]),
                device,
                dropout=config.text.condition_dropout,
            )
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                output_value = train_model(
                    bridge.current,
                    token_ids,
                    samples=config.loss.samples,
                    noise=bridge.target_noise if paired else None,
                    return_noise_variance=True,
                    return_noise_energy=spatial is not None,
                    goal_mask=token_mask,
                )
                assert isinstance(output_value, tuple)
                predicted, variance = output_value[:2]
                cloud_loss, mean_loss, deviation_loss, variance_loss = cloud_components(
                    criterion, predicted, bridge.target_cloud, bridge.current
                )
                perceptual_loss = cloud_loss.new_zeros(())
                if perceptual is not None:
                    perceptual_loss = perceptual(predicted, bridge.target_cloud)
                spatial_loss = cloud_loss.new_zeros(())
                if spatial is not None:
                    spatial_loss = spatial(
                        output_value[2], bridge.target_cloud, bridge.current
                    )
                loss = (
                    cloud_loss
                    + config.loss.perceptual_weight * perceptual_loss
                    + config.loss.spatial_ce_weight * spatial_loss
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.train.grad_clip
            )
            scaler.step(optimizer)
            scaler.update()
            count = clean.shape[0]
            totals += torch.stack(
                [
                    loss.detach().float() * count,
                    psnr(predicted.detach().float().mean(1), clean.float()).sum(),
                    diversity(predicted).sum(),
                    diversity(bridge.target_cloud).sum(),
                    variance.detach().float().sum(),
                    grad_norm.detach().float(),
                    torch.tensor(float(count), device=device),
                    cloud_loss.detach().float() * count,
                ]
            )
            if is_main and (batch_index + 1) % config.train.log_every == 0:
                assert writer is not None
                step = epoch * len(train_loader) + batch_index + 1
                progress.set_postfix(loss=f"{loss.item():.4f}")
                writer.add_scalar("batch/loss", loss.item(), step)
                writer.add_scalar("batch/gradient_norm", grad_norm.item(), step)
                writer.add_scalar("batch/generator_lr", optimizer.param_groups[0]["lr"], step)
                writer.add_scalar("batch/text_lr", optimizer.param_groups[1]["lr"], step)

        if world_size > 1:
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        denominator = max(totals[6].item(), 1.0)
        metrics = {
            "epoch": float(epoch + 1),
            "train_loss": totals[0].item() / denominator,
            "train_output_psnr": totals[1].item() / denominator,
            "train_predicted_diversity": totals[2].item() / denominator,
            "train_target_diversity": totals[3].item() / denominator,
            "train_noise_variance": totals[4].item() / denominator,
            "gradient_norm": totals[5].item() / max(len(train_loader) * world_size, 1),
            "train_cloud_loss": totals[7].item() / denominator,
            "generator_lr": optimizer.param_groups[0]["lr"],
            "text_lr": optimizer.param_groups[1]["lr"],
        }
        lr_scheduler.step()
        if world_size > 1:
            dist.barrier()
        if is_main:
            assert val_loader is not None and writer is not None
            metrics.update(
                validate(
                    model,
                    val_loader,
                    tokenizer,
                    config,
                    schedule,
                    mixture,
                    endpoint_mixture,
                    criterion,
                    perceptual,
                    spatial,
                    device,
                    amp_enabled,
                    amp_dtype,
                    args.max_val_batches,
                    writer,
                    epoch + 1,
                    args.validation_images,
                )
            )
            history.append(metrics)
            for name in (
                "train_loss",
                "val_loss",
                "train_output_psnr",
                "val_output_psnr",
                "val_current_psnr",
                "train_predicted_diversity",
                "train_target_diversity",
                "val_predicted_diversity",
                "val_target_diversity",
                "gradient_norm",
                "generator_lr",
                "text_lr",
            ):
                writer.add_scalar(f"epoch/{name}", metrics[name], epoch + 1)
            writer.flush()
            (output / "history.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8"
            )
            save_checkpoint(
                output / "latest.pt",
                model,
                optimizer,
                lr_scheduler,
                scaler,
                epoch + 1,
                history,
                config,
            )
            if metrics["val_loss"] < best_val:
                best_val = metrics["val_loss"]
                save_checkpoint(
                    output / "best.pt",
                    model,
                    optimizer,
                    lr_scheduler,
                    scaler,
                    epoch + 1,
                    history,
                    config,
                )
            print(json.dumps(metrics, indent=2))
        if world_size > 1:
            dist.barrier()

    if is_main:
        assert writer is not None
        save_checkpoint(
            output / "final.pt",
            model,
            optimizer,
            lr_scheduler,
            scaler,
            config.train.epochs,
            history,
            config,
        )
        writer.close()
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
