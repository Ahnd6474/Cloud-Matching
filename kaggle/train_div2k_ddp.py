from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.utils import make_grid
from tqdm.auto import tqdm

from stochastic_bridge.config import load_config
from stochastic_bridge.losses import (
    PairedMeanDeviationFullBandLoss,
    SpatialNoiseCrossEntropyLoss,
    build_cloud_loss,
    is_paired_cloud_loss,
)
from stochastic_bridge.model import StochasticImageBridge
from stochastic_bridge.prepared import PreparedBridgeDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DIV2K Gaussian-only DDP training")
    parser.add_argument("--config", default="configs/kaggle_div2k_gaussian.yaml")
    parser.add_argument("--train-prepared", required=True)
    parser.add_argument("--val-prepared", required=True)
    parser.add_argument("--output", default="/kaggle/working/cloud_matching_div2k")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--resume", default="")
    parser.add_argument(
        "--init-checkpoint",
        default="",
        help="Load model weights only and start a fresh fine-tuning schedule",
    )
    parser.add_argument(
        "--reset-energy-head",
        action="store_true",
        help="Reset the spatial energy head after loading initialization weights",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument(
        "--tensorboard-dir",
        default="",
        help="TensorBoard log directory (default: OUTPUT/tensorboard)",
    )
    parser.add_argument(
        "--validation-images",
        type=int,
        default=4,
        help="Number of validation examples to visualize per epoch (0 disables)",
    )
    return parser.parse_args()


def distributed_setup(requested_device: str) -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = torch.cuda.is_available() and requested_device != "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if world_size > 1:
        backend = "nccl" if use_cuda else "gloo"
        if use_cuda:
            torch.cuda.set_device(local_rank)
        # torchrun/Kaggle uses env://.  A file store is also accepted so the
        # same trainer can be smoke-tested on Windows builds without libuv.
        init_method = os.environ.get("CLOUD_MATCHING_DIST_INIT_METHOD", "env://")
        init_kwargs: dict[str, Any] = {}
        if init_method != "env://":
            init_kwargs.update(rank=rank, world_size=world_size)
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            **init_kwargs,
        )
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
    return rank, local_rank, world_size, device


def seed_everything(seed: int, rank: int) -> None:
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reduce_totals(values: Tensor, world_size: int) -> Tensor:
    if world_size > 1:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return values


def psnr(predicted: Tensor, target: Tensor) -> Tensor:
    mse = (predicted - target).square().flatten(1).mean(1).clamp_min(1e-10)
    return 10.0 * torch.log10(4.0 / mse)


def cloud_loss_components(
    criterion: nn.Module,
    predicted_cloud: Tensor,
    target_cloud: Tensor,
    current: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if isinstance(criterion, PairedMeanDeviationFullBandLoss):
        return criterion.components(predicted_cloud, target_cloud, current)
    total = criterion(predicted_cloud, target_cloud, current)
    zero = total.new_zeros(())
    return total, zero, zero, zero


def cloud_diversity(cloud: Tensor) -> Tensor:
    """Mean per-pixel sample standard deviation for each condition."""
    return cloud.detach().float().std(dim=1, unbiased=False).mean(dim=(1, 2, 3))


def move_batch(
    raw: dict[str, Tensor], device: torch.device, channels_last: bool
) -> dict[str, Tensor]:
    moved: dict[str, Tensor] = {}
    for key in (
        "clean",
        "current",
        "goal",
        "target_cloud",
        "current_level",
        "goal_level",
        "answer_level",
    ):
        value = raw[key].to(device, non_blocking=True)
        if channels_last and key in {"clean", "current", "goal"}:
            value = value.contiguous(memory_format=torch.channels_last)
        moved[key] = value
    if "target_noise" in raw:
        moved["target_noise"] = raw["target_noise"].to(device, non_blocking=True)
    return moved


def display_image(images: Tensor) -> Tensor:
    """Detach normalized model images and map [-1, 1] into [0, 1]."""
    return ((images.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) / 2.0)


def log_validation_images(
    writer: SummaryWriter,
    batch: dict[str, Tensor],
    predicted: Tensor,
    noise_energy: Tensor | None,
    epoch: int,
    count: int,
) -> None:
    """Log aligned validation inputs, targets, predictions, and energy maps."""
    count = min(count, batch["clean"].shape[0])
    if count < 1:
        return

    images = {
        "clean": display_image(batch["clean"][:count]),
        "goal": display_image(batch["goal"][:count]),
        "current": display_image(batch["current"][:count]),
        "target_mean": display_image(batch["target_cloud"][:count].mean(1)),
        "prediction_mean": display_image(predicted[:count].mean(1)),
        "prediction_sample_0": display_image(predicted[:count, 0]),
    }
    if noise_energy is not None:
        energy = noise_energy[:count].detach().float().cpu()
        if energy.ndim == 3:
            energy = energy[:, None]
        flat = energy.flatten(1)
        minimum = flat.min(dim=1).values[:, None, None, None]
        maximum = flat.max(dim=1).values[:, None, None, None]
        normalized = (energy - minimum) / (maximum - minimum).clamp_min(1e-8)
        images["noise_energy_normalized"] = normalized.repeat(1, 3, 1, 1)
        writer.add_histogram(
            "validation/noise_energy_distribution",
            energy,
            global_step=epoch,
        )

    # Each example occupies one row; columns follow the order in `images`.
    comparison = torch.stack(list(images.values()), dim=1).flatten(0, 1)
    writer.add_image(
        "validation/comparison",
        make_grid(comparison, nrow=len(images), padding=2),
        global_step=epoch,
    )
    for name, value in images.items():
        writer.add_images(f"validation/{name}", value, global_step=epoch)
    writer.add_text(
        "validation/comparison_columns",
        " | ".join(images),
        global_step=epoch,
    )


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    spatial_criterion: SpatialNoiseCrossEntropyLoss | None,
    spatial_ce_weight: float,
    paired: bool,
    device: torch.device,
    samples: int,
    channels_last: bool,
    amp_enabled: bool,
    max_batches: int,
    writer: SummaryWriter | None = None,
    epoch: int = 0,
    validation_images: int = 0,
) -> dict[str, float]:
    model.eval()
    # loss, input/output PSNR, mean energy, count, cloud/spatial loss,
    # and per-image spatial-energy p50/p95/p99/max.
    totals = torch.zeros(24, device=device)
    energy_quantile_levels = torch.tensor([0.50, 0.95, 0.99], device=device)
    for batch_index, raw in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        batch = move_batch(raw, device, channels_last)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            model_output = model(
                batch["current"],
                batch["goal"],
                samples=samples,
                noise=batch["target_noise"] if paired else None,
                return_noise_variance=True,
                return_noise_energy=(
                    spatial_criterion is not None
                    or getattr(model, "architecture", "") == "fullres_axial"
                ),
            )
            if not isinstance(model_output, tuple):
                raise RuntimeError("model did not return its noise variance")
            predicted, noise_variance = model_output[:2]
            cloud_loss, mean_loss, deviation_loss, variance_loss = cloud_loss_components(
                criterion,
                predicted, batch["target_cloud"], batch["current"]
            )
            spatial_loss = cloud_loss.new_zeros(())
            if spatial_criterion is not None:
                spatial_loss = spatial_criterion(
                    model_output[2], batch["target_cloud"], batch["current"]
                )
            loss = cloud_loss + spatial_ce_weight * spatial_loss
        if batch_index == 0 and writer is not None and validation_images > 0:
            log_validation_images(
                writer,
                batch,
                predicted,
                model_output[2] if len(model_output) == 3 else None,
                epoch,
                validation_images,
            )
        count = batch["clean"].shape[0]
        output_mean = predicted.float().mean(1)
        predicted_diversity = cloud_diversity(predicted)
        target_diversity = cloud_diversity(batch["target_cloud"])
        clean_endpoint = batch["answer_level"].eq(0)
        transition_endpoint = ~clean_endpoint
        clean_count = clean_endpoint.sum()
        transition_count = transition_endpoint.sum()
        energy_per_image = noise_variance.detach().float().flatten()
        if len(model_output) == 3:
            energy_flat = model_output[2].detach().float().flatten(1)
            energy_quantiles = torch.quantile(
                energy_flat,
                energy_quantile_levels,
                dim=1,
            ).sum(dim=1)
            energy_max = energy_flat.max(dim=1).values.sum()
        else:
            energy_quantiles = torch.zeros(3, device=device)
            energy_max = torch.zeros((), device=device)
        totals += torch.stack(
            [
                loss.float() * count,
                psnr(batch["current"].float(), batch["clean"].float()).sum(),
                psnr(output_mean, batch["clean"].float()).sum(),
                noise_variance.float().sum(),
                torch.tensor(float(count), device=device),
                cloud_loss.float() * count,
                spatial_loss.float() * count,
                energy_quantiles[0],
                energy_quantiles[1],
                energy_quantiles[2],
                energy_max,
                mean_loss.float() * count,
                deviation_loss.float() * count,
                variance_loss.float() * count,
                predicted_diversity.sum(),
                target_diversity.sum(),
                predicted_diversity[clean_endpoint].sum(),
                target_diversity[clean_endpoint].sum(),
                energy_per_image[clean_endpoint].sum(),
                clean_count.float(),
                predicted_diversity[transition_endpoint].sum(),
                target_diversity[transition_endpoint].sum(),
                energy_per_image[transition_endpoint].sum(),
                transition_count.float(),
            ],
        )
    model.train()
    denominator = max(totals[4].item(), 1.0)
    clean_denominator = max(totals[19].item(), 1.0)
    transition_denominator = max(totals[23].item(), 1.0)
    return {
        "val_loss": totals[0].item() / denominator,
        "val_current_psnr": totals[1].item() / denominator,
        "val_output_psnr": totals[2].item() / denominator,
        "val_noise_variance": totals[3].item() / denominator,
        "val_cloud_loss": totals[5].item() / denominator,
        "val_spatial_ce": totals[6].item() / denominator,
        "val_noise_energy_p50": totals[7].item() / denominator,
        "val_noise_energy_p95": totals[8].item() / denominator,
        "val_noise_energy_p99": totals[9].item() / denominator,
        "val_noise_energy_max": totals[10].item() / denominator,
        "val_paired_mean_loss": totals[11].item() / denominator,
        "val_paired_deviation_loss": totals[12].item() / denominator,
        "val_paired_variance_loss": totals[13].item() / denominator,
        "val_predicted_diversity": totals[14].item() / denominator,
        "val_target_diversity": totals[15].item() / denominator,
        "val_clean_predicted_diversity": totals[16].item() / clean_denominator,
        "val_clean_target_diversity": totals[17].item() / clean_denominator,
        "val_clean_noise_variance": totals[18].item() / clean_denominator,
        "val_transition_predicted_diversity": totals[20].item()
        / transition_denominator,
        "val_transition_target_diversity": totals[21].item()
        / transition_denominator,
        "val_transition_noise_variance": totals[22].item()
        / transition_denominator,
    }


def save_checkpoint(
    path: Path,
    model: StochasticImageBridge,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    history: list[dict[str, float]],
    config: dict[str, Any],
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "history": history,
            "config": config,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size, device = distributed_setup(args.device)
    is_main = rank == 0
    config = load_config(args.config)
    if args.epochs is not None:
        config.train.epochs = args.epochs
    seed_everything(config.train.seed, rank)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = config.train.tf32
        torch.backends.cudnn.allow_tf32 = config.train.tf32
    torch.set_float32_matmul_precision("high")

    output = Path(args.output).resolve()
    if is_main:
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.json").write_text(
            json.dumps(config.to_dict(), indent=2), encoding="utf-8"
        )
    if world_size > 1:
        dist.barrier()

    train_dataset = PreparedBridgeDataset(
        args.train_prepared, cache_shards=config.data.prepared_cache_shards
    )
    val_dataset = PreparedBridgeDataset(
        args.val_prepared, cache_shards=config.data.prepared_cache_shards
    )
    paired = is_paired_cloud_loss(config.loss.name)
    for dataset in (train_dataset, val_dataset):
        if paired and not dataset.has_target_noise:
            raise RuntimeError("Gaussian paired training requires stored target_noise")
        if int(dataset.manifest["target_samples"]) != config.loss.samples:
            raise RuntimeError("prepared target_samples does not match config")

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=config.train.seed,
        drop_last=True,
    )
    loader_kwargs: dict[str, Any] = {
        "batch_size": config.data.batch_size,
        "num_workers": config.data.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": config.data.workers > 0,
    }
    if config.data.workers > 0:
        loader_kwargs["prefetch_factor"] = config.data.prefetch_factor
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = (
        DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_kwargs)
        if is_main
        else None
    )

    model = StochasticImageBridge(**config.model.__dict__).to(device)
    channels_last = (
        config.train.channels_last
        and config.model.architecture == "pyramid"
        and config.model.encoder_type == "cnn"
    )
    if channels_last:
        model.to(memory_format=torch.channels_last)
    train_model: nn.Module = model
    if world_size > 1:
        ddp_device = (
            {"device_ids": [local_rank], "output_device": local_rank}
            if device.type == "cuda"
            else {}
        )
        train_model = DDP(
            model,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            static_graph=True,
            **ddp_device,
        )
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
    spatial_criterion = (
        SpatialNoiseCrossEntropyLoss(
            highpass=config.loss.spatial_ce_highpass,
            kernel_size=config.loss.spatial_ce_kernel_size,
        ).to(device)
        if config.loss.spatial_ce_weight > 0.0
        else None
    )
    if spatial_criterion is not None and config.model.architecture != "fullres_axial":
        raise ValueError(
            "loss.spatial_ce_weight requires model.architecture=fullres_axial"
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
        fused=config.train.fused_optimizer and device.type == "cuda",
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(config.train.epochs, 1)
    )
    amp_enabled = config.train.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
        init_scale=config.train.amp_init_scale,
        growth_interval=config.train.amp_growth_interval,
    )
    history: list[dict[str, float]] = []
    start_epoch = 0
    best_val = math.inf
    resume = args.resume or config.train.resume
    if resume and args.init_checkpoint:
        raise ValueError("--resume and --init-checkpoint are mutually exclusive")
    if args.init_checkpoint:
        state = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        source_model_config = state.get("config", {}).get("model", {})
        source_energy_parameterization = source_model_config.get(
            "noise_energy_parameterization", "bounded"
        )
        if args.reset_energy_head and model.architecture == "fullres_axial":
            model.fullres.reset_energy_head(config.model.noise_variance_init)
            if is_main:
                print(
                    "reset energy head by request; "
                    f"initial energy={config.model.noise_variance_init:g}"
                )
        elif (
            model.architecture == "fullres_axial"
            and source_energy_parameterization
            != config.model.noise_energy_parameterization
        ):
            model.fullres.reset_energy_head(config.model.noise_variance_init)
            if is_main:
                print(
                    "reset energy head after parameterization change: "
                    f"{source_energy_parameterization} -> "
                    f"{config.model.noise_energy_parameterization}; "
                    f"initial energy={config.model.noise_variance_init:g}"
                )
        if is_main:
            print(f"initialized model weights from {args.init_checkpoint}")
    if resume:
        state = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state.get("scaler", {}))
        history = state.get("history", [])
        start_epoch = int(state.get("epoch", 0))
        best_val = min((row["val_loss"] for row in history), default=math.inf)

    writer = None
    if is_main:
        tensorboard_dir = (
            Path(args.tensorboard_dir).resolve()
            if args.tensorboard_dir
            else output / "tensorboard"
        )
        writer = SummaryWriter(log_dir=tensorboard_dir)
        writer.add_text("run/config", json.dumps(config.to_dict(), indent=2), 0)

    for epoch in range(start_epoch, config.train.epochs):
        train_sampler.set_epoch(epoch)
        train_model.train()
        totals = torch.zeros(12, device=device)
        progress = tqdm(
            train_loader,
            desc=f"epoch {epoch + 1}/{config.train.epochs}",
            disable=not is_main,
        )
        for batch_index, raw in enumerate(progress):
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
            optimizer.zero_grad(set_to_none=True)
            batch = move_batch(raw, device, channels_last)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                model_output = train_model(
                    batch["current"],
                    batch["goal"],
                    samples=config.loss.samples,
                    noise=batch["target_noise"] if paired else None,
                    return_noise_variance=True,
                    return_noise_energy=spatial_criterion is not None,
                )
                if not isinstance(model_output, tuple):
                    raise RuntimeError("model did not return its noise variance")
                predicted, noise_variance = model_output[:2]
                cloud_loss, mean_loss, deviation_loss, variance_loss = cloud_loss_components(
                    criterion,
                    predicted, batch["target_cloud"], batch["current"]
                )
                spatial_loss = cloud_loss.new_zeros(())
                if spatial_criterion is not None:
                    spatial_loss = spatial_criterion(
                        model_output[2], batch["target_cloud"], batch["current"]
                    )
                loss = cloud_loss + config.loss.spatial_ce_weight * spatial_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.train.grad_clip
            )
            scaler.step(optimizer)
            scaler.update()

            count = batch["clean"].shape[0]
            predicted_diversity = cloud_diversity(predicted)
            target_diversity = cloud_diversity(batch["target_cloud"])
            totals += torch.stack(
                [
                    loss.detach().float() * count,
                    psnr(
                        predicted.detach().float().mean(1), batch["clean"].float()
                    ).sum(),
                    grad_norm.detach().float(),
                    noise_variance.detach().float().sum(),
                    torch.tensor(float(count), device=device),
                    cloud_loss.detach().float() * count,
                    spatial_loss.detach().float() * count,
                    mean_loss.detach().float() * count,
                    deviation_loss.detach().float() * count,
                    variance_loss.detach().float() * count,
                    predicted_diversity.sum(),
                    target_diversity.sum(),
                ]
            )
            if is_main and (batch_index + 1) % config.train.log_every == 0:
                progress.set_postfix(loss=f"{loss.item():.4f}")
                assert writer is not None
                global_step = epoch * len(train_loader) + batch_index + 1
                writer.add_scalar("batch/train_loss", loss.item(), global_step)
                writer.add_scalar(
                    "batch/train_cloud_loss", cloud_loss.item(), global_step
                )
                writer.add_scalar(
                    "batch/train_spatial_ce", spatial_loss.item(), global_step
                )
                writer.add_scalar("batch/paired_mean_loss", mean_loss.item(), global_step)
                writer.add_scalar(
                    "batch/paired_deviation_loss", deviation_loss.item(), global_step
                )
                writer.add_scalar(
                    "batch/paired_variance_loss", variance_loss.item(), global_step
                )
                writer.add_scalar(
                    "batch/predicted_diversity",
                    predicted_diversity.mean().item(),
                    global_step,
                )
                writer.add_scalar(
                    "batch/target_diversity",
                    target_diversity.mean().item(),
                    global_step,
                )
                writer.add_scalar(
                    "batch/noise_variance", noise_variance.float().mean().item(), global_step
                )
                writer.add_scalar("batch/gradient_norm", grad_norm.item(), global_step)
                writer.add_scalar(
                    "batch/learning_rate", optimizer.param_groups[0]["lr"], global_step
                )

        totals = reduce_totals(totals, world_size)
        count = max(totals[4].item(), 1.0)
        metrics = {
            "epoch": float(epoch + 1),
            "train_loss": totals[0].item() / count,
            "train_output_psnr": totals[1].item() / count,
            "gradient_norm": totals[2].item() / max(len(train_loader) * world_size, 1),
            "train_noise_variance": totals[3].item() / count,
            "train_cloud_loss": totals[5].item() / count,
            "train_spatial_ce": totals[6].item() / count,
            "train_paired_mean_loss": totals[7].item() / count,
            "train_paired_deviation_loss": totals[8].item() / count,
            "train_paired_variance_loss": totals[9].item() / count,
            "train_predicted_diversity": totals[10].item() / count,
            "train_target_diversity": totals[11].item() / count,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        # Advance before checkpointing so a resumed run uses the exact next-
        # epoch learning rate rather than repeating the previous scheduler step.
        scheduler.step()

        if world_size > 1:
            dist.barrier()
        if is_main:
            assert val_loader is not None
            metrics.update(
                validate(
                    model,
                    val_loader,
                    criterion,
                    spatial_criterion,
                    config.loss.spatial_ce_weight,
                    paired,
                    device,
                    config.loss.samples,
                    channels_last,
                    amp_enabled,
                    args.max_val_batches,
                    writer,
                    epoch + 1,
                    args.validation_images,
                )
            )
            history.append(metrics)
            assert writer is not None
            for name, value in metrics.items():
                if name != "epoch":
                    writer.add_scalar(f"epoch/{name}", value, epoch + 1)
            writer.flush()
            (output / "history.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8"
            )
            save_checkpoint(
                output / "latest.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch + 1,
                history,
                config.to_dict(),
            )
            if metrics["val_loss"] < best_val:
                best_val = metrics["val_loss"]
                save_checkpoint(
                    output / "best.pt",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch + 1,
                    history,
                    config.to_dict(),
                )
            print(json.dumps(metrics, indent=2))
        if world_size > 1:
            dist.barrier()

    if is_main:
        save_checkpoint(
            output / "final.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            config.train.epochs,
            history,
            config.to_dict(),
        )
        assert writer is not None
        writer.close()
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
