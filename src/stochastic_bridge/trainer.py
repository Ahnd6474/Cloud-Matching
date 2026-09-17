from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm

from .config import ExperimentConfig
from .corruptions import GoalDetailCorruptor
from .data import BridgeBatch, build_bridge_batch
from .datasets import ImageDirectoryDataset, SyntheticImageDataset
from .losses import build_cloud_loss
from .model import StochasticImageBridge
from .noise import CorruptionMixture
from .prepared import PreparedBridgeDataset, ShardBatchSampler
from .schedule import VPNoiseSchedule


class Trainer:
    def __init__(self, config: ExperimentConfig, device: str | None = None) -> None:
        self.config = config
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._seed_everything(config.train.seed)
        self._configure_backend()
        self.output_dir = Path(config.train.output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "previews").mkdir(exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        (self.output_dir / "config.json").write_text(
            json.dumps(config.to_dict(), indent=2), encoding="utf-8"
        )

        self.loader = self._build_loader()
        self.using_prepared_data = isinstance(
            self.loader.dataset, PreparedBridgeDataset
        )
        self.schedule = VPNoiseSchedule(
            steps=config.schedule.steps,
            beta_start=config.schedule.beta_start,
            beta_end=config.schedule.beta_end,
        ).to(self.device)
        self.corruption_mixture = (
            None if self.using_prepared_data else self._build_corruption_mixture()
        )
        self.goal_corruptor = (
            None if self.using_prepared_data else self._build_goal_corruptor()
        )
        self._validate_loss_data_compatibility()
        self.model = StochasticImageBridge(**config.model.__dict__).to(self.device)
        self.use_channels_last = (
            config.train.channels_last and config.model.encoder_type.lower() == "cnn"
        )
        if self.use_channels_last:
            self.model.to(memory_format=torch.channels_last)
        self.forward_model = self._compile_model(self.model)
        self.criterion = build_cloud_loss(
            config.loss.name, blur=config.loss.sinkhorn_blur
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
            fused=config.train.fused_optimizer and self.device.type == "cuda",
        )
        amp_enabled = config.train.amp and self.device.type == "cuda"
        self.amp_dtype = self._amp_dtype(config.train.amp_dtype)
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=amp_enabled and self.amp_dtype == torch.float16
        )
        self.amp_enabled = amp_enabled
        self.writer = SummaryWriter(self.output_dir / "tensorboard")
        self.global_step = 0
        self.start_epoch = 0
        if config.train.resume:
            self.load_checkpoint(config.train.resume)

    def train(self) -> None:
        try:
            for epoch in range(self.start_epoch, self.config.train.epochs):
                self._train_epoch(epoch)
                self.save_checkpoint(epoch + 1, name=f"epoch-{epoch + 1:04d}.pt")
        finally:
            self.writer.close()

    def _train_epoch(self, epoch: int) -> None:
        self.model.train()
        if isinstance(self.loader.batch_sampler, ShardBatchSampler):
            self.loader.batch_sampler.set_epoch(epoch)
        progress = tqdm(self.loader, desc=f"epoch {epoch + 1}/{self.config.train.epochs}")
        for raw_batch in progress:
            # Release the previous gradient buffers before moving the next
            # target cloud to the accelerator, reducing peak device memory.
            self.optimizer.zero_grad(set_to_none=True)
            if self.using_prepared_data:
                bridge = self._prepared_bridge(raw_batch)
            else:
                clean = raw_batch.to(self.device, non_blocking=True)
                bridge = build_bridge_batch(
                    clean=clean,
                    schedule=self.schedule,
                    target_samples=self.config.loss.samples,
                    answer_jump=self.config.schedule.answer_jump,
                    goal_corruptor=self.goal_corruptor,
                    corruption_mixture=self.corruption_mixture,
                )
            paired = self.config.loss.name.lower() == "paired"
            model_noise = bridge.target_noise if paired else None
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.amp_enabled,
            ):
                model_output = self.forward_model(
                    bridge.current,
                    bridge.goal,
                    samples=self.config.loss.samples,
                    noise=model_noise,
                    return_noise_variance=True,
                )
                if not isinstance(model_output, tuple):
                    raise RuntimeError("model did not return its noise variance")
                predicted, noise_variance = model_output
                loss = self.criterion(predicted, bridge.target_cloud, bridge.current)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.train.grad_clip
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.global_step += 1
            if self.global_step % self.config.train.log_every == 0:
                loss_value = loss.detach().item()
                progress.set_postfix(loss=f"{loss_value:.4f}")
                self.writer.add_scalar("train/loss", loss_value, self.global_step)
                self.writer.add_scalar(
                    "train/gradient_norm", float(grad_norm), self.global_step
                )
                self.writer.add_scalar(
                    "train/current_level", bridge.current_level.float().mean(), self.global_step
                )
                self.writer.add_scalar(
                    "train/goal_level", bridge.goal_level.float().mean(), self.global_step
                )
                self.writer.add_scalar(
                    "train/noise_variance", noise_variance.detach().float().mean(), self.global_step
                )
            if self.global_step % self.config.train.preview_every == 0:
                self._save_preview(bridge, predicted)
            if self.global_step % self.config.train.save_every == 0:
                self.save_checkpoint(epoch, name=f"step-{self.global_step:08d}.pt")

    def _build_loader(self) -> DataLoader:
        data = self.config.data
        if data.prepared_root:
            dataset = PreparedBridgeDataset(
                data.prepared_root, cache_shards=data.prepared_cache_shards
            )
            expected_samples = int(dataset.manifest["target_samples"])
            if expected_samples != self.config.loss.samples:
                raise ValueError(
                    "prepared dataset target_samples mismatch: "
                    f"dataset={expected_samples}, config={self.config.loss.samples}"
                )
            expected_size = int(dataset.manifest["image_size"])
            if expected_size != data.image_size:
                raise ValueError(
                    "prepared dataset image_size mismatch: "
                    f"dataset={expected_size}, config={data.image_size}"
                )
        else:
            dataset = (
                SyntheticImageDataset(data.synthetic_length, data.image_size)
                if data.synthetic
                else ImageDirectoryDataset(
                    data.root,
                    data.image_size,
                    random_crop=data.random_crop,
                    horizontal_flip=data.horizontal_flip,
                )
            )
        common: dict[str, Any] = {
            "dataset": dataset,
            "num_workers": data.workers,
            "pin_memory": self.device.type == "cuda",
            "persistent_workers": data.workers > 0,
        }
        if data.workers > 0:
            common["prefetch_factor"] = data.prefetch_factor
        if isinstance(dataset, PreparedBridgeDataset):
            return DataLoader(
                **common,
                batch_sampler=ShardBatchSampler(
                    dataset,
                    batch_size=data.batch_size,
                    drop_last=False,
                    seed=self.config.train.seed,
                ),
            )
        return DataLoader(
            **common,
            batch_size=data.batch_size,
            shuffle=True,
            drop_last=len(dataset) >= data.batch_size,
        )

    def _validate_loss_data_compatibility(self) -> None:
        if self.config.loss.name.lower() != "paired":
            return
        if self.using_prepared_data:
            dataset = self.loader.dataset
            assert isinstance(dataset, PreparedBridgeDataset)
            if not dataset.has_target_noise:
                raise ValueError(
                    "paired loss requires an analytic VP prepared dataset with target noise"
                )
        elif self.corruption_mixture is not None:
            raise ValueError(
                "paired loss requires corruption.enabled=false; mixed corruption "
                "clouds do not have sample-wise target/model-noise correspondence"
            )

    def _prepared_bridge(self, raw: dict[str, Tensor]) -> BridgeBatch:
        def move(name: str) -> Tensor:
            value = raw[name].to(self.device, non_blocking=True)
            if self.use_channels_last and value.ndim == 4 and name in {"current", "goal"}:
                value = value.contiguous(memory_format=torch.channels_last)
            return value

        return BridgeBatch(
            # Clean images are only needed by occasional previews. Keeping them
            # on CPU removes an otherwise unused host-to-device transfer.
            clean=raw["clean"],
            current=move("current"),
            goal=move("goal"),
            target_cloud=move("target_cloud"),
            target_noise=(
                move("target_noise") if "target_noise" in raw else None
            ),
            current_level=move("current_level"),
            goal_level=move("goal_level"),
            answer_level=move("answer_level"),
            corruption_types=None,
        )

    def _build_goal_corruptor(self) -> GoalDetailCorruptor | None:
        goal = self.config.goal_corruption
        if not goal.enabled:
            return None
        values = goal.__dict__.copy()
        values.pop("enabled")
        return GoalDetailCorruptor(**values).to(self.device)

    def _build_corruption_mixture(self) -> CorruptionMixture | None:
        corruption = self.config.corruption
        if not corruption.enabled:
            return None
        return CorruptionMixture(
            schedule=self.schedule,
            weights=corruption.weights,
            spatial_floor=corruption.spatial_floor,
            student_t_df=corruption.student_t_df,
        ).to(self.device)

    @torch.no_grad()
    def _save_preview(self, bridge: BridgeBatch, predicted: Tensor) -> None:
        count = min(4, bridge.clean.shape[0])
        clean = bridge.clean[:count].to(predicted.device, non_blocking=True)
        rows = torch.cat(
            [
                clean,
                bridge.current[:count],
                bridge.goal[:count],
                bridge.target_cloud[:count, 0],
                predicted[:count, 0],
            ],
            dim=0,
        )
        rows = (rows.clamp(-1.0, 1.0) + 1.0) / 2.0
        save_image(
            rows,
            self.output_dir / "previews" / f"step-{self.global_step:08d}.png",
            nrow=count,
        )

    def save_checkpoint(self, epoch: int, name: str) -> Path:
        destination = self.output_dir / "checkpoints" / name
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict(),
                "epoch": epoch,
                "global_step": self.global_step,
                "config": self.config.to_dict(),
            },
            destination,
        )
        latest = self.output_dir / "checkpoints" / "latest.pt"
        shutil.copyfile(destination, latest)
        return destination

    def load_checkpoint(self, path: str | Path) -> None:
        state: dict[str, Any] = torch.load(
            Path(path), map_location=self.device, weights_only=False
        )
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scaler.load_state_dict(state.get("scaler", {}))
        self.start_epoch = int(state.get("epoch", 0))
        self.global_step = int(state.get("global_step", 0))

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _configure_backend(self) -> None:
        torch.set_float32_matmul_precision("high" if self.config.train.tf32 else "highest")
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = self.config.train.tf32
            torch.backends.cudnn.allow_tf32 = self.config.train.tf32
            torch.backends.cudnn.benchmark = True

    def _compile_model(self, model: StochasticImageBridge) -> torch.nn.Module:
        if not self.config.train.compile:
            return model
        return torch.compile(model, mode=self.config.train.compile_mode)

    @staticmethod
    def _amp_dtype(name: str) -> torch.dtype:
        normalized = name.strip().lower()
        if normalized in {"float16", "fp16"}:
            return torch.float16
        if normalized in {"bfloat16", "bf16"}:
            return torch.bfloat16
        raise ValueError("train.amp_dtype must be float16 or bfloat16")
