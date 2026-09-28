from __future__ import annotations

import argparse
import json
import math
from contextlib import nullcontext
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F

from stochastic_bridge.config import config_from_dict
from stochastic_bridge.model import StochasticImageBridge
from stochastic_bridge.prepared import PreparedBridgeDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the legacy recurrent sampler with correction-only and "
            "correction-then-external-noise rollouts."
        )
    )
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--sample-stride", type=int, default=17)
    parser.add_argument("--paths", type=int, default=2)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument(
        "--noise-starts",
        type=float,
        nargs="+",
        default=[0.02, 0.05, 0.10],
        help="External image-space standard deviations in the [-1, 1] domain.",
    )
    parser.add_argument("--noise-end", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260921)
    return parser.parse_args()


def psnr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mse = (x.float() - y.float()).square().flatten(1).mean(1).clamp_min(1e-10)
    return 10 * torch.log10(4.0 / mse)


def ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x, y = (x.float() + 1) / 2, (y.float() + 1) / 2
    mu_x, mu_y = F.avg_pool2d(x, 7, 1, 3), F.avg_pool2d(y, 7, 1, 3)
    var_x = F.avg_pool2d(x * x, 7, 1, 3) - mu_x.square()
    var_y = F.avg_pool2d(y * y, 7, 1, 3) - mu_y.square()
    covariance = F.avg_pool2d(x * y, 7, 1, 3) - mu_x * mu_y
    numerator = (2 * mu_x * mu_y + 0.01**2) * (2 * covariance + 0.03**2)
    denominator = (
        (mu_x.square() + mu_y.square() + 0.01**2)
        * (var_x + var_y + 0.03**2)
    )
    return (numerator / denominator.clamp_min(1e-8)).flatten(1).mean(1)


def laplacian(image: torch.Tensor) -> torch.Tensor:
    kernel = image.new_tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    kernel = kernel.reshape(1, 1, 3, 3).expand(image.shape[1], 1, 3, 3)
    return F.conv2d(image.float(), kernel.float(), padding=1, groups=image.shape[1])


def correlation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.flatten(1) - x.flatten(1).mean(1, keepdim=True)
    y = y.flatten(1) - y.flatten(1).mean(1, keepdim=True)
    return F.cosine_similarity(x, y, dim=1)


def cosine_noise_schedule(
    start: float,
    end: float,
    injection_index: int,
    injection_count: int,
) -> float:
    """Return an annealed std for noise inserted between correction steps."""
    if injection_count <= 1:
        return float(start)
    fraction = injection_index / (injection_count - 1)
    weight = 0.5 * (1.0 + math.cos(math.pi * fraction))
    return float(end + (start - end) * weight)


def display_image(value: torch.Tensor):
    return (
        ((value.detach().float().cpu().clamp(-1, 1) + 1) / 2)
        .permute(1, 2, 0)
        .numpy()
    )


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def mode_label(mode: str, noise_start: float | None) -> str:
    if mode == "external":
        assert noise_start is not None
        return f"external_{noise_start:.3f}"
    return mode


def measure(
    value: torch.Tensor,
    target: torch.Tensor,
    sample_count: int,
    paths: int,
) -> dict[str, float]:
    value_laplacian = laplacian(value)
    target_laplacian = laplacian(target)
    grouped = value.reshape(sample_count, paths, *value.shape[1:])
    return {
        "psnr": psnr(value, target).mean().item(),
        "ssim": ssim(value, target).mean().item(),
        "laplacian_correlation": correlation(
            value_laplacian, target_laplacian
        ).mean().item(),
        "laplacian_l1": F.l1_loss(value_laplacian, target_laplacian).item(),
        "sharpness": value_laplacian.abs().mean().item(),
        "trajectory_diversity": grouped.std(1, unbiased=False).mean().item(),
    }


def main() -> None:
    args = parse_args()
    if args.samples < 1 or args.paths < 1 or args.steps < 1:
        raise ValueError("samples, paths, and steps must be positive")
    if args.noise_end < 0 or any(value < 0 for value in args.noise_starts):
        raise ValueError("noise standard deviations must be non-negative")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)

    dataset = PreparedBridgeDataset(args.validation)
    indices = [
        min(index * args.sample_stride, len(dataset) - 1)
        for index in range(args.samples)
    ]
    records = [dataset[index] for index in indices]
    clean_base = torch.stack([record["clean"] for record in records]).to(device)
    current_base = torch.stack([record["current"] for record in records]).to(device)
    goal_base = torch.stack([record["goal"] for record in records]).to(device)

    def expand_paths(value: torch.Tensor) -> torch.Tensor:
        return (
            value[:, None]
            .expand(-1, args.paths, -1, -1, -1)
            .reshape(args.samples * args.paths, *value.shape[1:])
        )

    target = expand_paths(clean_base)
    initial = expand_paths(current_base)
    goal = expand_paths(goal_base)
    all_summaries: dict[str, object] = {
        "device": str(device),
        "sample_indices": indices,
        "steps": args.steps,
        "paths": args.paths,
        "noise_domain": "normalized image [-1, 1]",
        "checkpoints": {},
    }

    modes: list[tuple[str, float | None]] = [
        ("correction_only", None),
        ("legacy_internal_persistent", None),
    ]
    modes.extend(("external", value) for value in args.noise_starts)

    for checkpoint_path in args.checkpoints:
        checkpoint_path = checkpoint_path.expanduser().resolve()
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        config = config_from_dict(checkpoint["config"])
        model = StochasticImageBridge(**config.model.__dict__).to(device).eval()
        model.load_state_dict(checkpoint["model"])

        checkpoint_name = checkpoint_path.parent.name
        if checkpoint_name in all_summaries["checkpoints"]:
            checkpoint_name = f"{checkpoint_name}_{checkpoint_path.stem}"
        checkpoint_output = args.output / checkpoint_name
        checkpoint_output.mkdir(parents=True, exist_ok=True)
        zero_internal = torch.zeros(
            target.shape[0], 1, *target.shape[1:], device=device
        )
        generator = torch.Generator(device=device).manual_seed(args.seed)
        persistent_internal = torch.randn(
            zero_internal.shape, device=device, generator=generator
        )
        rows: list[dict[str, float | int | str]] = []
        final_states: dict[str, torch.Tensor] = {}
        selected_states: dict[str, dict[int, torch.Tensor]] = {}
        selected_steps = sorted(
            step
            for step in set([0, 1, 2, 4, args.steps])
            if step <= args.steps
        )

        with torch.inference_mode():
            for mode, noise_start in modes:
                label = mode_label(mode, noise_start)
                current = initial.clone()
                selected_states[label] = {0: current.detach().cpu()}
                initial_metrics = measure(
                    current, target, args.samples, args.paths
                )
                rows.append(
                    {
                        "mode": label,
                        "step": 0,
                        "external_noise_std": 0.0,
                        **initial_metrics,
                    }
                )

                for step in range(1, args.steps + 1):
                    internal_noise = (
                        persistent_internal
                        if mode == "legacy_internal_persistent"
                        else zero_internal
                    )
                    with autocast_context(device):
                        corrected = model(
                            current,
                            goal,
                            samples=1,
                            noise=internal_noise,
                        )[:, 0]
                    corrected = corrected.float().clamp(-1, 1)

                    sigma = 0.0
                    if mode == "external" and step < args.steps:
                        assert noise_start is not None
                        sigma = cosine_noise_schedule(
                            noise_start,
                            args.noise_end,
                            injection_index=step - 1,
                            injection_count=max(args.steps - 1, 1),
                        )
                    step_metrics = measure(
                        corrected, target, args.samples, args.paths
                    )
                    rows.append(
                        {
                            "mode": label,
                            "step": step,
                            "external_noise_std": sigma,
                            **step_metrics,
                        }
                    )
                    if step in selected_steps:
                        selected_states[label][step] = corrected.detach().cpu()

                    if mode == "external" and step < args.steps:
                        external_noise = torch.randn(
                            corrected.shape, device=device, generator=generator
                        )
                        current = (corrected + sigma * external_noise).clamp(-1, 1)
                    else:
                        current = corrected

                final_states[label] = corrected.detach().cpu()

        metrics = pd.DataFrame(rows)
        metrics.to_csv(checkpoint_output / "metrics.csv", index=False)
        checkpoint_summary: dict[str, object] = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "modes": {},
        }
        for label, frame in metrics.groupby("mode", sort=False):
            best = frame.loc[frame.psnr.idxmax()]
            final = frame.iloc[-1]
            checkpoint_summary["modes"][label] = {
                "best_step": int(best.step),
                "best_psnr": float(best.psnr),
                "final_psnr": float(final.psnr),
                "final_ssim": float(final.ssim),
                "final_laplacian_correlation": float(
                    final.laplacian_correlation
                ),
                "final_laplacian_l1": float(final.laplacian_l1),
                "final_sharpness": float(final.sharpness),
                "final_trajectory_diversity": float(
                    final.trajectory_diversity
                ),
            }
        all_summaries["checkpoints"][checkpoint_name] = checkpoint_summary

        labels = [mode_label(mode, value) for mode, value in modes]
        figure, axes = plt.subplots(
            args.samples,
            3 + len(labels),
            figsize=(2.35 * (3 + len(labels)), 2.4 * args.samples),
            squeeze=False,
        )
        for sample_index in range(args.samples):
            values = [
                clean_base[sample_index].cpu(),
                goal_base[sample_index].cpu(),
                current_base[sample_index].cpu(),
            ]
            values.extend(
                final_states[label][sample_index * args.paths] for label in labels
            )
            titles = ["clean", "fixed goal", "initial current", *labels]
            for column, (value, title) in enumerate(
                zip(values, titles, strict=True)
            ):
                axes[sample_index, column].imshow(display_image(value))
                axes[sample_index, column].axis("off")
                if sample_index == 0:
                    axes[sample_index, column].set_title(title)
        figure.suptitle(f"External-noise rollout: {checkpoint_name}", y=1.01)
        figure.tight_layout()
        figure.savefig(
            checkpoint_output / "final_comparison.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(figure)

        early_step = min(2, args.steps)
        figure, axes = plt.subplots(
            args.samples,
            3 + len(labels),
            figsize=(2.35 * (3 + len(labels)), 2.4 * args.samples),
            squeeze=False,
        )
        for sample_index in range(args.samples):
            values = [
                clean_base[sample_index].cpu(),
                goal_base[sample_index].cpu(),
                current_base[sample_index].cpu(),
            ]
            values.extend(
                selected_states[label][early_step][sample_index * args.paths]
                for label in labels
            )
            titles = ["clean", "fixed goal", "initial current", *labels]
            for column, (value, title) in enumerate(
                zip(values, titles, strict=True)
            ):
                axes[sample_index, column].imshow(display_image(value))
                axes[sample_index, column].axis("off")
                if sample_index == 0:
                    axes[sample_index, column].set_title(title)
        figure.suptitle(
            f"Early correction step {early_step}: {checkpoint_name}", y=1.01
        )
        figure.tight_layout()
        figure.savefig(
            checkpoint_output / f"step_{early_step}_comparison.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(figure)

        figure, axes = plt.subplots(1, 4, figsize=(19, 4.2))
        for label, frame in metrics.groupby("mode", sort=False):
            axes[0].plot(frame.step, frame.psnr, marker="o", label=label)
            axes[1].plot(frame.step, frame.ssim, marker="o", label=label)
            axes[2].plot(
                frame.step,
                frame.laplacian_correlation,
                marker="o",
                label=label,
            )
            axes[3].plot(
                frame.step,
                frame.laplacian_l1,
                marker="o",
                label=label,
            )
        titles = [
            "PSNR to clean",
            "SSIM to clean",
            "High-frequency correlation",
            "High-frequency L1 error",
        ]
        for axis, title in zip(axes, titles, strict=True):
            axis.set_title(title)
            axis.set_xlabel("correction step")
            axis.grid(alpha=0.25)
            axis.legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(checkpoint_output / "metric_curves.png", dpi=180)
        plt.close(figure)

        best_external = max(
            (label for label in labels if label.startswith("external_")),
            key=lambda label: checkpoint_summary["modes"][label]["final_psnr"],
        )
        trajectory_modes = ["correction_only", best_external]
        figure, axes = plt.subplots(
            len(trajectory_modes) * args.samples,
            len(selected_steps),
            figsize=(2.25 * len(selected_steps), 2.2 * len(trajectory_modes) * args.samples),
            squeeze=False,
        )
        for mode_index, label in enumerate(trajectory_modes):
            for sample_index in range(args.samples):
                row = mode_index * args.samples + sample_index
                for column, step in enumerate(selected_steps):
                    value = selected_states[label][step][sample_index * args.paths]
                    axes[row, column].imshow(display_image(value))
                    axes[row, column].axis("off")
                    if row == 0:
                        axes[row, column].set_title(f"step {step}")
                axes[row, 0].set_ylabel(f"{label}\nsample {sample_index}")
        figure.tight_layout()
        figure.savefig(checkpoint_output / "trajectory_comparison.png", dpi=180)
        plt.close(figure)

        del model, checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()

    (args.output / "summary.json").write_text(
        json.dumps(all_summaries, indent=2), encoding="utf-8"
    )
    print(json.dumps(all_summaries, indent=2))


if __name__ == "__main__":
    main()
