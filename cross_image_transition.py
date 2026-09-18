from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor

from stochastic_bridge.config import config_from_dict
from stochastic_bridge.datasets import ImageDirectoryDataset
from stochastic_bridge.model import StochasticImageBridge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test a clean full-resolution image A -> clean image B rollout"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--a-index", type=int, default=0)
    parser.add_argument("--b-index", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--paths", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def psnr(predicted: Tensor, target: Tensor) -> Tensor:
    mse = (predicted.float() - target.float()).square().flatten(1).mean(1)
    return 10.0 * torch.log10(4.0 / mse.clamp_min(1e-10))


def ssim(predicted: Tensor, target: Tensor) -> Tensor:
    x = (predicted.float() + 1.0) / 2.0
    y = (target.float() + 1.0) / 2.0
    mean_x = F.avg_pool2d(x, 7, 1, 3)
    mean_y = F.avg_pool2d(y, 7, 1, 3)
    variance_x = F.avg_pool2d(x * x, 7, 1, 3) - mean_x.square()
    variance_y = F.avg_pool2d(y * y, 7, 1, 3) - mean_y.square()
    covariance = F.avg_pool2d(x * y, 7, 1, 3) - mean_x * mean_y
    score = (
        (2 * mean_x * mean_y + 0.01**2) * (2 * covariance + 0.03**2)
    ) / (
        (mean_x.square() + mean_y.square() + 0.01**2)
        * (variance_x + variance_y + 0.03**2)
    ).clamp_min(1e-8)
    return score.flatten(1).mean(1)


def cosine(left: Tensor, right: Tensor) -> Tensor:
    return F.cosine_similarity(left.float().flatten(1), right.float().flatten(1), dim=1)


def show_image(axis, image: Tensor, title: str) -> None:
    display = ((image.detach().float().cpu().clamp(-1, 1) + 1.0) / 2.0).permute(1, 2, 0)
    axis.imshow(display)
    axis.set_title(title)
    axis.axis("off")


def main() -> None:
    args = parse_args()
    if args.a_index == args.b_index:
        raise ValueError("A and B must be different source images")
    if args.paths < 2 or args.iterations < 1:
        raise ValueError("paths must be >=2 and iterations must be positive")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = config_from_dict(state["config"])
    model = StochasticImageBridge(**config.model.__dict__).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    dataset = ImageDirectoryDataset(
        args.data,
        args.image_size,
        random_crop=False,
        crop_mode="native",
        horizontal_flip=False,
    )
    image_a = dataset[args.a_index][None].to(device)
    image_b = dataset[args.b_index][None].to(device)
    current = image_a.expand(args.paths, -1, -1, -1).clone()
    goal = image_b.expand(args.paths, -1, -1, -1)
    target_a = image_a.expand_as(current)
    target_b = image_b.expand_as(current)
    persistent_latent = torch.randn(
        args.paths,
        1,
        *current.shape[1:],
        device=device,
        dtype=current.dtype,
    )

    requested_steps = {0, 1, 2, 4, 6, 10, 20, args.iterations}
    requested_steps = {step for step in requested_steps if step <= args.iterations}
    states: dict[int, Tensor] = {}
    rows: list[dict[str, float]] = []
    previous: Tensor | None = None

    with torch.inference_mode():
        goal_hidden, _ = model.encode_condition(goal, goal)
        for step in range(args.iterations + 1):
            hidden, _ = model.encode_condition(current, goal)
            variance = model.noise_variance_from_condition(hidden).squeeze(-1)
            update = (
                torch.zeros(args.paths, device=device)
                if previous is None
                else (current - previous).abs().flatten(1).mean(1)
            )
            rows.append(
                {
                    "step": float(step),
                    "psnr_to_a": psnr(current, target_a).mean().item(),
                    "psnr_to_b": psnr(current, target_b).mean().item(),
                    "ssim_to_a": ssim(current, target_a).mean().item(),
                    "ssim_to_b": ssim(current, target_b).mean().item(),
                    "cosine_to_a": cosine(current, target_a).mean().item(),
                    "cosine_to_b": cosine(current, target_b).mean().item(),
                    "l1_to_a": (current - target_a).abs().mean().item(),
                    "l1_to_b": (current - target_b).abs().mean().item(),
                    "condition_cosine_to_goal": cosine(hidden, goal_hidden).mean().item(),
                    "noise_variance_mean": variance.mean().item(),
                    "noise_variance_std": variance.std().item(),
                    "update_l1": update.mean().item(),
                    "trajectory_diversity": current.std(0).mean().item(),
                }
            )
            if step in requested_steps:
                states[step] = current.detach().cpu()
            if step == args.iterations:
                break
            previous = current
            current = model(
                current,
                goal,
                samples=1,
                noise=persistent_latent,
            )[:, 0]

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "cross_image_metrics.csv", index=False)

    selected_steps = sorted(states)
    display_paths = min(4, args.paths)
    columns = 2 + len(selected_steps)
    figure, axes = plt.subplots(
        display_paths, columns, figsize=(2.2 * columns, 2.25 * display_paths), squeeze=False
    )
    for row in range(display_paths):
        show_image(axes[row, 0], image_a[0], "A (clean)" if row == 0 else "")
        show_image(axes[row, 1], image_b[0], "B goal (clean)" if row == 0 else "")
        for column, step in enumerate(selected_steps, start=2):
            show_image(
                axes[row, column],
                states[step][row],
                f"step {step}" if row == 0 else "",
            )
        axes[row, 0].set_ylabel(f"trajectory {row}")
    figure.suptitle("Clean full-resolution A → B rollout with persistent latent", y=1.01)
    figure.tight_layout()
    figure.savefig(output / "cross_image_rollout.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(2, 3, figsize=(16, 8))
    steps = metrics.step
    axes[0, 0].plot(steps, metrics.psnr_to_a, label="to A")
    axes[0, 0].plot(steps, metrics.psnr_to_b, label="to B")
    axes[0, 0].set_title("PSNR")
    axes[0, 0].legend()
    axes[0, 1].plot(steps, metrics.ssim_to_a, label="to A")
    axes[0, 1].plot(steps, metrics.ssim_to_b, label="to B")
    axes[0, 1].set_title("SSIM")
    axes[0, 1].legend()
    axes[0, 2].plot(steps, metrics.l1_to_a, label="to A")
    axes[0, 2].plot(steps, metrics.l1_to_b, label="to B")
    axes[0, 2].set_title("Image L1")
    axes[0, 2].legend()
    axes[1, 0].plot(steps, metrics.condition_cosine_to_goal)
    axes[1, 0].set_title("cosine(h(A_t,B), h(B,B))")
    axes[1, 1].plot(steps, metrics.noise_variance_mean, label="lambda")
    axes[1, 1].plot(steps, metrics.trajectory_diversity, label="path diversity")
    axes[1, 1].set_title("Latent variance / diversity")
    axes[1, 1].legend()
    axes[1, 2].semilogy(steps[1:], metrics.update_l1.iloc[1:].clip(lower=1e-8))
    axes[1, 2].set_title("Per-step update L1")
    for axis in axes.flat:
        axis.grid(alpha=0.3)
        axis.set_xlabel("rollout step")
    figure.tight_layout()
    figure.savefig(output / "cross_image_curves.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    final = states[args.iterations]
    final_mean = final.mean(0)
    final_std = final.std(0).mean(0)
    figure, axes = plt.subplots(1, 5, figsize=(14, 3))
    show_image(axes[0], image_a[0], "A")
    show_image(axes[1], image_b[0], "B goal")
    show_image(axes[2], final[0], "final path 0")
    show_image(axes[3], final_mean, "final mean")
    axes[4].imshow(final_std.numpy(), cmap="magma")
    axes[4].set_title("final path std")
    axes[4].axis("off")
    figure.tight_layout()
    figure.savefig(output / "cross_image_endpoint.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    initial = metrics.iloc[0]
    final_row = metrics.iloc[-1]
    best_b_index = int(metrics.psnr_to_b.idxmax())
    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "a_index": args.a_index,
        "b_index": args.b_index,
        "image_size": args.image_size,
        "paths": args.paths,
        "iterations": args.iterations,
        "initial_psnr_to_b": float(initial.psnr_to_b),
        "final_psnr_to_a": float(final_row.psnr_to_a),
        "final_psnr_to_b": float(final_row.psnr_to_b),
        "final_ssim_to_a": float(final_row.ssim_to_a),
        "final_ssim_to_b": float(final_row.ssim_to_b),
        "best_b_step": int(metrics.loc[best_b_index, "step"]),
        "best_psnr_to_b": float(metrics.loc[best_b_index, "psnr_to_b"]),
        "final_noise_variance": float(final_row.noise_variance_mean),
        "final_trajectory_diversity": float(final_row.trajectory_diversity),
    }
    (output / "cross_image_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
