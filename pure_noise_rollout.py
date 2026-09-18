from __future__ import annotations

import argparse
import json
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
    parser = argparse.ArgumentParser(description="Roll out from independent pure noise")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--goals", type=int, default=4)
    parser.add_argument("--paths", type=int, default=4)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=731)
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
    score = ((2 * mu_x * mu_y + 0.01**2) * (2 * covariance + 0.03**2)) / (
        (mu_x.square() + mu_y.square() + 0.01**2)
        * (var_x + var_y + 0.03**2)
    ).clamp_min(1e-8)
    return score.flatten(1).mean(1)


def laplacian(image: torch.Tensor) -> torch.Tensor:
    kernel = image.new_tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    kernel = kernel.reshape(1, 1, 3, 3).expand(image.shape[1], 1, 3, 3)
    return F.conv2d(image.float(), kernel.float(), padding=1, groups=image.shape[1])


def batch_correlation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.flatten(1) - x.flatten(1).mean(1, keepdim=True)
    y = y.flatten(1) - y.flatten(1).mean(1, keepdim=True)
    return F.cosine_similarity(x, y, dim=1)


def to_image(value: torch.Tensor):
    return ((value.detach().float().cpu().clamp(-1, 1) + 1) / 2).permute(1, 2, 0)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = args.run.resolve()
    output_dir = run_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    state = torch.load(run_dir / args.checkpoint, map_location=device, weights_only=False)
    config = config_from_dict(state["config"])
    model = StochasticImageBridge(**config.model.__dict__).to(device).eval()
    model.load_state_dict(state["model"])

    dataset = PreparedBridgeDataset(args.validation)
    goal_count = min(args.goals, len(dataset))
    clean_goals = torch.stack([dataset[index * 2]["clean"] for index in range(goal_count)]).to(device)
    height, width = clean_goals.shape[-2:]
    lowpass_size = (max(1, height // 8), max(1, width // 8))
    lowpass = F.interpolate(clean_goals, size=lowpass_size, mode="area")
    lowpass = F.interpolate(lowpass, size=(height, width), mode="bilinear", align_corners=False)
    conditions = {
        "clean_goal": clean_goals,
        "lowpass_goal": lowpass,
        "zero_goal": torch.zeros_like(clean_goals),
    }

    targets = clean_goals[:, None].expand(-1, args.paths, -1, -1, -1)
    targets = targets.reshape(goal_count * args.paths, *clean_goals.shape[1:])
    start_noise = torch.randn_like(targets)
    persistent_latent = torch.randn(
        goal_count * args.paths,
        1,
        *clean_goals.shape[1:],
        device=device,
    )
    selected_steps = sorted(set([0, 1, 2, 4, 6, 10, 20, args.steps]))
    selected_steps = [step for step in selected_steps if step <= args.steps]
    all_rows: list[dict[str, float | int | str]] = []
    summaries: dict[str, dict[str, float | int]] = {}

    with torch.inference_mode():
        for mode, condition_base in conditions.items():
            condition = condition_base[:, None].expand(-1, args.paths, -1, -1, -1)
            condition = condition.reshape_as(targets)
            current = start_noise.clone()
            states: dict[int, torch.Tensor] = {}
            mode_rows: list[dict[str, float | int | str]] = []
            for step in range(args.steps + 1):
                psnr_values = psnr(current, targets)
                ssim_values = ssim(current, targets)
                high_frequency_correlation = batch_correlation(
                    laplacian(current), laplacian(targets)
                )
                grouped = current.reshape(
                    goal_count, args.paths, *current.shape[1:]
                )
                trajectory_diversity = grouped.std(1).mean()
                noise_variance = model.predict_noise_variance(current, condition)
                row: dict[str, float | int | str] = {
                    "mode": mode,
                    "step": step,
                    "psnr_mean": psnr_values.mean().item(),
                    "psnr_std": psnr_values.std().item(),
                    "ssim_mean": ssim_values.mean().item(),
                    "high_frequency_correlation": high_frequency_correlation.mean().item(),
                    "trajectory_diversity": trajectory_diversity.item(),
                    "noise_variance_mean": noise_variance.mean().item(),
                }
                mode_rows.append(row)
                all_rows.append(row)
                if step in selected_steps:
                    states[step] = current.detach().float().cpu()
                if step == args.steps:
                    break
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=device.type == "cuda",
                ):
                    current = model(
                        current,
                        condition,
                        samples=1,
                        noise=persistent_latent,
                    )[:, 0].float()

            mode_frame = pd.DataFrame(mode_rows)
            best_row = mode_frame.loc[mode_frame.psnr_mean.idxmax()]
            summaries[mode] = {
                "best_step": int(best_row.step),
                "best_psnr": float(best_row.psnr_mean),
                "final_psnr": float(mode_frame.psnr_mean.iloc[-1]),
                "final_ssim": float(mode_frame.ssim_mean.iloc[-1]),
                "final_high_frequency_correlation": float(
                    mode_frame.high_frequency_correlation.iloc[-1]
                ),
                "final_trajectory_diversity": float(
                    mode_frame.trajectory_diversity.iloc[-1]
                ),
            }

            fig, axes = plt.subplots(
                goal_count,
                len(selected_steps) + 1,
                figsize=(2.1 * (len(selected_steps) + 1), 2.1 * goal_count),
                squeeze=False,
            )
            for goal_index in range(goal_count):
                axes[goal_index, 0].imshow(to_image(condition_base[goal_index]))
                axes[goal_index, 0].axis("off")
                if goal_index == 0:
                    axes[goal_index, 0].set_title("condition")
                path_index = goal_index * args.paths
                for column, step in enumerate(selected_steps, start=1):
                    axes[goal_index, column].imshow(to_image(states[step][path_index]))
                    axes[goal_index, column].axis("off")
                    if goal_index == 0:
                        axes[goal_index, column].set_title(f"step {step}")
            fig.suptitle(f"Pure-noise rollout: {mode}", y=1.005)
            fig.tight_layout()
            fig.savefig(output_dir / f"06_pure_noise_{mode}.png", dpi=180, bbox_inches="tight")
            plt.close(fig)

    metrics = pd.DataFrame(all_rows)
    metrics.to_csv(output_dir / "pure_noise_rollout_metrics.csv", index=False)
    (output_dir / "pure_noise_rollout_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 4, figsize=(19, 4))
    for mode, frame in metrics.groupby("mode", sort=False):
        axes[0].plot(frame.step, frame.psnr_mean, label=mode)
        axes[1].plot(frame.step, frame.ssim_mean, label=mode)
        axes[2].plot(frame.step, frame.high_frequency_correlation, label=mode)
        axes[3].plot(frame.step, frame.trajectory_diversity, label=mode)
    titles = ["PSNR to clean", "SSIM to clean", "Laplacian correlation", "Trajectory diversity"]
    for axis, title in zip(axes, titles, strict=True):
        axis.set_title(title)
        axis.set_xlabel("rollout step")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "07_pure_noise_metrics.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
