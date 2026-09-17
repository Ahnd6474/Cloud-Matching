from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from stochastic_bridge.config import config_from_dict
from stochastic_bridge.model import StochasticImageBridge
from stochastic_bridge.prepared import PreparedBridgeDataset
from stochastic_bridge.schedule import VPNoiseSchedule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a trained cloud-matching run")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--cloud-samples", type=int, default=16)
    parser.add_argument("--rollout-paths", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def image_psnr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    mse = (a.float() - b.float()).square().flatten(1).mean(1).clamp_min(1e-10)
    return 10 * torch.log10(4.0 / mse)


def batch_ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x, y = (x.float() + 1) / 2, (y.float() + 1) / 2
    mu_x = F.avg_pool2d(x, 7, 1, 3)
    mu_y = F.avg_pool2d(y, 7, 1, 3)
    var_x = F.avg_pool2d(x * x, 7, 1, 3) - mu_x.square()
    var_y = F.avg_pool2d(y * y, 7, 1, 3) - mu_y.square()
    covariance = F.avg_pool2d(x * y, 7, 1, 3) - mu_x * mu_y
    numerator = (2 * mu_x * mu_y + 0.01**2) * (2 * covariance + 0.03**2)
    denominator = (
        mu_x.square() + mu_y.square() + 0.01**2
    ) * (var_x + var_y + 0.03**2)
    return (numerator / denominator.clamp_min(1e-8)).flatten(1).mean(1)


def to_image(value: torch.Tensor) -> np.ndarray:
    return ((value.detach().float().cpu().clamp(-1, 1) + 1) / 2).permute(1, 2, 0).numpy()


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    run_dir = args.run.expanduser().resolve()
    output_dir = run_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(
        run_dir / args.checkpoint,
        map_location=device,
        weights_only=False,
    )
    config = config_from_dict(checkpoint["config"])
    model = StochasticImageBridge(**config.model.__dict__).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    schedule = VPNoiseSchedule(
        config.schedule.steps,
        config.schedule.beta_start,
        config.schedule.beta_end,
    ).to(device)

    history = pd.DataFrame(json.loads((run_dir / "history.json").read_text()))
    history.to_csv(output_dir / "history.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history.epoch, history.train_loss, label="train")
    axes[0].plot(history.epoch, history.val_loss, label="validation")
    axes[0].set_title("Paired Full-Band Cloud Loss")
    axes[0].legend()
    axes[1].plot(history.epoch, history.val_current_psnr, label="input")
    axes[1].plot(history.epoch, history.val_output_psnr, label="output")
    axes[1].set_title("Validation PSNR")
    axes[1].legend()
    axes[2].plot(history.epoch, history.train_noise_variance, label="train")
    axes[2].plot(history.epoch, history.val_noise_variance, label="validation")
    axes[2].set_title("Learned latent variance lambda")
    axes[2].legend()
    for axis in axes:
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "01_training_curves.png", dpi=180)
    plt.close(fig)

    dataset = PreparedBridgeDataset(args.validation)
    clean = dataset[args.sample_index]["clean"][None].to(device)

    @torch.inference_mode()
    def evaluate_level(level: int) -> dict[str, torch.Tensor | int]:
        samples = args.cloud_samples
        current_level = torch.tensor([level], device=device, dtype=torch.long)
        goal_level = torch.tensor([max(level - 20, 0)], device=device, dtype=torch.long)
        answer_level = (goal_level - config.schedule.answer_jump).clamp_min(0)
        current = schedule.q_sample(clean, current_level)
        goal = schedule.q_sample(clean, goal_level)
        latent = torch.randn(1, samples, *clean.shape[1:], device=device)
        target = schedule.sample_answer_cloud(
            clean, current, current_level, answer_level, samples, latent
        )
        with autocast_context(device):
            predicted, learned_variance = model(
                current,
                goal,
                samples=samples,
                noise=latent,
                return_noise_variance=True,
            )
        return {
            "level": level,
            "current": current,
            "goal": goal,
            "target": target,
            "predicted": predicted.float(),
            "learned_variance": learned_variance.float(),
        }

    levels = [10, 25, 40, 60, 80, 100]
    results = [evaluate_level(level) for level in levels]
    severity_rows: list[dict[str, float | int]] = []
    for result in results:
        current = result["current"]
        target = result["target"]
        predicted = result["predicted"]
        assert isinstance(current, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        assert isinstance(predicted, torch.Tensor)
        pred_mean = predicted.mean(1)
        target_mean = target.mean(1)
        learned_variance = result["learned_variance"]
        assert isinstance(learned_variance, torch.Tensor)
        severity_rows.append(
            {
                "input_level": int(result["level"]),
                "current_psnr": image_psnr(current, clean).item(),
                "output_mean_psnr": image_psnr(pred_mean, clean).item(),
                "target_mean_psnr": image_psnr(target_mean, clean).item(),
                "predicted_diversity": predicted.std(1).mean().item(),
                "target_diversity": target.std(1).mean().item(),
                "learned_noise_variance": learned_variance.mean().item(),
            }
        )
    severity = pd.DataFrame(severity_rows)
    severity.to_csv(output_dir / "severity_metrics.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(severity.input_level, severity.current_psnr, "o-", label="input")
    axes[0].plot(severity.input_level, severity.output_mean_psnr, "o-", label="output mean")
    axes[0].plot(severity.input_level, severity.target_mean_psnr, "o--", label="target mean")
    axes[0].set_title("Restoration by corruption level")
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].legend()
    axes[1].plot(severity.input_level, severity.predicted_diversity, "o-", label="predicted")
    axes[1].plot(severity.input_level, severity.target_diversity, "o-", label="target")
    axes[1].set_title("Cloud diversity")
    axes[1].legend()
    axes[2].plot(severity.input_level, severity.learned_noise_variance, "o-", color="tab:purple")
    axes[2].set_title("Learned variance lambda")
    for axis in axes:
        axis.set_xlabel("input corruption level")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "02_severity_curves.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(len(results), 6, figsize=(14, 2.25 * len(results)))
    for row, result in enumerate(results):
        current = result["current"]
        goal = result["goal"]
        target = result["target"]
        predicted = result["predicted"]
        assert all(isinstance(x, torch.Tensor) for x in (current, goal, target, predicted))
        target_mean = target.mean(1)[0]
        pred_mean = predicted.mean(1)[0]
        pred_std = predicted[0].std(0).mean(0)
        values = [clean[0], current[0], goal[0], target_mean, pred_mean]
        titles = ["clean", "current", "goal", "target mean", "predicted mean"]
        for column, (value, title) in enumerate(zip(values, titles, strict=True)):
            axes[row, column].imshow(to_image(value))
            axes[row, column].axis("off")
            if row == 0:
                axes[row, column].set_title(title)
        axes[row, 5].imshow(pred_std.detach().cpu(), cmap="magma")
        axes[row, 5].axis("off")
        if row == 0:
            axes[row, 5].set_title("predicted std")
        axes[row, 0].set_ylabel(f"s={result['level']}")
    fig.tight_layout()
    fig.savefig(output_dir / "03_severity_samples.png", dpi=180)
    plt.close(fig)

    rollout_paths = args.rollout_paths
    rollout_steps = args.rollout_steps
    target = clean.expand(rollout_paths, -1, -1, -1)
    goal_0 = target
    initial_noise = torch.randn_like(target[:1]).expand_as(target)
    current = schedule.q_sample(
        target,
        torch.full((rollout_paths,), config.schedule.steps, device=device, dtype=torch.long),
        initial_noise,
    )
    persistent_latent = torch.randn(rollout_paths, 1, *target.shape[1:], device=device)
    states: list[torch.Tensor] = []
    rollout_rows: list[dict[str, float | int]] = []
    previous: torch.Tensor | None = None
    with torch.inference_mode():
        target_h, _ = model.encode_condition(goal_0, goal_0)
        for step in range(rollout_steps + 1):
            h_t, _ = model.encode_condition(current, goal_0)
            variance = model.noise_variance_from_condition(h_t).squeeze(-1)
            update = (
                torch.zeros(rollout_paths, device=device)
                if previous is None
                else (current - previous).abs().flatten(1).mean(1)
            )
            psnr = image_psnr(current, target)
            ssim = batch_ssim(current, target)
            hidden_cosine = F.cosine_similarity(
                h_t.float().flatten(1), target_h.float().flatten(1), dim=1
            )
            rollout_rows.append(
                {
                    "step": step,
                    "psnr_mean": psnr.mean().item(),
                    "psnr_std": psnr.std().item(),
                    "ssim_mean": ssim.mean().item(),
                    "h_goal_cosine_mean": hidden_cosine.mean().item(),
                    "noise_variance_mean": variance.mean().item(),
                    "update_l1_mean": update.mean().item(),
                    "trajectory_std": current.std(0).mean().item(),
                }
            )
            states.append(current.detach().float().cpu())
            if step == rollout_steps:
                break
            previous = current
            with autocast_context(device):
                current = model(
                    current,
                    goal_0,
                    samples=1,
                    noise=persistent_latent,
                )[:, 0].float()

    rollout = pd.DataFrame(rollout_rows)
    rollout.to_csv(output_dir / "rollout_metrics.csv", index=False)
    selected_steps = sorted(set([0, 1, 2, 5, 10, 20, rollout_steps]))
    selected_steps = [step for step in selected_steps if step <= rollout_steps]
    fig, axes = plt.subplots(
        rollout_paths,
        len(selected_steps),
        figsize=(2.2 * len(selected_steps), 2.2 * rollout_paths),
        squeeze=False,
    )
    for row in range(rollout_paths):
        for column, step in enumerate(selected_steps):
            axes[row, column].imshow(to_image(states[step][row]))
            axes[row, column].axis("off")
            if row == 0:
                axes[row, column].set_title(f"step {step}")
        axes[row, 0].set_ylabel(f"path {row}")
    fig.tight_layout()
    fig.savefig(output_dir / "04_rollout_trajectories.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    steps = rollout.step
    axes[0, 0].plot(steps, rollout.psnr_mean)
    axes[0, 0].set_title("PSNR to fixed goal")
    axes[0, 1].plot(steps, rollout.ssim_mean)
    axes[0, 1].set_title("SSIM to fixed goal")
    axes[0, 2].plot(steps, rollout.noise_variance_mean, color="tab:red")
    axes[0, 2].set_title("Learned variance lambda")
    axes[1, 0].plot(steps, rollout.h_goal_cosine_mean, color="tab:purple")
    axes[1, 0].set_title("cosine(h_t, h_goal)")
    axes[1, 1].semilogy(steps[1:], rollout.update_l1_mean.iloc[1:].clip(lower=1e-8))
    axes[1, 1].set_title("Update L1")
    axes[1, 2].plot(steps, rollout.trajectory_std)
    axes[1, 2].set_title("Trajectory diversity")
    for axis in axes.flat:
        axis.set_xlabel("rollout step")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "05_rollout_metrics.png", dpi=180)
    plt.close(fig)

    best_loss_row = history.loc[history.val_loss.idxmin()]
    best_psnr_row = history.loc[history.val_output_psnr.idxmax()]
    best_rollout_row = rollout.loc[rollout.psnr_mean.idxmax()]
    summary = {
        "device": str(device),
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "best_loss_epoch": int(best_loss_row.epoch),
        "best_val_loss": float(best_loss_row.val_loss),
        "best_psnr_epoch": int(best_psnr_row.epoch),
        "best_val_output_psnr": float(best_psnr_row.val_output_psnr),
        "final_val_output_psnr": float(history.val_output_psnr.iloc[-1]),
        "final_val_input_psnr": float(history.val_current_psnr.iloc[-1]),
        "final_val_noise_variance": float(history.val_noise_variance.iloc[-1]),
        "rollout_best_step": int(best_rollout_row.step),
        "rollout_best_psnr": float(best_rollout_row.psnr_mean),
        "rollout_final_psnr": float(rollout.psnr_mean.iloc[-1]),
        "rollout_final_ssim": float(rollout.ssim_mean.iloc[-1]),
        "rollout_final_h_goal_cosine": float(rollout.h_goal_cosine_mean.iloc[-1]),
        "rollout_final_update_l1": float(rollout.update_l1_mean.iloc[-1]),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
