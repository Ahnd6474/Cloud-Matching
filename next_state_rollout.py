from __future__ import annotations

import argparse
import csv
import json
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from stochastic_bridge.config import config_from_dict
from stochastic_bridge.model import StochasticImageBridge
from stochastic_bridge.prepared import PreparedBridgeDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Roll out a learned current-relative bridge with one fixed goal."
    )
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=4)
    parser.add_argument("--paths", type=int, default=4)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--minimum-level", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260922)
    return parser.parse_args()


def psnr(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = (value.float() - target.float()).square().flatten(1).mean(1).clamp_min(1e-10)
    return 10.0 * torch.log10(4.0 / mse)


def ssim(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    value = (value.float() + 1.0) / 2.0
    target = (target.float() + 1.0) / 2.0
    mu_x = F.avg_pool2d(value, 7, 1, 3)
    mu_y = F.avg_pool2d(target, 7, 1, 3)
    var_x = F.avg_pool2d(value.square(), 7, 1, 3) - mu_x.square()
    var_y = F.avg_pool2d(target.square(), 7, 1, 3) - mu_y.square()
    covariance = F.avg_pool2d(value * target, 7, 1, 3) - mu_x * mu_y
    numerator = (2 * mu_x * mu_y + 0.01**2) * (2 * covariance + 0.03**2)
    denominator = (
        (mu_x.square() + mu_y.square() + 0.01**2)
        * (var_x + var_y + 0.03**2)
    )
    return (numerator / denominator.clamp_min(1e-8)).flatten(1).mean(1)


def laplacian(value: torch.Tensor) -> torch.Tensor:
    kernel = value.new_tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    kernel = kernel.reshape(1, 1, 3, 3).expand(value.shape[1], 1, 3, 3)
    return F.conv2d(value.float(), kernel.float(), padding=1, groups=value.shape[1])


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def measure(
    state: torch.Tensor,
    clean: torch.Tensor,
    goal: torch.Tensor,
    previous: torch.Tensor | None,
) -> dict[str, float]:
    mean_state = state.mean(0, keepdim=True)
    mean_clean = clean[:1]
    mean_goal = goal[:1]
    clean_sharpness = laplacian(mean_clean).abs().mean().clamp_min(1e-8)
    metrics = {
        "path_psnr_clean": psnr(state, clean).mean().item(),
        "mean_psnr_clean": psnr(mean_state, mean_clean).item(),
        "path_ssim_clean": ssim(state, clean).mean().item(),
        "mean_psnr_goal": psnr(mean_state, mean_goal).item(),
        "path_diversity": state.std(0, unbiased=False).mean().item(),
        "sharpness_ratio": (
            laplacian(mean_state).abs().mean() / clean_sharpness
        ).item(),
        "saturation_fraction": (state.abs() >= 0.999).float().mean().item(),
    }
    metrics["step_delta_l1"] = (
        0.0 if previous is None else (state - previous).abs().mean().item()
    )
    return metrics


def select_records(
    dataset: PreparedBridgeDataset, examples: int, minimum_level: int
) -> list[tuple[int, dict[str, torch.Tensor]]]:
    candidates: list[tuple[int, int, dict[str, torch.Tensor]]] = []
    for index in range(len(dataset)):
        record = dataset[index]
        level = int(record["current_level"])
        if level >= minimum_level:
            candidates.append((level, index, record))
    if len(candidates) < examples:
        raise RuntimeError(
            f"only {len(candidates)} records have current_level >= {minimum_level}"
        )
    candidates.sort(key=lambda item: (-item[0], item[1]))
    # Spread selection across the high-noise candidate list instead of taking
    # adjacent crops from the beginning of a shard.
    positions = torch.linspace(0, len(candidates) - 1, examples).round().long()
    return [(candidates[int(position)][1], candidates[int(position)][2]) for position in positions]


def write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_steps(rows: list[dict[str, float | int]]) -> list[dict[str, float | int]]:
    aggregates: list[dict[str, float | int]] = []
    steps = sorted({int(row["step"]) for row in rows})
    excluded = {"example", "dataset_index", "initial_level", "step", "nominal_level"}
    for step in steps:
        selected = [row for row in rows if int(row["step"]) == step]
        aggregate: dict[str, float | int] = {"step": step}
        for key in rows[0]:
            if key in excluded:
                continue
            aggregate[key] = sum(float(row[key]) for row in selected) / len(selected)
        aggregate["nominal_level"] = sum(
            float(row["nominal_level"]) for row in selected
        ) / len(selected)
        aggregates.append(aggregate)
    return aggregates


def main() -> None:
    args = parse_args()
    if args.examples < 1 or args.paths < 2 or args.steps < 1:
        raise ValueError("examples and steps must be positive; paths must be at least two")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)

    dataset = PreparedBridgeDataset(args.validation)
    selected = select_records(dataset, args.examples, args.minimum_level)
    all_summary: dict[str, object] = {
        "device": str(device),
        "paths": args.paths,
        "steps": args.steps,
        "selection": [
            {"index": index, "current_level": int(record["current_level"])}
            for index, record in selected
        ],
        "checkpoints": {},
    }

    for checkpoint_path in args.checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = config_from_dict(checkpoint["config"])
        model = StochasticImageBridge(**config.model.__dict__).to(device).eval()
        model.load_state_dict(checkpoint["model"])
        label = f"{checkpoint_path.stem}_epoch_{int(checkpoint.get('epoch', -1))}"
        output = args.output / label
        output.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, float | int]] = []
        visual_states: list[dict[int, torch.Tensor]] = []
        selected_steps = sorted(
            step
            for step in set(
                [0, 1, 2, 4, 6, 8, 10, 12, 15, 20, 30, 40, 50, 75, 100, args.steps]
            )
            if step <= args.steps
        )

        with torch.inference_mode():
            for example_index, (dataset_index, record) in enumerate(selected):
                clean_one = record["clean"][None].to(device)
                goal_one = record["goal"][None].to(device)
                current_one = record["current"][None].to(device)
                clean = clean_one.expand(args.paths, -1, -1, -1)
                goal = goal_one.expand(args.paths, -1, -1, -1)
                state = current_one.expand(args.paths, -1, -1, -1).clone()
                initial_level = int(record["current_level"])
                states = {0: state[0].detach().cpu()}
                rows.append(
                    {
                        "example": example_index,
                        "dataset_index": dataset_index,
                        "initial_level": initial_level,
                        "step": 0,
                        "nominal_level": initial_level,
                        **measure(state, clean, goal, None),
                    }
                )
                for step in range(1, args.steps + 1):
                    previous = state
                    with autocast_context(device):
                        state = model(previous, goal, samples=1)[:, 0]
                    state = state.float().clamp(-1, 1)
                    rows.append(
                        {
                            "example": example_index,
                            "dataset_index": dataset_index,
                            "initial_level": initial_level,
                            "step": step,
                            "nominal_level": max(
                                initial_level - step * config.schedule.answer_jump, 0
                            ),
                            **measure(state, clean, goal, previous),
                        }
                    )
                    if step in selected_steps:
                        states[step] = state[0].detach().cpu()
                states[-2] = clean_one[0].detach().cpu()
                states[-1] = goal_one[0].detach().cpu()
                visual_states.append(states)

        write_csv(output / "metrics.csv", rows)
        aggregate = aggregate_steps(rows)
        write_csv(output / "aggregate_metrics.csv", aggregate)
        best = max(aggregate, key=lambda row: float(row["mean_psnr_clean"]))
        final = aggregate[-1]
        summary = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "schedule_steps": config.schedule.steps,
            "answer_jump": config.schedule.answer_jump,
            "adaptive_answer_jump": config.schedule.adaptive_answer_jump,
            "best_step": int(best["step"]),
            "best_mean_psnr_clean": float(best["mean_psnr_clean"]),
            "final_mean_psnr_clean": float(final["mean_psnr_clean"]),
            "final_path_diversity": float(final["path_diversity"]),
            "final_sharpness_ratio": float(final["sharpness_ratio"]),
            "final_saturation_fraction": float(final["saturation_fraction"]),
        }
        all_summary["checkpoints"][label] = summary
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

        columns = ["clean", "fixed goal", "initial"] + [
            f"step {step}" for step in selected_steps if step > 0
        ]
        grid_values: list[torch.Tensor] = []
        for states in visual_states:
            grid_values.extend([states[-2], states[-1], states[0]] + [
                states[step] for step in selected_steps if step > 0
            ])
        grid = (torch.stack(grid_values).clamp(-1, 1) + 1.0) / 2.0
        save_image(grid, output / "rollout.png", nrow=len(columns), padding=2)
        (output / "rollout_columns.json").write_text(
            json.dumps(columns, indent=2), encoding="utf-8"
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    (args.output / "summary.json").write_text(
        json.dumps(all_summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(all_summary, indent=2))


if __name__ == "__main__":
    main()
