from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from next_state_rollout import (
    aggregate_steps,
    autocast_context,
    measure,
    select_records,
    write_csv,
)
from stochastic_bridge.config import config_from_dict
from stochastic_bridge.model import StochasticImageBridge
from stochastic_bridge.prepared import PreparedBridgeDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablate the random-attention gate.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=4)
    parser.add_argument("--paths", type=int, default=4)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--minimum-level", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260922)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    dataset = PreparedBridgeDataset(args.validation)
    selected = select_records(dataset, args.examples, args.minimum_level)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = config_from_dict(checkpoint["config"])
    model = StochasticImageBridge(**config.model.__dict__).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    layers = model.fullres.random_attentions
    if len(layers) != 1:
        raise RuntimeError("the ablation requires exactly one random-attention layer")
    trained_gates = [float(layer.gate.detach()) for layer in layers]
    modes = {
        "normal": (1.0,),
        "gate_zero": (0.0,),
    }
    all_rows: list[dict[str, float | int | str]] = []
    summary: dict[str, object] = {
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "trained_gates": trained_gates,
        "steps": args.steps,
        "paths": args.paths,
        "modes": {},
    }

    with torch.inference_mode():
        for mode, scales in modes.items():
            for layer, value, scale in zip(layers, trained_gates, scales, strict=True):
                layer.gate.fill_(value * scale)
            torch.manual_seed(args.seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(args.seed)
            mode_rows: list[dict[str, float | int]] = []
            for example, (dataset_index, record) in enumerate(selected):
                clean_one = record["clean"][None].to(device)
                goal_one = record["goal"][None].to(device)
                current_one = record["current"][None].to(device)
                clean = clean_one.expand(args.paths, -1, -1, -1)
                goal = goal_one.expand(args.paths, -1, -1, -1)
                state = current_one.expand(args.paths, -1, -1, -1).clone()
                initial_level = int(record["current_level"])
                first = {
                    "example": example,
                    "dataset_index": dataset_index,
                    "initial_level": initial_level,
                    "step": 0,
                    "nominal_level": initial_level,
                    **measure(state, clean, goal, None),
                }
                mode_rows.append(first)
                for step in range(1, args.steps + 1):
                    previous = state
                    with autocast_context(device):
                        state = model(previous, goal, samples=1)[:, 0]
                    state = state.float().clamp(-1, 1)
                    mode_rows.append(
                        {
                            "example": example,
                            "dataset_index": dataset_index,
                            "initial_level": initial_level,
                            "step": step,
                            "nominal_level": max(
                                initial_level - step * config.schedule.answer_jump, 0
                            ),
                            **measure(state, clean, goal, previous),
                        }
                    )
            aggregate = aggregate_steps(mode_rows)
            write_csv(args.output / f"{mode}.csv", aggregate)
            best = max(aggregate, key=lambda row: float(row["mean_psnr_clean"]))
            for row in mode_rows:
                all_rows.append({"mode": mode, **row})
            summary["modes"][mode] = {
                "best_step": int(best["step"]),
                "best_psnr": float(best["mean_psnr_clean"]),
                "step_10_psnr": float(aggregate[10]["mean_psnr_clean"]),
                "step_20_psnr": float(aggregate[20]["mean_psnr_clean"]),
                "step_30_psnr": float(aggregate[30]["mean_psnr_clean"]),
                "step_30_ssim": float(aggregate[30]["path_ssim_clean"]),
                "step_30_diversity": float(aggregate[30]["path_diversity"]),
                "step_30_sharpness_ratio": float(aggregate[30]["sharpness_ratio"]),
                "step_30_saturation": float(aggregate[30]["saturation_fraction"]),
            }

    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
