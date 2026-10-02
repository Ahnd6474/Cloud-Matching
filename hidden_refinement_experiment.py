from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from torch import Tensor
from torchvision.utils import save_image

from next_state_rollout import autocast_context, measure, select_records
from stochastic_bridge.config import config_from_dict
from stochastic_bridge.model import StochasticImageBridge
from stochastic_bridge.prepared import PreparedBridgeDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare ordinary RGB recurrence with decoder-hidden recurrence. "
            "Hidden recurrence encodes current/goal once, repeatedly applies the "
            "trained full-resolution refinement stack, and uses RGB only as a readout."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=2)
    parser.add_argument("--paths", type=int, default=2)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--minimum-level", type=int, default=80)
    parser.add_argument("--hidden-alphas", type=float, nargs="+", default=[1.0, 0.5, 0.25])
    parser.add_argument("--seed", type=int, default=20261002)
    return parser.parse_args()


def initialize_hidden(
    model: StochasticImageBridge,
    condition: Tensor,
    samples: int,
    noise: Tensor,
) -> tuple[Tensor, Tensor | None]:
    """Mirror the pre-block portion of FullResolutionBridge.decode()."""
    core = model.fullres
    batch, height, width, dim = condition.shape
    expanded = condition[:, None].expand(-1, samples, -1, -1, -1)
    if core.random_attention_enabled:
        random_memory = core._random_memories(condition, samples, noise)
        tokens = expanded.reshape(batch * samples, height, width, dim)
        return tokens, random_memory

    energy = core.spatial_energy(condition)
    latent = core._latent_noise(condition, samples, noise)
    latent = latent * (energy / dim).sqrt()[:, None, :, :, None]
    tokens = (expanded + latent).reshape(batch * samples, height, width, dim)
    return tokens, None


def refine_hidden_once(
    model: StochasticImageBridge,
    tokens: Tensor,
    random_memory: Tensor | None,
    alpha: float,
) -> Tensor:
    """Apply one trained decoder stack while retaining its hidden output."""
    if not 0.0 < alpha <= 1.0:
        raise ValueError("hidden refinement alpha must lie in (0, 1]")
    core = model.fullres
    initial = tokens
    random_index = 0
    for index in range(len(core.blocks) + 1):
        while (
            core.random_attention_enabled
            and random_index < len(core.random_positions)
            and core.random_positions[random_index] == index
        ):
            if random_memory is None:
                raise RuntimeError("random attention requires a random memory")
            tokens = core.random_attentions[random_index](tokens, random_memory)
            random_index += 1
        if index == len(core.blocks):
            break
        tokens = core.blocks[index](tokens)
    if alpha < 1.0:
        tokens = initial + alpha * (tokens - initial)
    return tokens


def read_hidden(
    model: StochasticImageBridge,
    tokens: Tensor,
    anchor: Tensor,
    samples: int,
) -> Tensor:
    """Decode a hidden state without feeding the resulting RGB back into it."""
    core = model.fullres
    batch = anchor.shape[0]
    residual = core.max_residual * torch.tanh(
        core.output_head(core.output_norm(tokens))
    )
    residual = residual.permute(0, 3, 1, 2)
    flat_anchor = (
        anchor[:, None]
        .expand(-1, samples, -1, -1, -1)
        .reshape(batch * samples, *anchor.shape[1:])
    )
    output = (flat_anchor + residual).clamp(-1.0, 1.0)
    return output.reshape(batch, samples, *anchor.shape[1:])


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(
    rows: list[dict[str, float | int | str]], mode: str
) -> list[dict[str, float | int | str]]:
    selected_mode = [row for row in rows if row["mode"] == mode]
    excluded = {"mode", "example", "dataset_index", "initial_level", "step"}
    result: list[dict[str, float | int | str]] = []
    for step in sorted({int(row["step"]) for row in selected_mode}):
        selected = [row for row in selected_mode if int(row["step"]) == step]
        entry: dict[str, float | int | str] = {"mode": mode, "step": step}
        for key in selected[0]:
            if key in excluded:
                continue
            entry[key] = sum(float(row[key]) for row in selected) / len(selected)
        result.append(entry)
    return result


def selected_visual_steps(steps: int) -> list[int]:
    return sorted({step for step in (0, 1, 2, 4, 8, 12, 16, steps) if step <= steps})


def main() -> None:
    args = parse_args()
    if args.examples < 1 or args.paths < 2 or args.steps < 1:
        raise ValueError("examples/steps must be positive and paths must be at least two")
    if any(not 0.0 < alpha <= 1.0 for alpha in args.hidden_alphas):
        raise ValueError("all hidden alphas must lie in (0, 1]")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = config_from_dict(checkpoint["config"])
    if config.model.architecture != "fullres_axial":
        raise ValueError("hidden refinement experiment requires fullres_axial")
    model = StochasticImageBridge(**config.model.__dict__).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    dataset = PreparedBridgeDataset(args.validation)
    selected = select_records(dataset, args.examples, args.minimum_level)
    modes = ["rgb"] + [f"hidden_alpha_{alpha:g}" for alpha in args.hidden_alphas]
    rows: list[dict[str, float | int | str]] = []
    elapsed = {mode: 0.0 for mode in modes}
    states: dict[str, list[dict[int, Tensor]]] = {mode: [] for mode in modes}
    visual_steps = selected_visual_steps(args.steps)
    one_step_max_error = 0.0

    with torch.inference_mode():
        for example, (dataset_index, record) in enumerate(selected):
            clean_one = record["clean"][None].to(device)
            goal_one = record["goal"][None].to(device)
            current_one = record["current"][None].to(device)
            clean = clean_one.expand(args.paths, -1, -1, -1)
            goal = goal_one.expand(args.paths, -1, -1, -1)
            initial = current_one.expand(args.paths, -1, -1, -1).clone()
            initial_level = int(record["current_level"])
            generator = torch.Generator(device=device).manual_seed(args.seed + dataset_index)
            noise = torch.randn(
                args.paths,
                1,
                initial.shape[1],
                initial.shape[2],
                initial.shape[3],
                device=device,
                dtype=initial.dtype,
                generator=generator,
            )

            # Existing image-state recurrence, using the same persistent noise at
            # every step so the only structural difference is the recurrent state.
            state = initial.clone()
            rgb_visual = {0: state[0].detach().cpu()}
            rows.append(
                {
                    "mode": "rgb",
                    "example": example,
                    "dataset_index": dataset_index,
                    "initial_level": initial_level,
                    "step": 0,
                    **measure(state, clean, goal, None),
                }
            )
            synchronize(device)
            started = time.perf_counter()
            for step in range(1, args.steps + 1):
                previous = state
                with autocast_context(device):
                    state = model(previous, goal, samples=1, noise=noise)[:, 0]
                state = state.float().clamp(-1, 1)
                rows.append(
                    {
                        "mode": "rgb",
                        "example": example,
                        "dataset_index": dataset_index,
                        "initial_level": initial_level,
                        "step": step,
                        **measure(state, clean, goal, previous),
                    }
                )
                if step in visual_steps:
                    rgb_visual[step] = state[0].detach().cpu()
            synchronize(device)
            elapsed["rgb"] += time.perf_counter() - started
            states["rgb"].append(rgb_visual)

            # Encode once. Every hidden mode starts from exactly these tensors.
            with autocast_context(device):
                condition = model.fullres.encode_condition(initial, goal)
                initial_hidden, random_memory = initialize_hidden(
                    model, condition, samples=1, noise=noise
                )

            for alpha in args.hidden_alphas:
                mode = f"hidden_alpha_{alpha:g}"
                hidden = initial_hidden.clone()
                hidden_visual = {0: initial[0].detach().cpu()}
                previous_readout = initial
                rows.append(
                    {
                        "mode": mode,
                        "example": example,
                        "dataset_index": dataset_index,
                        "initial_level": initial_level,
                        "step": 0,
                        **measure(initial, clean, goal, None),
                    }
                )
                synchronize(device)
                started = time.perf_counter()
                for step in range(1, args.steps + 1):
                    with autocast_context(device):
                        hidden = refine_hidden_once(
                            model, hidden, random_memory, alpha=alpha
                        )
                        readout = read_hidden(model, hidden, initial, samples=1)[:, 0]
                    readout = readout.float().clamp(-1, 1)
                    rows.append(
                        {
                            "mode": mode,
                            "example": example,
                            "dataset_index": dataset_index,
                            "initial_level": initial_level,
                            "step": step,
                            **measure(readout, clean, goal, previous_readout),
                        }
                    )
                    if alpha == 1.0 and step == 1:
                        # With alpha=1, the first pass must be the exact original
                        # one-step decoder computation from the same condition/noise.
                        with autocast_context(device):
                            direct = model(initial, goal, samples=1, noise=noise)[:, 0]
                        one_step_max_error = max(
                            one_step_max_error,
                            float((readout - direct.float()).abs().max()),
                        )
                    previous_readout = readout
                    if step in visual_steps:
                        hidden_visual[step] = readout[0].detach().cpu()
                synchronize(device)
                elapsed[mode] += time.perf_counter() - started
                states[mode].append(hidden_visual)

    write_csv(args.output / "metrics.csv", rows)
    aggregates: dict[str, list[dict[str, float | int | str]]] = {
        mode: aggregate(rows, mode) for mode in modes
    }
    write_csv(
        args.output / "aggregate_metrics.csv",
        [row for mode in modes for row in aggregates[mode]],
    )

    summary_modes: dict[str, object] = {}
    total_transitions = args.examples * args.steps
    for mode in modes:
        mode_rows = aggregates[mode]
        best = max(mode_rows, key=lambda row: float(row["mean_psnr_clean"]))
        final = mode_rows[-1]
        summary_modes[mode] = {
            "best_step": int(best["step"]),
            "best_mean_psnr_clean": float(best["mean_psnr_clean"]),
            "final_mean_psnr_clean": float(final["mean_psnr_clean"]),
            "final_path_ssim_clean": float(final["path_ssim_clean"]),
            "final_path_diversity": float(final["path_diversity"]),
            "final_sharpness_ratio": float(final["sharpness_ratio"]),
            "final_saturation_fraction": float(final["saturation_fraction"]),
            "elapsed_seconds": elapsed[mode],
            "milliseconds_per_transition": 1000.0 * elapsed[mode] / total_transitions,
        }

        columns = ["clean", "goal", "initial"] + [
            f"step {step}" for step in visual_steps if step > 0
        ]
        images: list[Tensor] = []
        for (_, record), example_states in zip(selected, states[mode], strict=True):
            images.extend(
                [record["clean"], record["goal"], example_states[0]]
                + [example_states[step] for step in visual_steps if step > 0]
            )
        grid = (torch.stack(images).clamp(-1, 1) + 1.0) / 2.0
        save_image(grid, args.output / f"{mode}.png", nrow=len(columns), padding=2)

    summary = {
        "device": str(device),
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "validation": str(args.validation),
        "examples": args.examples,
        "paths": args.paths,
        "steps": args.steps,
        "persistent_noise": True,
        "hidden_anchor": "initial current RGB",
        "one_step_max_abs_error": one_step_max_error,
        "selection": [
            {"index": index, "current_level": int(record["current_level"])}
            for index, record in selected
        ],
        "modes": summary_modes,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (args.output / "columns.json").write_text(
        json.dumps(
            ["clean", "goal", "initial"]
            + [f"step {step}" for step in visual_steps if step > 0],
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
