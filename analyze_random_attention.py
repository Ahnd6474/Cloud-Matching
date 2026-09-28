from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchvision.utils import make_grid, save_image

from stochastic_bridge.config import config_from_dict
from stochastic_bridge.model import StochasticImageBridge
from stochastic_bridge.prepared import PreparedBridgeDataset, materialize_target_noise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a baseline, random-attention model, and gate-zero ablation."
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--random", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=25)
    parser.add_argument("--visualize", type=int, default=4)
    return parser.parse_args()


def psnr(predicted: Tensor, target: Tensor) -> Tensor:
    mse = (predicted.float() - target.float()).square().flatten(1).mean(1)
    return 10.0 * torch.log10(4.0 / mse.clamp_min(1e-10))


def ssim(predicted: Tensor, target: Tensor) -> Tensor:
    predicted = (predicted.float() + 1.0) / 2.0
    target = (target.float() + 1.0) / 2.0
    mean_x = F.avg_pool2d(predicted, 7, 1, 3)
    mean_y = F.avg_pool2d(target, 7, 1, 3)
    variance_x = F.avg_pool2d(predicted.square(), 7, 1, 3) - mean_x.square()
    variance_y = F.avg_pool2d(target.square(), 7, 1, 3) - mean_y.square()
    covariance = F.avg_pool2d(predicted * target, 7, 1, 3) - mean_x * mean_y
    numerator = (2 * mean_x * mean_y + 0.01**2) * (
        2 * covariance + 0.03**2
    )
    denominator = (mean_x.square() + mean_y.square() + 0.01**2) * (
        variance_x + variance_y + 0.03**2
    )
    return (numerator / denominator.clamp_min(1e-8)).flatten(1).mean(1)


def laplacian(image: Tensor) -> Tensor:
    kernel = image.new_tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    kernel = kernel.reshape(1, 1, 3, 3).expand(image.shape[1], 1, 3, 3)
    return F.conv2d(image.float(), kernel.float(), padding=1, groups=image.shape[1])


def correlation(first: Tensor, second: Tensor) -> Tensor:
    first = first.flatten(1) - first.flatten(1).mean(1, keepdim=True)
    second = second.flatten(1) - second.flatten(1).mean(1, keepdim=True)
    return F.cosine_similarity(first, second, dim=1)


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def load_model(path: Path, device: torch.device) -> StochasticImageBridge:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = config_from_dict(checkpoint["config"])
    model = StochasticImageBridge(**config.model.__dict__).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    return model


@torch.inference_mode()
def evaluate(
    checkpoint: Path,
    loader: DataLoader,
    device: torch.device,
    gate_zero: bool,
    visualize: int,
) -> tuple[dict[str, float], list[Tensor]]:
    model = load_model(checkpoint, device)
    gate_values: list[float] = []
    if model.architecture == "fullres_axial" and model.fullres.random_attention_enabled:
        gate_values = [layer.gate.item() for layer in model.fullres.random_attentions]
        if gate_zero:
            for layer in model.fullres.random_attentions:
                layer.gate.zero_()

    totals = {
        "mean_psnr": 0.0,
        "mean_ssim": 0.0,
        "sample_psnr": 0.0,
        "diversity": 0.0,
        "laplacian_correlation": 0.0,
        "sharpness": 0.0,
        "clean_sharpness": 0.0,
    }
    previews: list[Tensor] = []
    count = 0
    for raw in loader:
        current = raw["current"].to(device)
        goal = raw["goal"].to(device)
        clean = raw["clean"].to(device)
        noise = materialize_target_noise(raw, device)
        if noise is None:
            raise RuntimeError("validation data must contain reconstructable noise")
        with autocast_context(device):
            predicted = model(current, goal, samples=noise.shape[1], noise=noise)
        predicted = predicted.float()
        prediction_mean = predicted.mean(1)
        batch = clean.shape[0]
        clean_samples = clean[:, None].expand_as(predicted).flatten(0, 1)
        prediction_samples = predicted.flatten(0, 1)
        predicted_laplacian = laplacian(prediction_mean)
        clean_laplacian = laplacian(clean)
        totals["mean_psnr"] += psnr(prediction_mean, clean).sum().item()
        totals["mean_ssim"] += ssim(prediction_mean, clean).sum().item()
        totals["sample_psnr"] += psnr(prediction_samples, clean_samples).reshape(
            batch, -1
        ).mean(1).sum().item()
        totals["diversity"] += predicted.std(1, unbiased=False).mean(
            dim=(1, 2, 3)
        ).sum().item()
        totals["laplacian_correlation"] += correlation(
            predicted_laplacian, clean_laplacian
        ).sum().item()
        totals["sharpness"] += predicted_laplacian.abs().mean(
            dim=(1, 2, 3)
        ).sum().item()
        totals["clean_sharpness"] += clean_laplacian.abs().mean(
            dim=(1, 2, 3)
        ).sum().item()
        for image in prediction_mean[: max(0, visualize - len(previews))]:
            previews.append(image.detach().cpu())
        count += batch

    metrics = {name: value / count for name, value in totals.items()}
    metrics["sharpness_ratio"] = metrics["sharpness"] / max(
        metrics["clean_sharpness"], 1e-12
    )
    metrics["gate_1"] = gate_values[0] if gate_values else 0.0
    metrics["gate_2"] = gate_values[1] if gate_values else 0.0
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics, previews


def main() -> None:
    args = parse_args()
    if args.examples < 1 or args.visualize < 1:
        raise ValueError("examples and visualize must be positive")
    torch.manual_seed(17)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PreparedBridgeDataset(args.validation)
    count = min(args.examples, len(dataset))
    loader = DataLoader(Subset(dataset, range(count)), batch_size=1, shuffle=False)
    variants = {
        "baseline": (args.baseline, False),
        "random_attention": (args.random, False),
        "random_attention_gate_zero": (args.random, True),
    }
    results: dict[str, object] = {"device": str(device), "examples": count}
    predictions: dict[str, list[Tensor]] = {}
    for name, (checkpoint, gate_zero) in variants.items():
        metrics, previews = evaluate(
            checkpoint.resolve(), loader, device, gate_zero, args.visualize
        )
        results[name] = metrics
        predictions[name] = previews
        print(name, json.dumps(metrics, indent=2))

    raw_previews = [dataset[index] for index in range(min(args.visualize, count))]
    columns: list[Tensor] = []
    for index, record in enumerate(raw_previews):
        columns.extend(
            [
                record["clean"],
                record["goal"],
                record["current"],
                predictions["baseline"][index],
                predictions["random_attention"][index],
                predictions["random_attention_gate_zero"][index],
            ]
        )
    args.output.mkdir(parents=True, exist_ok=True)
    grid = make_grid(
        torch.stack(columns).clamp(-1, 1).add(1).div(2),
        nrow=6,
        padding=2,
    )
    save_image(grid, args.output / "comparison.png")
    (args.output / "metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print("columns: clean | goal | current | baseline | random | random_gate_zero")


if __name__ == "__main__":
    main()
