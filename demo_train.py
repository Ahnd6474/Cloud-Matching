from __future__ import annotations

import torch

from stochastic_bridge import (
    SinkhornCorrectionCloudLoss,
    StochasticImageBridge,
    VPNoiseSchedule,
    build_bridge_batch,
)


def synthetic_images(batch: int, size: int, device: torch.device) -> torch.Tensor:
    """Create simple structured images without an external dataset."""
    y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, size, device=device),
        torch.linspace(-1.0, 1.0, size, device=device),
        indexing="ij",
    )
    images = []
    for index in range(batch):
        radius = 0.25 + 0.08 * index
        circle = ((x**2 + y**2) < radius).float() * 2.0 - 1.0
        stripe = torch.sin((index + 2) * 3.14159 * x).clamp(-1.0, 1.0)
        diagonal = (x + y + 0.2 * index).tanh()
        images.append(torch.stack([circle, stripe, diagonal]))
    return torch.stack(images)


def main() -> None:
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = 16
    clean = synthetic_images(batch=2, size=image_size, device=device)

    schedule = VPNoiseSchedule(steps=100).to(device)
    model = StochasticImageBridge(
        image_size=image_size,
        dim=64,
        patch_size=4,
        heads=4,
        noise_dim=32,
    ).to(device)
    criterion = SinkhornCorrectionCloudLoss(blur=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    batch = build_bridge_batch(
        clean=clean,
        schedule=schedule,
        target_samples=8,
        answer_jump=10,
    )
    predicted = model(batch.current, batch.goal, samples=8)
    loss = criterion(predicted, batch.target_cloud, batch.current)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    print(f"device: {device}")
    print(f"current levels: {batch.current_level.tolist()}")
    print(f"goal levels:    {batch.goal_level.tolist()}")
    print(f"answer levels:  {batch.answer_level.tolist()}")
    print(f"predicted cloud: {tuple(predicted.shape)}")
    print(f"target cloud:    {tuple(batch.target_cloud.shape)}")
    print(f"sinkhorn loss:   {loss.item():.6f}")
    print(f"gradient norm:   {float(gradient_norm):.6f}")


if __name__ == "__main__":
    main()

