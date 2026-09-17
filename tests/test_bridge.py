from __future__ import annotations

import torch
import pytest

from stochastic_bridge import (
    EnergyCorrectionCloudLoss,
    PairedFullBandCloudLoss,
    PairedCorrectionLoss,
    SinkhornCorrectionCloudLoss,
    StochasticImageBridge,
    VPNoiseSchedule,
    sample_level_triplet,
)
from stochastic_bridge.corruptions import GoalDetailCorruptor
from stochastic_bridge.data import build_bridge_batch


def test_level_zero_is_clean() -> None:
    schedule = VPNoiseSchedule(steps=100)
    clean = torch.randn(3, 3, 8, 8)
    level = torch.zeros(3, dtype=torch.long)
    actual = schedule.q_sample(clean, level)
    torch.testing.assert_close(actual, clean)


def test_level_triplet_rule() -> None:
    current, goal, answer = sample_level_triplet(
        batch_size=128, max_level=100, answer_jump=10
    )
    assert torch.all(current > goal)
    assert torch.all(answer == (goal - 10).clamp_min(0))


def test_clean_answer_cloud_is_a_point_mass() -> None:
    schedule = VPNoiseSchedule(steps=100)
    clean = torch.randn(2, 3, 8, 8)
    current_level = torch.tensor([100, 70], dtype=torch.long)
    current = schedule.q_sample(clean, current_level)
    answer_level = torch.zeros(2, dtype=torch.long)
    cloud = schedule.sample_answer_cloud(
        clean, current, current_level, answer_level, samples=5
    )
    expected = clean[:, None].expand_as(cloud)
    torch.testing.assert_close(cloud, expected, atol=1e-6, rtol=1e-6)


def test_sinkhorn_is_small_for_identical_clouds() -> None:
    torch.manual_seed(3)
    criterion = SinkhornCorrectionCloudLoss(blur=0.1)
    current = torch.zeros(2, 3, 8, 8)
    cloud = torch.randn(2, 4, 3, 8, 8) * 0.2
    identical = criterion(cloud, cloud, current)
    shifted = criterion(cloud + 0.5, cloud, current)
    assert identical.abs().item() < 1e-5
    assert shifted.item() > identical.item() + 1e-3


def test_model_and_sinkhorn_backward() -> None:
    torch.manual_seed(11)
    model = StochasticImageBridge(
        image_size=8,
        dim=32,
        patch_size=4,
        heads=4,
        noise_dim=16,
    )
    current = torch.randn(2, 3, 8, 8).clamp(-1.0, 1.0)
    goal = torch.randn(2, 3, 8, 8).clamp(-1.0, 1.0)
    target = torch.randn(2, 4, 3, 8, 8).clamp(-1.0, 1.0)
    predicted = model(current, goal, samples=4)
    assert predicted.shape == target.shape

    loss = SinkhornCorrectionCloudLoss(blur=0.1)(predicted, target, current)
    loss.backward()
    gradient_sum = sum(
        parameter.grad.abs().sum().item()
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    assert torch.isfinite(loss)
    assert gradient_sum > 0.0


def test_cross_attention_condition_predicts_trainable_scalar_noise_variance() -> None:
    torch.manual_seed(13)
    model = StochasticImageBridge(
        in_channels=3,
        base_channels=8,
        heads=4,
        attention_depth=1,
        noise_variance_min=1e-4,
        noise_variance_max=1.0,
        noise_variance_init=0.1,
    )
    current = torch.randn(2, 3, 16, 16).clamp(-1.0, 1.0)
    goal = torch.randn(2, 3, 16, 16).clamp(-1.0, 1.0)
    noise = torch.randn(2, 3, 3, 16, 16)

    output = model(
        current,
        goal,
        samples=3,
        noise=noise,
        return_noise_variance=True,
    )
    assert isinstance(output, tuple)
    cloud, noise_variance = output
    assert cloud.shape == (2, 3, 3, 16, 16)
    assert noise_variance.shape == (2,)
    torch.testing.assert_close(noise_variance, torch.full_like(noise_variance, 0.1))

    # There is no direct lambda label: an ordinary cloud objective must be able
    # to update the variance head through the reparameterized noise path.
    target = torch.randn_like(cloud).clamp(-1.0, 1.0)
    torch.nn.functional.mse_loss(cloud, target).backward()
    variance_gradient = sum(
        parameter.grad.abs().sum().item()
        for parameter in model.noise_variance_head.parameters()
        if parameter.grad is not None
    )
    assert variance_gradient > 0.0


def test_training_batch_contains_reparameterization_noise() -> None:
    torch.manual_seed(19)
    schedule = VPNoiseSchedule(steps=100)
    clean = torch.randn(2, 3, 16, 16).clamp(-1.0, 1.0)
    batch = build_bridge_batch(
        clean,
        schedule,
        target_samples=3,
        goal_corruptor=GoalDetailCorruptor(
            blur_probability=1.0,
            downsample_probability=1.0,
            mask_probability=1.0,
        ),
    )
    mean, variance = schedule.posterior_mean_variance(
        clean, batch.current, batch.current_level, batch.answer_level
    )
    expected = mean[:, None] + variance.sqrt()[:, None] * batch.target_noise
    torch.testing.assert_close(batch.target_cloud, expected)
    assert batch.goal.shape == clean.shape
    assert torch.all(batch.current_level > batch.goal_level)


def test_paired_and_energy_losses_backpropagate() -> None:
    torch.manual_seed(23)
    current = torch.zeros(2, 3, 8, 8)
    target = torch.randn(2, 3, 3, 8, 8).clamp(-1.0, 1.0)
    predicted = (target.detach().clone() + 0.1).requires_grad_(True)
    paired = PairedCorrectionLoss()(predicted, target, current)
    energy = EnergyCorrectionCloudLoss()(predicted, target, current)
    (paired + energy).backward()
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()


def test_paired_loss_does_not_average_normalized_features_twice() -> None:
    current = torch.zeros(1, 3, 16, 16)
    target = torch.zeros(1, 2, 3, 16, 16)
    predicted = torch.full_like(target, 0.1)
    criterion = PairedCorrectionLoss()

    current_features = criterion.features(current)
    predicted_features = criterion.features(predicted.flatten(0, 1)).reshape(1, 2, -1)
    target_features = criterion.features(target.flatten(0, 1)).reshape(1, 2, -1)
    predicted_correction = predicted_features - current_features[:, None]
    target_correction = target_features - current_features[:, None]
    old_double_mean = torch.nn.functional.smooth_l1_loss(
        predicted_correction, target_correction
    )

    actual = criterion(predicted, target, current)
    # D_total=(3*4^2 + 3*8^2 + 3*16^2)=1008 and there are three scales.
    torch.testing.assert_close(actual, old_double_mean * (1008 / 3))


def test_full_band_paired_loss_detects_checkerboard_null_space() -> None:
    current = torch.zeros(1, 3, 64, 64)
    target = torch.zeros(1, 2, 3, 64, 64)
    yy, xx = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
    checkerboard = ((xx + yy).remainder(2) * 2 - 1).float() * 0.1
    predicted = checkerboard[None, None, None].expand_as(target).clone()

    pooled = PairedCorrectionLoss()(predicted, target, current)
    full_band = PairedFullBandCloudLoss(levels=3)(predicted, target, current)

    # Every 4x4-or-larger average is zero, but the full-resolution high band
    # must retain and penalize the alternating phase pattern.
    assert pooled.abs().item() < 1e-8
    assert full_band.item() > 1e-2


def test_full_band_paired_loss_is_zero_for_identical_cloud_and_backpropagates() -> None:
    torch.manual_seed(29)
    current = torch.randn(2, 3, 16, 16)
    target = torch.randn(2, 3, 3, 16, 16).clamp(-1.0, 1.0)
    predicted = (target.detach().clone() + 0.05).requires_grad_(True)
    criterion = PairedFullBandCloudLoss(levels=2)

    identical = criterion(target, target, current)
    shifted = criterion(predicted, target, current)
    shifted.backward()

    torch.testing.assert_close(identical, torch.zeros_like(identical))
    assert shifted.item() > 0.0
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()


@pytest.mark.parametrize("encoder_type", ["cnn", "vit"])
def test_both_shared_encoder_types(encoder_type: str) -> None:
    model = StochasticImageBridge(
        in_channels=3,
        base_channels=8,
        heads=4,
        attention_depth=1,
        encoder_type=encoder_type,
        vit_depth=1,
    )
    current = torch.randn(1, 3, 32, 32)
    goal = torch.randn(1, 3, 32, 32)
    output = model(current, goal, samples=2)
    assert output.shape == (1, 2, 3, 32, 32)
    assert model.encoder_type == encoder_type


@pytest.mark.parametrize("encoder_type", ["cnn", "vit"])
def test_implicit_coordinate_decoder_supports_both_encoders(encoder_type: str) -> None:
    torch.manual_seed(31)
    model = StochasticImageBridge(
        in_channels=3,
        base_channels=8,
        heads=4,
        attention_depth=1,
        encoder_type=encoder_type,
        vit_depth=1,
        decoder_type="implicit",
        implicit_hidden_dim=32,
        implicit_depth=2,
        implicit_fourier_bands=3,
        implicit_chunk_size=257,
    )
    current = torch.randn(2, 3, 16, 16).clamp(-1.0, 1.0)
    goal = torch.randn(2, 3, 16, 16).clamp(-1.0, 1.0)
    noise = torch.randn(2, 2, 3, 16, 16)

    output, variance = model(
        current,
        goal,
        samples=2,
        noise=noise,
        return_noise_variance=True,
    )
    loss = PairedFullBandCloudLoss(levels=2)(
        output, torch.zeros_like(output), current
    )
    loss.backward()

    assert output.shape == (2, 2, 3, 16, 16)
    assert variance.shape == (2,)
    assert model.decoder_type == "implicit"
    assert not any(
        isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d))
        for module in model.implicit_decoder.modules()
    )
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum().item() > 0.0
        for parameter in model.implicit_decoder.parameters()
    )
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum().item() > 0.0
        for parameter in model.noise_variance_head.parameters()
    )
