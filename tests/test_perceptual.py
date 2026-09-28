import pytest
import torch
from torch import nn

from stochastic_bridge.losses import PairedPerceptualCloudLoss
from stochastic_bridge.perceptual import EfficientNetB0Features


def test_efficientnet_b0_features_are_frozen_and_multiscale() -> None:
    encoder = EfficientNetB0Features(pretrained=False)
    encoder.train()
    outputs = encoder(torch.zeros(1, 3, 128, 128))

    assert not encoder.training
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    assert [tuple(value.shape) for value in outputs] == [
        (1, 16, 64, 64),
        (1, 24, 32, 32),
        (1, 40, 16, 16),
        (1, 112, 8, 8),
        (1, 320, 4, 4),
    ]


@pytest.mark.parametrize("stages", [(), (2, 1), (1, 1), (9,)])
def test_efficientnet_b0_features_rejects_invalid_stages(stages) -> None:
    with pytest.raises(ValueError):
        EfficientNetB0Features(pretrained=False, stages=stages)


class TinyFeatures(nn.Module):
    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return image, torch.nn.functional.avg_pool2d(image, 2)


def test_paired_perceptual_loss_matches_features_and_backpropagates() -> None:
    predicted = torch.randn(2, 3, 3, 8, 8, requires_grad=True)
    target = torch.randn_like(predicted)
    criterion = PairedPerceptualCloudLoss(TinyFeatures())

    loss = criterion(predicted, target)
    loss.backward()

    assert loss.item() > 0.0
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()
    torch.testing.assert_close(criterion(target, target), torch.tensor(0.0))
