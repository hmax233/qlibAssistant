from __future__ import annotations

import math

import pytest
import torch

from roll.alpha360_decoupled import (
    Alpha360DecoupledConfig,
    Alpha360DecoupledTransformer,
    distribution_report,
    independent_gaussian_nll,
)
from roll.alpha360_cross_stock import HORIZON_NAMES


def small_config() -> Alpha360DecoupledConfig:
    return Alpha360DecoupledConfig(
        model_width=16,
        temporal_layers=1,
        cross_section_layers=1,
        attention_heads=4,
        feedforward_width=32,
        stock_embedding_width=64,
        output_head_width=24,
        dropout=0.0,
    )


def shared_model() -> Alpha360DecoupledTransformer:
    return Alpha360DecoupledTransformer(
        stock_count=12, mode="shared_four_head", config=small_config()
    )


def test_shared_four_head_shapes_positive_std_and_reports() -> None:
    model = shared_model().eval()
    output = model(torch.randn(2, 5, 360), torch.arange(1, 6).repeat(2, 1))
    assert output["mean"].shape == (2, 5, 4)
    assert output["std"].shape == (2, 5, 4)
    assert output["expected_return"].shape == (2, 5, 4)
    assert output["return_std"].shape == (2, 5, 4)
    assert output["probability_positive"].shape == (2, 5, 4)
    assert output["horizon_names"] == HORIZON_NAMES
    assert torch.all(output["std"] > 0)
    assert torch.all((output["probability_positive"] >= 0) &
                     (output["probability_positive"] <= 1))


def test_distribution_report_matches_closed_form_values() -> None:
    mean = torch.tensor([[[0.0, math.log(1.1)]]])
    std = torch.tensor([[[1.0, 0.2]]])
    report = distribution_report(mean, std)
    expected = torch.exp(mean + 0.5 * std.square()) - 1.0
    torch.testing.assert_close(report["expected_return"], expected)
    torch.testing.assert_close(
        report["probability_positive"][..., 0], torch.tensor([[0.5]])
    )


def test_nll_ignores_padding_and_individual_nan_labels() -> None:
    target = torch.tensor(
        [[[0.1, float("nan")], [99.0, 99.0]], [[0.3, -0.1], [0.2, 0.4]]]
    )
    mean = torch.zeros_like(target, requires_grad=True)
    std = torch.ones_like(target, requires_grad=True)
    mask = torch.tensor([[True, False], [True, True]])
    actual = independent_gaussian_nll(target, mean, std, mask)

    constant = 0.5 * math.log(2.0 * math.pi)
    date0 = 0.5 * 0.1**2 + constant
    values1 = torch.tensor([0.3, -0.1, 0.2, 0.4])
    date1 = (0.5 * values1.square() + constant).mean()
    torch.testing.assert_close(actual, (torch.tensor(date0) + date1) / 2)
    actual.backward()
    assert torch.isfinite(mean.grad).all()
    assert torch.isfinite(std.grad).all()
    assert mean.grad[0, 1].abs().sum() == 0


def test_nll_weights_dates_equally_instead_of_observations() -> None:
    target = torch.tensor(
        [[[1.0], [1.0], [1.0]], [[5.0], [float("nan")], [float("nan")]]]
    )
    mean = torch.zeros_like(target)
    std = torch.ones_like(target)
    actual = independent_gaussian_nll(target, mean, std)
    first = independent_gaussian_nll(target[:1], mean[:1], std[:1])
    second = independent_gaussian_nll(target[1:], mean[1:], std[1:])
    torch.testing.assert_close(actual, (first + second) / 2)


def test_fully_missing_batch_returns_differentiable_zero() -> None:
    mean = torch.randn(2, 3, 4, requires_grad=True)
    raw_std = torch.randn(2, 3, 4, requires_grad=True)
    std = torch.nn.functional.softplus(raw_std) + 1e-4
    target = torch.full_like(mean, float("nan"))
    loss = independent_gaussian_nll(target, mean, std)
    assert loss.item() == 0.0
    loss.backward()
    assert mean.grad is not None
    assert raw_std.grad is not None


def test_model_backward_reaches_backbone_identity_and_all_heads() -> None:
    torch.manual_seed(5)
    model = shared_model()
    features = torch.randn(2, 4, 360)
    ids = torch.tensor([[1, 2, 3, 0], [4, 5, 6, 7]])
    mask = torch.tensor([[True, True, True, False], [True, True, True, True]])
    output = model(features, ids, mask)
    target = torch.randn(2, 4, 4)
    loss = independent_gaussian_nll(target, output["mean"], output["std"], mask)
    loss.backward()
    assert model.backbone.feature_projection.weight.grad is not None
    assert model.backbone.stock_identity.weight.grad is not None
    assert model.backbone.stock_identity.weight.requires_grad
    assert model.backbone.stock_identity.embedding_dim == 64
    for head in model.heads.values():
        assert head.final_linear.weight.grad is not None


def test_stock_permutation_equivariance() -> None:
    torch.manual_seed(8)
    model = shared_model().eval()
    features = torch.randn(2, 5, 360)
    ids = torch.tensor([[1, 2, 3, 4, 0], [5, 6, 7, 8, 9]])
    mask = torch.tensor([[True, True, True, True, False], [True] * 5])
    order = torch.tensor([3, 0, 4, 1, 2])
    original = model(features, ids, mask)
    permuted = model(features[:, order], ids[:, order], mask[:, order])
    for key in ("mean", "std", "expected_return", "probability_positive"):
        torch.testing.assert_close(
            permuted[key], original[key][:, order], atol=2e-6, rtol=2e-6
        )


def test_masked_stock_cannot_affect_valid_predictions() -> None:
    torch.manual_seed(11)
    model = shared_model().eval()
    features = torch.randn(1, 5, 360)
    ids = torch.tensor([[1, 2, 3, 4, 0]])
    mask = torch.tensor([[True, True, True, True, False]])
    original = model(features, ids, mask)
    features[:, -1] = torch.randn_like(features[:, -1]) * 100
    changed = model(features, ids, mask)
    for key in ("mean", "std"):
        torch.testing.assert_close(
            original[key][:, :4], changed[key][:, :4], atol=1e-6, rtol=1e-6
        )


def test_four_heads_have_independent_final_linear_layers() -> None:
    model = shared_model()
    final_layers = [model.heads[name].final_linear for name in HORIZON_NAMES]
    assert len({id(layer) for layer in final_layers}) == 4
    assert len({layer.weight.data_ptr() for layer in final_layers}) == 4
    assert all(layer.out_features == 2 for layer in final_layers)


@pytest.mark.parametrize("horizon", HORIZON_NAMES)
def test_single_horizon_is_complete_one_head_model(horizon: str) -> None:
    model = Alpha360DecoupledTransformer(
        stock_count=10,
        mode="single_horizon",
        horizon=horizon,
        config=small_config(),
    )
    output = model(torch.randn(2, 3, 360), torch.tensor([[1, 2, 3], [4, 5, 6]]))
    assert output["mean"].shape == (2, 3, 1)
    assert output["std"].shape == (2, 3, 1)
    assert output["horizon_names"] == (horizon,)
    assert tuple(model.heads) == (horizon,)
    independent_gaussian_nll(
        torch.randn(2, 3, 1), output["mean"], output["std"]
    ).backward()
    assert model.heads[horizon].final_linear.weight.grad is not None


def test_invalid_single_horizon_and_non_64_identity_are_rejected() -> None:
    with pytest.raises(ValueError, match="requires one of"):
        Alpha360DecoupledTransformer(10, mode="single_horizon", horizon="bad")
    with pytest.raises(ValueError, match="64-dimensional"):
        Alpha360DecoupledTransformer(
            10,
            config=Alpha360DecoupledConfig(stock_embedding_width=16),
        )

