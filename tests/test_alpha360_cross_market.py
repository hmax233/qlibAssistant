from __future__ import annotations

import torch

from roll.alpha360_cross_market import (
    Alpha360CrossMarketConfig,
    Alpha360CrossMarketTransformer,
)
from roll.alpha360_cross_stock import HORIZON_NAMES
from roll.alpha360_decoupled import independent_gaussian_nll


def small_config() -> Alpha360CrossMarketConfig:
    return Alpha360CrossMarketConfig(
        model_width=16,
        temporal_layers=1,
        cross_market_layers=1,
        attention_heads=4,
        feedforward_width=32,
        stock_embedding_width=64,
        market_embedding_width=8,
        output_head_width=24,
        dropout=0.0,
    )


def model() -> Alpha360CrossMarketTransformer:
    return Alpha360CrossMarketTransformer(12, 15, small_config())


def sample_inputs() -> tuple[torch.Tensor, ...]:
    a_features = torch.randn(2, 4, 360)
    us_features = torch.randn(2, 6, 360)
    a_ids = torch.tensor([[1, 2, 3, 0], [4, 5, 6, 7]])
    us_ids = torch.tensor([[1, 2, 3, 4, 0, 0], [5, 6, 7, 8, 9, 10]])
    a_mask = torch.tensor([[True, True, True, False], [True] * 4])
    us_mask = torch.tensor(
        [[True, True, True, True, False, False], [True] * 6]
    )
    return a_features, us_features, a_ids, us_ids, a_mask, us_mask


def test_output_shapes_positive_std_and_compatible_report() -> None:
    network = model().eval()
    output = network(*sample_inputs())
    for key in (
        "mean",
        "std",
        "expected_return",
        "return_std",
        "probability_positive",
    ):
        assert output[key].shape == (2, 4, 4)
    assert output["horizon_names"] == HORIZON_NAMES
    assert torch.all(output["std"] > 0)
    assert torch.all((output["probability_positive"] >= 0.0) &
                     (output["probability_positive"] <= 1.0))


def test_a_and_us_temporal_encoders_share_no_parameters() -> None:
    network = model()
    a_parameters = dict(network.a_encoder.named_parameters())
    us_parameters = dict(network.us_encoder.named_parameters())
    assert a_parameters.keys() == us_parameters.keys()
    for name in a_parameters:
        assert a_parameters[name] is not us_parameters[name]
        assert a_parameters[name].data_ptr() != us_parameters[name].data_ptr()
    assert network.a_encoder.feature_projection is not network.us_encoder.feature_projection
    assert network.a_encoder.temporal_encoder is not network.us_encoder.temporal_encoder
    assert network.a_encoder.temporal_pool is not network.us_encoder.temporal_pool
    assert network.a_encoder.time_position is not network.us_encoder.time_position


def test_market_and_stock_identity_embeddings_are_trainable() -> None:
    network = model()
    assert network.market_embedding.weight.requires_grad
    assert network.market_embedding.num_embeddings == 2
    for encoder in (network.a_encoder, network.us_encoder):
        assert encoder.stock_identity.weight.requires_grad
        assert encoder.stock_identity.embedding_dim == 64


def test_masked_stocks_cannot_change_valid_a_predictions() -> None:
    torch.manual_seed(11)
    network = model().eval()
    inputs = list(sample_inputs())
    original = network(*inputs)
    inputs[0][~inputs[4]] = torch.randn_like(inputs[0][~inputs[4]]) * 1000
    inputs[1][~inputs[5]] = torch.randn_like(inputs[1][~inputs[5]]) * 1000
    changed = network(*inputs)
    for key in ("mean", "std", "expected_return", "probability_positive"):
        torch.testing.assert_close(
            original[key][inputs[4]],
            changed[key][inputs[4]],
            atol=2e-6,
            rtol=2e-6,
        )


def test_a_stock_permutation_equivariance() -> None:
    torch.manual_seed(12)
    network = model().eval()
    inputs = sample_inputs()
    order = torch.tensor([2, 0, 3, 1])
    original = network(*inputs)
    permuted = network(
        inputs[0][:, order],
        inputs[1],
        inputs[2][:, order],
        inputs[3],
        inputs[4][:, order],
        inputs[5],
    )
    for key in ("mean", "std", "expected_return", "probability_positive"):
        torch.testing.assert_close(
            permuted[key],
            original[key][:, order],
            atol=2e-6,
            rtol=2e-6,
        )


def test_us_stock_permutation_invariance_for_a_outputs() -> None:
    torch.manual_seed(13)
    network = model().eval()
    inputs = sample_inputs()
    order = torch.tensor([5, 2, 0, 4, 1, 3])
    original = network(*inputs)
    permuted = network(
        inputs[0],
        inputs[1][:, order],
        inputs[2],
        inputs[3][:, order],
        inputs[4],
        inputs[5][:, order],
    )
    for key in ("mean", "std", "expected_return", "probability_positive"):
        torch.testing.assert_close(
            permuted[key], original[key], atol=2e-6, rtol=2e-6
        )


def test_unmasked_us_features_affect_a_outputs_through_attention() -> None:
    torch.manual_seed(14)
    network = model().eval()
    inputs = list(sample_inputs())
    original = network(*inputs)["mean"]
    inputs[1] = inputs[1].clone()
    inputs[1][:, :4] += 8.0
    changed = network(*inputs)["mean"]
    assert (changed - original).abs().max().item() > 1e-6


def test_fully_masked_us_date_is_safe_and_feature_independent() -> None:
    torch.manual_seed(15)
    network = model().eval()
    inputs = list(sample_inputs())
    inputs[5] = inputs[5].clone()
    inputs[5][0] = False
    original = network(*inputs)
    assert torch.isfinite(original["mean"]).all()
    assert torch.isfinite(original["std"]).all()

    inputs[1] = inputs[1].clone()
    inputs[1][0] = torch.randn_like(inputs[1][0]) * 1000
    changed = network(*inputs)
    torch.testing.assert_close(
        original["mean"][0], changed["mean"][0], atol=2e-6, rtol=2e-6
    )
    torch.testing.assert_close(
        original["std"][0], changed["std"][0], atol=2e-6, rtol=2e-6
    )


def test_backward_reaches_both_encoders_embeddings_attention_and_heads() -> None:
    torch.manual_seed(16)
    network = model()
    inputs = sample_inputs()
    output = network(*inputs)
    target = torch.randn(2, 4, 4)
    loss = independent_gaussian_nll(
        target,
        output["mean"],
        output["std"],
        inputs[4],
    )
    loss.backward()

    assert network.a_encoder.feature_projection.weight.grad is not None
    assert network.us_encoder.feature_projection.weight.grad is not None
    assert network.a_encoder.stock_identity.weight.grad is not None
    assert network.us_encoder.stock_identity.weight.grad is not None
    assert network.market_embedding.weight.grad is not None
    cross_parameter = next(network.cross_market_encoder.parameters())
    assert cross_parameter.grad is not None
    for head in network.heads.values():
        assert head.final_linear.weight.grad is not None


def test_different_a_and_us_stock_counts_and_all_default_masks() -> None:
    network = model().eval()
    output = network(
        torch.randn(1, 3, 360),
        torch.randn(1, 7, 360),
        torch.tensor([[1, 2, 3]]),
        torch.tensor([[1, 2, 3, 4, 5, 6, 7]]),
    )
    assert output["mean"].shape == (1, 3, 4)
    assert torch.isfinite(output["std"]).all()
