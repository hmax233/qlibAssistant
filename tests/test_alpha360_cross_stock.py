from __future__ import annotations

import torch

from roll.alpha360_cross_stock import (
    Alpha360CrossStockTransformer,
    Alpha360TransformerConfig,
    HORIZON_MATRIX,
    alpha360_flat_to_sequence,
    cholesky_from_raw,
    derive_horizon_distribution,
    joint_gaussian_nll,
)


def small_model() -> Alpha360CrossStockTransformer:
    return Alpha360CrossStockTransformer(
        stock_count=12,
        config=Alpha360TransformerConfig(
            model_width=16,
            temporal_layers=1,
            cross_section_layers=1,
            attention_heads=4,
            feedforward_width=32,
            stock_embedding_width=4,
            dropout=0.0,
        ),
    )


def test_alpha360_field_major_layout_is_transposed_correctly() -> None:
    flat = torch.arange(360.0).reshape(1, 360)
    sequence = alpha360_flat_to_sequence(flat)
    assert sequence.shape == (1, 60, 6)
    torch.testing.assert_close(sequence[0, 0], torch.tensor([0, 60, 120, 180, 240, 300.0]))
    torch.testing.assert_close(sequence[0, -1], torch.tensor([59, 119, 179, 239, 299, 359.0]))


def test_cholesky_builds_positive_definite_covariance() -> None:
    torch.manual_seed(17)
    factor = cholesky_from_raw(torch.randn(8, 6, dtype=torch.float64) * 2)
    covariance = factor @ factor.transpose(-1, -2)
    eigenvalues = torch.linalg.eigvalsh(covariance)
    assert torch.all(torch.diagonal(factor, dim1=-2, dim2=-1) > 0)
    assert torch.all(eigenvalues > 0)


def test_four_horizons_obey_log_return_identity() -> None:
    leg_mean = torch.tensor([[0.01, -0.02, 0.03]])
    factor = torch.eye(3).unsqueeze(0)
    horizon_mean, covariance = derive_horizon_distribution(leg_mean, factor)
    torch.testing.assert_close(horizon_mean, leg_mean @ HORIZON_MATRIX.T)
    torch.testing.assert_close(horizon_mean[..., 0] + horizon_mean[..., 1],
                               horizon_mean[..., 2] + horizon_mean[..., 3])
    assert covariance.shape == (1, 4, 4)


def test_masked_nll_is_finite_and_differentiable() -> None:
    mean = torch.randn(2, 4, 3, requires_grad=True)
    raw = torch.randn(2, 4, 6, requires_grad=True)
    target = torch.randn(2, 4, 3)
    target[0, 3] = float("nan")
    mask = torch.tensor([[True, True, True, True], [True, True, False, False]])
    loss = joint_gaussian_nll(target, mean, cholesky_from_raw(raw), mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(mean.grad).all()
    assert torch.isfinite(raw.grad).all()


def test_model_is_permutation_equivariant_and_identity_is_trainable() -> None:
    torch.manual_seed(7)
    model = small_model().eval()
    features = torch.randn(1, 5, 360)
    stock_ids = torch.tensor([[1, 2, 3, 4, 5]])
    order = torch.tensor([3, 0, 4, 1, 2])
    original = model(features, stock_ids)["leg_mean"]
    permuted = model(features[:, order], stock_ids[:, order])["leg_mean"]
    torch.testing.assert_close(permuted, original[:, order], atol=2e-6, rtol=2e-6)
    assert model.stock_identity.weight.requires_grad is True


def test_model_forward_and_backward_shapes() -> None:
    model = small_model()
    output = model(torch.randn(2, 6, 360), torch.arange(1, 7).repeat(2, 1))
    assert output["leg_mean"].shape == (2, 6, 3)
    assert output["leg_cholesky"].shape == (2, 6, 3, 3)
    assert output["horizon_mean"].shape == (2, 6, 4)
    assert output["horizon_covariance"].shape == (2, 6, 4, 4)
    target = torch.randn(2, 6, 3)
    joint_gaussian_nll(target, output["leg_mean"], output["leg_cholesky"]).backward()
    assert model.distribution_head[-1].weight.grad is not None
    assert model.stock_identity.weight.grad is not None


def test_nll_matches_pytorch_multivariate_normal() -> None:
    torch.manual_seed(9)
    mean, target = torch.randn(2, 5, 3), torch.randn(2, 5, 3)
    factor = cholesky_from_raw(torch.randn(2, 5, 6))
    reference = -torch.distributions.MultivariateNormal(mean, scale_tril=factor).log_prob(target).mean()
    torch.testing.assert_close(joint_gaussian_nll(target, mean, factor), reference)


def test_future_legs_and_boundary_purge() -> None:
    import numpy as np
    import pandas as pd
    from script.train_alpha360_cross_stock import future_log_legs, purged_signal_dates

    result = future_log_legs([100.0, 0.0], [110.0, 1.0], [105.0, 1.0], [120.0, 1.0])
    np.testing.assert_allclose(result[0], np.log([1.1, 105.0 / 110.0, 120.0 / 105.0]), rtol=1e-6)
    assert np.isnan(result[1]).all()
    horizons = np.expm1(result[0] @ HORIZON_MATRIX.numpy().T)
    np.testing.assert_allclose(horizons, [0.2, 105 / 110 - 1, 0.05, 120 / 110 - 1], rtol=1e-5)
    calendar = pd.bdate_range("2026-01-01", "2026-01-30")
    segments = {"train": ["2026-01-01", "2026-01-09"], "valid": ["2026-01-12", "2026-01-16"],
                "selection_valid": ["2026-01-19", "2026-01-23"], "test": ["2026-01-26", "2026-01-30"]}
    purged = purged_signal_dates(calendar, segments)
    assert purged["train"] == ["2026-01-08", "2026-01-09"]
    assert purged["selection_valid"] == ["2026-01-22", "2026-01-23"]
    assert "test" not in purged


def test_masked_stock_does_not_change_other_predictions() -> None:
    torch.manual_seed(12)
    model = small_model().eval()
    x = torch.randn(1, 5, 360)
    ids = torch.tensor([[1, 2, 3, 4, 0]])
    mask = torch.tensor([[True, True, True, True, False]])
    original = model(x, ids, mask)["leg_mean"]
    x[:, -1] = torch.randn_like(x[:, -1]) * 20
    changed = model(x, ids, mask)["leg_mean"]
    torch.testing.assert_close(original[:, :4], changed[:, :4], atol=1e-6, rtol=1e-6)


def test_vectorized_features_never_read_future_and_follow_qlib_layout() -> None:
    import numpy as np
    from script.train_alpha360_cross_stock import construct_alpha360

    values = np.arange(1, 481, dtype="float32").reshape(80, 6)
    features = construct_alpha360(values, [60, 61])
    assert features.shape == (2, 360)
    np.testing.assert_allclose(features[0, :60], values[1:61, 0] / values[60, 0])
    np.testing.assert_allclose(features[0, 60:120], values[1:61, 1] / values[60, 0])
    np.testing.assert_allclose(features[0, 300:360], values[1:61, 5] / values[60, 5])
    changed = values.copy()
    changed[62:] = -1e8
    np.testing.assert_array_equal(features, construct_alpha360(changed, [60, 61]))


def test_checkpoint_preserves_trainable_stock_identity(tmp_path) -> None:
    first = small_model().eval()
    path = tmp_path / "model.pt"
    torch.save(first.state_dict(), path)
    torch.manual_seed(987)
    second = small_model().eval()
    second.load_state_dict(torch.load(path, weights_only=True))
    torch.testing.assert_close(first.stock_identity.weight, second.stock_identity.weight)
    x, ids = torch.randn(1, 3, 360), torch.tensor([[1, 3, 5]])
    torch.testing.assert_close(first(x, ids)["leg_mean"], second(x, ids)["leg_mean"])


def test_nll_weights_dates_equally_not_stock_counts() -> None:
    target = torch.tensor([[[1., 1., 1.], [1., 1., 1.]], [[5., 5., 5.], [0., 0., 0.]]])
    mean = torch.zeros_like(target)
    factor = torch.eye(3).expand(2, 2, 3, 3)
    mask = torch.tensor([[True, True], [True, False]])
    actual = joint_gaussian_nll(target, mean, factor, mask)
    first = joint_gaussian_nll(target[0], mean[0], factor[0])
    second = joint_gaussian_nll(target[1, :1], mean[1, :1], factor[1, :1])
    torch.testing.assert_close(actual, (first + second) / 2)
