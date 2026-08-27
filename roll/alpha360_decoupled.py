"""Decoupled Gaussian Alpha360 models for four executable return horizons.

This module deliberately lives beside, rather than inside, the E0 joint-leg
model.  It reuses E0's canonical Alpha360 field layout and horizon names while
keeping all parameters independent, so experiments E1--E5 cannot alter E0.

The Gaussian variables are log returns.  ``expected_return`` therefore reports
the corresponding expected ordinary (simple) return under a log-normal price
ratio, while ``probability_positive`` reports ``P(log_return > 0)``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final

import torch
from torch import nn
import torch.nn.functional as F

from roll.alpha360_cross_stock import HORIZON_NAMES, alpha360_flat_to_sequence


SUPPORTED_MODES: Final = ("shared_four_head", "single_horizon")


@dataclass(frozen=True)
class Alpha360DecoupledConfig:
    """Architecture configuration shared by E1 and E2--E5."""

    model_width: int = 64
    temporal_layers: int = 2
    cross_section_layers: int = 2
    attention_heads: int = 4
    feedforward_width: int = 256
    stock_embedding_width: int = 64
    output_head_width: int = 128
    dropout: float = 0.10
    identity_seed: int = 20260827
    minimum_std: float = 1e-4
    target_scale: float = 100.0


def independent_gaussian_nll(
    target: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gaussian NLL with equal weight for every usable trading date.

    Args:
        target: Log-return labels with shape ``[B, N, H]``. Individual NaNs
            are ignored, so one unavailable horizon does not discard the other
            horizons for that stock.
        mean: Gaussian means with the same shape as ``target``.
        std: Strictly positive Gaussian standard deviations of the same shape.
        mask: Optional stock-validity mask with shape ``[B, N]``. It is applied
            to every horizon and is intended for variable-size padded batches.

    Each date first averages over all of its valid stock/horizon observations;
    the final loss then averages over dates. A date with more listed stocks can
    therefore not dominate a date with fewer stocks.
    """

    if target.ndim != 3:
        raise ValueError("target, mean, and std must have shape [B, N, H]")
    if mean.shape != target.shape or std.shape != target.shape:
        raise ValueError("target, mean, and std must have identical shapes")
    if mask is not None and mask.shape != target.shape[:2]:
        raise ValueError("mask must have shape [B, N]")
    if torch.any(std <= 0):
        raise ValueError("std must be strictly positive")

    target_f = target.float()
    mean_f = mean.float()
    std_f = std.float()
    valid = torch.isfinite(target_f)
    if mask is not None:
        valid &= mask.bool().unsqueeze(-1)

    # Substituting detached means makes invalid residuals exactly zero without
    # allowing NaNs to contaminate the graph before masking.
    safe_target = torch.where(valid, target_f, mean_f.detach())
    standardized = (safe_target - mean_f) / std_f
    per_observation = (
        0.5 * standardized.square()
        + torch.log(std_f)
        + 0.5 * math.log(2.0 * math.pi)
    )
    per_observation = torch.where(valid, per_observation, 0.0)

    valid_per_date = valid.sum(dim=(1, 2))
    usable_dates = valid_per_date > 0
    if not usable_dates.any():
        # Return a differentiable zero so a fully missing minibatch is safe.
        return (mean_f.sum() + std_f.sum()) * 0.0
    date_loss = per_observation.sum(dim=(1, 2)) / valid_per_date.clamp_min(1)
    return date_loss[usable_dates].mean()


def distribution_report(mean: torch.Tensor, std: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return useful distribution statistics for Gaussian log-return outputs."""

    if mean.shape != std.shape:
        raise ValueError("mean and std must have identical shapes")
    if torch.any(std <= 0):
        raise ValueError("std must be strictly positive")
    mean_f, std_f = mean.float(), std.float()
    variance = std_f.square()
    probability_positive = 0.5 * (
        1.0 + torch.erf(mean_f / std_f / math.sqrt(2.0))
    )
    expected_return = torch.exp(mean_f + 0.5 * variance) - 1.0
    ordinary_variance = (torch.exp(variance) - 1.0) * torch.exp(
        2.0 * mean_f + variance
    )
    return {
        "log_mean": mean_f,
        "log_std": std_f,
        "expected_return": expected_return,
        "return_std": ordinary_variance.clamp_min(0.0).sqrt(),
        "probability_positive": probability_positive,
    }


class Alpha360DecoupledBackbone(nn.Module):
    """Temporal attention pooling followed by stock-set self-attention."""

    def __init__(self, stock_count: int, config: Alpha360DecoupledConfig):
        super().__init__()
        if stock_count < 1:
            raise ValueError("stock_count must be positive")
        if config.model_width % config.attention_heads:
            raise ValueError("model_width must be divisible by attention_heads")
        if config.stock_embedding_width != 64:
            raise ValueError("stock identity embedding must be 64-dimensional")
        self.config = config

        self.feature_projection = nn.Linear(6, config.model_width)
        self.time_position = nn.Parameter(torch.empty(60, config.model_width))
        nn.init.normal_(self.time_position, std=0.02)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=config.model_width,
            nhead=config.attention_heads,
            dim_feedforward=config.feedforward_width,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer,
            num_layers=config.temporal_layers,
            enable_nested_tensor=False,
        )
        # Learned attention pooling, deliberately not mean pooling.
        self.temporal_pool = nn.Linear(config.model_width, 1)

        generator = torch.Generator(device="cpu").manual_seed(config.identity_seed)
        identity = torch.randn(
            stock_count + 1,
            config.stock_embedding_width,
            generator=generator,
        ) * 0.02
        identity[0].zero_()
        self.stock_identity = nn.Embedding.from_pretrained(
            identity, freeze=False, padding_idx=0
        )
        self.identity_fusion = nn.Sequential(
            nn.Linear(
                config.model_width + config.stock_embedding_width,
                config.model_width,
            ),
            nn.LayerNorm(config.model_width),
            nn.GELU(),
        )
        cross_section_layer = nn.TransformerEncoderLayer(
            d_model=config.model_width,
            nhead=config.attention_heads,
            dim_feedforward=config.feedforward_width,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.cross_section_encoder = nn.TransformerEncoder(
            cross_section_layer,
            num_layers=config.cross_section_layers,
            enable_nested_tensor=False,
        )

    def forward(
        self,
        alpha360: torch.Tensor,
        stock_ids: torch.Tensor,
        stock_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if alpha360.ndim != 3 or alpha360.shape[-1] != 360:
            raise ValueError("alpha360 must have shape [B, N, 360]")
        batch, stocks, _ = alpha360.shape
        if stock_ids.shape != (batch, stocks):
            raise ValueError("stock_ids must have shape [B, N]")
        if stock_mask is not None and stock_mask.shape != (batch, stocks):
            raise ValueError("stock_mask must have shape [B, N]")

        sequence = alpha360_flat_to_sequence(alpha360)
        sequence = self.feature_projection(sequence)
        sequence = sequence + self.time_position[None, None, :, :]
        sequence = sequence.reshape(batch * stocks, 60, -1)
        sequence = self.temporal_encoder(sequence)
        temporal_weights = torch.softmax(self.temporal_pool(sequence), dim=-2)
        stock_state = (sequence * temporal_weights).sum(dim=-2)
        stock_state = stock_state.reshape(batch, stocks, -1)

        identity = self.stock_identity(stock_ids.long())
        stock_state = self.identity_fusion(torch.cat((stock_state, identity), dim=-1))

        if stock_mask is None:
            valid = torch.ones(
                (batch, stocks), dtype=torch.bool, device=stock_state.device
            )
        else:
            valid = stock_mask.bool()
        padding_mask = ~valid
        # PyTorch attention has undefined softmax when an entire date is masked.
        # Temporarily expose one zero token, then zero the complete date again.
        all_padding = padding_mask.all(dim=1)
        safe_padding_mask = padding_mask.clone()
        if all_padding.any():
            safe_padding_mask[all_padding, 0] = False
            stock_state = stock_state.clone()
            stock_state[all_padding, 0] = 0.0
        stock_state = self.cross_section_encoder(
            stock_state, src_key_padding_mask=safe_padding_mask
        )
        return stock_state.masked_fill(~valid.unsqueeze(-1), 0.0)


class IndependentGaussianHead(nn.Module):
    """Explicit MLP that maps a stock state to one Gaussian mean/std pair."""

    def __init__(self, config: Alpha360DecoupledConfig):
        super().__init__()
        self.minimum_std = config.minimum_std
        self.network = nn.Sequential(
            nn.LayerNorm(config.model_width),
            nn.Linear(config.model_width, config.output_head_width),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.output_head_width, 2),
        )

    @property
    def final_linear(self) -> nn.Linear:
        return self.network[-1]

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.network(state)
        # Distribution arithmetic remains FP32 under mixed-precision encoders.
        with torch.autocast(device_type=raw.device.type, enabled=False):
            raw = raw.float()
            mean = raw[..., 0]
            std = F.softplus(raw[..., 1]) + self.minimum_std
        return mean, std


class Alpha360DecoupledTransformer(nn.Module):
    """E1 shared-four-head model or an E2--E5 single-horizon model."""

    def __init__(
        self,
        stock_count: int,
        mode: str = "shared_four_head",
        horizon: str | None = None,
        config: Alpha360DecoupledConfig | None = None,
    ):
        super().__init__()
        if mode not in SUPPORTED_MODES:
            raise ValueError(f"mode must be one of {SUPPORTED_MODES}, got {mode!r}")
        if mode == "shared_four_head" and horizon is not None:
            raise ValueError("shared_four_head does not accept a single horizon")
        if mode == "single_horizon" and horizon not in HORIZON_NAMES:
            raise ValueError(f"single_horizon requires one of {HORIZON_NAMES}")

        self.mode = mode
        self.horizon = horizon
        self.config = config or Alpha360DecoupledConfig()
        self.backbone = Alpha360DecoupledBackbone(stock_count, self.config)
        selected_horizons = HORIZON_NAMES if mode == "shared_four_head" else (horizon,)
        self.horizon_names = tuple(selected_horizons)
        self.heads = nn.ModuleDict(
            {name: IndependentGaussianHead(self.config) for name in self.horizon_names}
        )

    def forward(
        self,
        alpha360: torch.Tensor,
        stock_ids: torch.Tensor,
        stock_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | tuple[str, ...]]:
        state = self.backbone(alpha360, stock_ids, stock_mask)
        parameters = [self.heads[name](state) for name in self.horizon_names]
        mean = torch.stack([item[0] for item in parameters], dim=-1)
        std = torch.stack([item[1] for item in parameters], dim=-1)
        report = distribution_report(mean, std)
        return {
            "mean": mean,
            "std": std,
            "expected_return": report["expected_return"],
            "return_std": report["return_std"],
            "probability_positive": report["probability_positive"],
            "horizon_names": self.horizon_names,
        }
