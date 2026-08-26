"""Factorized temporal/cross-sectional Transformer for Alpha360.

The network predicts a joint Gaussian over three consecutive future log-return
legs.  Four executable holding-period distributions are derived algebraically,
so their means and covariances cannot contradict one another.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


HORIZON_MATRIX = torch.tensor(
    [
        [1.0, 1.0, 1.0],  # T+1 open  -> T+2 close
        [0.0, 1.0, 0.0],  # T+1 close -> T+2 open
        [1.0, 1.0, 0.0],  # T+1 open  -> T+2 open
        [0.0, 1.0, 1.0],  # T+1 close -> T+2 close
    ],
    dtype=torch.float32,
)

HORIZON_NAMES = (
    "open1_close2",
    "close1_open2",
    "open1_open2",
    "close1_close2",
)


def alpha360_flat_to_sequence(values: torch.Tensor) -> torch.Tensor:
    """Convert Qlib's field-major Alpha360 columns to ``[..., 60, 6]``.

    Qlib emits CLOSE59..0, OPEN59..0, HIGH59..0, LOW59..0, VWAP59..0,
    VOLUME59..0.  A blind reshape to ``[60, 6]`` silently mixes fields.
    """

    if values.shape[-1] != 360:
        raise ValueError(f"Alpha360 requires 360 columns, got {values.shape[-1]}")
    return values.reshape(*values.shape[:-1], 6, 60).transpose(-1, -2)


def cholesky_from_raw(raw: torch.Tensor, minimum_diagonal: float = 1e-3) -> torch.Tensor:
    """Build a valid 3x3 Cholesky factor from six unrestricted outputs."""

    if raw.shape[-1] != 6:
        raise ValueError(f"Expected six covariance parameters, got {raw.shape[-1]}")
    factor = raw.new_zeros(*raw.shape[:-1], 3, 3)
    factor[..., 0, 0] = F.softplus(raw[..., 0]) + minimum_diagonal
    factor[..., 1, 0] = raw[..., 1]
    factor[..., 1, 1] = F.softplus(raw[..., 2]) + minimum_diagonal
    factor[..., 2, 0] = raw[..., 3]
    factor[..., 2, 1] = raw[..., 4]
    factor[..., 2, 2] = F.softplus(raw[..., 5]) + minimum_diagonal
    return factor


def joint_gaussian_nll(
    target: torch.Tensor,
    mean: torch.Tensor,
    cholesky: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean per-date Gaussian NLL, with each date weighted equally."""

    target = target.float()
    mean = mean.float()
    cholesky = cholesky.float()
    valid = torch.isfinite(target).all(dim=-1)
    if mask is not None:
        valid &= mask.bool()
    safe_target = torch.where(valid[..., None], target, mean.detach())
    residual = (safe_target - mean).unsqueeze(-1)
    whitened = torch.linalg.solve_triangular(cholesky, residual, upper=False)
    mahalanobis = whitened.square().sum(dim=(-2, -1))
    log_determinant = 2.0 * torch.log(
        torch.diagonal(cholesky, dim1=-2, dim2=-1)
    ).sum(dim=-1)
    per_stock = 0.5 * (mahalanobis + log_determinant + 3.0 * math.log(2.0 * math.pi))
    per_stock = torch.where(valid, per_stock, 0.0)
    if per_stock.ndim == 1:
        denominator = valid.sum().clamp_min(1)
        return per_stock.sum() / denominator
    date_loss = per_stock.sum(dim=-1) / valid.sum(dim=-1).clamp_min(1)
    usable_dates = valid.any(dim=-1)
    if not usable_dates.any():
        return mean.sum() * 0.0
    return date_loss[usable_dates].mean()


def derive_horizon_distribution(
    leg_mean: torch.Tensor,
    leg_cholesky: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return four log-return means and their 4x4 covariance matrix."""

    matrix = HORIZON_MATRIX.to(device=leg_mean.device, dtype=leg_mean.dtype)
    leg_covariance = leg_cholesky @ leg_cholesky.transpose(-1, -2)
    horizon_mean = torch.einsum("hc,...c->...h", matrix, leg_mean)
    horizon_covariance = torch.einsum(
        "hi,...ij,kj->...hk", matrix, leg_covariance, matrix
    )
    return horizon_mean, horizon_covariance


def distribution_report(
    horizon_mean: torch.Tensor,
    horizon_covariance: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Convert log-normal parameters to ordinary-return moments/probability."""

    variance = torch.diagonal(horizon_covariance, dim1=-2, dim2=-1).clamp_min(1e-12)
    standard_deviation = variance.sqrt()
    expected_return = torch.exp(horizon_mean + 0.5 * variance) - 1.0
    ordinary_variance = (torch.exp(variance) - 1.0) * torch.exp(
        2.0 * horizon_mean + variance
    )
    probability_positive = 0.5 * (
        1.0 + torch.erf(horizon_mean / standard_deviation / math.sqrt(2.0))
    )
    return {
        "log_mean": horizon_mean,
        "log_variance": variance,
        "expected_return": expected_return,
        "return_std": ordinary_variance.clamp_min(0.0).sqrt(),
        "probability_positive": probability_positive,
    }


@dataclass(frozen=True)
class Alpha360TransformerConfig:
    model_width: int = 64
    temporal_layers: int = 2
    cross_section_layers: int = 2
    attention_heads: int = 4
    feedforward_width: int = 256
    stock_embedding_width: int = 16
    dropout: float = 0.10
    identity_seed: int = 20260827
    target_scale: float = 100.0


class Alpha360CrossStockTransformer(nn.Module):
    """Shared temporal encoder followed by permutation-equivariant stock attention."""

    def __init__(self, stock_count: int, config: Alpha360TransformerConfig | None = None):
        super().__init__()
        self.config = config or Alpha360TransformerConfig()
        cfg = self.config
        if cfg.model_width % cfg.attention_heads:
            raise ValueError("model_width must be divisible by attention_heads")
        self.feature_projection = nn.Linear(6, cfg.model_width)
        self.time_position = nn.Parameter(torch.empty(60, cfg.model_width))
        nn.init.normal_(self.time_position, std=0.02)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=cfg.model_width,
            nhead=cfg.attention_heads,
            dim_feedforward=cfg.feedforward_width,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer, num_layers=cfg.temporal_layers, enable_nested_tensor=False
        )
        self.temporal_pool = nn.Linear(cfg.model_width, 1)

        generator = torch.Generator(device="cpu").manual_seed(cfg.identity_seed)
        identity = torch.randn(stock_count + 1, cfg.stock_embedding_width, generator=generator) * 0.02
        identity[0].zero_()  # reserved for unknown/padding
        self.stock_identity = nn.Embedding.from_pretrained(identity, freeze=True, padding_idx=0)
        self.identity_fusion = nn.Sequential(
            nn.Linear(cfg.model_width + cfg.stock_embedding_width, cfg.model_width),
            nn.LayerNorm(cfg.model_width),
            nn.GELU(),
        )
        cross_layer = nn.TransformerEncoderLayer(
            d_model=cfg.model_width,
            nhead=cfg.attention_heads,
            dim_feedforward=cfg.feedforward_width,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.cross_section_encoder = nn.TransformerEncoder(
            cross_layer, num_layers=cfg.cross_section_layers, enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(cfg.model_width)
        self.distribution_head = nn.Linear(cfg.model_width, 9)

    def forward(
        self,
        alpha360: torch.Tensor,
        stock_ids: torch.Tensor,
        stock_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if alpha360.ndim != 3:
            raise ValueError("alpha360 must have shape [batch, stocks, 360]")
        batch, stocks, _ = alpha360.shape
        sequence = alpha360_flat_to_sequence(alpha360)
        sequence = self.feature_projection(sequence)
        sequence = sequence + self.time_position[None, None, :, :]
        sequence = sequence.reshape(batch * stocks, 60, -1)
        sequence = self.temporal_encoder(sequence)
        pool_weights = torch.softmax(self.temporal_pool(sequence), dim=-2)
        stock_state = (sequence * pool_weights).sum(dim=-2).reshape(batch, stocks, -1)
        identity = self.stock_identity(stock_ids.long())
        stock_state = self.identity_fusion(torch.cat([stock_state, identity], dim=-1))
        padding_mask = None if stock_mask is None else ~stock_mask.bool()
        stock_state = self.cross_section_encoder(
            stock_state, src_key_padding_mask=padding_mask
        )
        raw = self.distribution_head(self.output_norm(stock_state))
        # Keep small variances/covariances and the probability head in float32
        # even when the encoders use CUDA BF16/FP16 autocast.
        with torch.autocast(device_type=raw.device.type, enabled=False):
            leg_mean = raw[..., :3].float()
            leg_cholesky = cholesky_from_raw(raw[..., 3:].float())
            horizon_mean, horizon_covariance = derive_horizon_distribution(
                leg_mean, leg_cholesky
            )
        return {
            "leg_mean": leg_mean,
            "leg_cholesky": leg_cholesky,
            "horizon_mean": horizon_mean,
            "horizon_covariance": horizon_covariance,
        }
