"""Cross-market Alpha360 Transformer with independent A/US encoders.

The two markets deliberately do not share feature projections, temporal
positions, temporal Transformer weights, attention-pooling weights, or stock
identity tables.  Their independently encoded stock states are mapped into a
shared 64-dimensional token space only after a trainable market embedding has
been appended.  Cross-market self-attention can then transfer US information
to A-share tokens, while predictions are emitted for A shares only.

All Gaussian variables are log returns.  The public output mirrors
``Alpha360DecoupledTransformer``: four independent heads expose log-return
``mean``/``std`` plus ordinary-return moments and positive-return probability.
This module contains no optimizer, training loop, gradient accumulation, or
gradient clipping policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from roll.alpha360_cross_stock import HORIZON_NAMES, alpha360_flat_to_sequence
from roll.alpha360_decoupled import distribution_report


@dataclass(frozen=True)
class Alpha360CrossMarketConfig:
    """Architecture parameters for the E6 cross-market experiment."""

    model_width: int = 64
    temporal_layers: int = 2
    cross_market_layers: int = 2
    attention_heads: int = 4
    feedforward_width: int = 256
    stock_embedding_width: int = 64
    market_embedding_width: int = 16
    output_head_width: int = 128
    dropout: float = 0.10
    identity_seed: int = 20260828
    minimum_std: float = 1e-4
    target_scale: float = 100.0


class MarketTemporalEncoder(nn.Module):
    """One market's fully independent temporal and identity encoder."""

    def __init__(
        self,
        stock_count: int,
        config: Alpha360CrossMarketConfig,
        *,
        identity_seed: int,
    ) -> None:
        super().__init__()
        if stock_count < 1:
            raise ValueError("stock_count must be positive")
        if config.stock_embedding_width != 64:
            raise ValueError("stock identity embedding must be 64-dimensional")
        if config.model_width % config.attention_heads:
            raise ValueError("model_width must be divisible by attention_heads")

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
        # Learned attention pooling over the 60 historical observations.
        self.temporal_pool = nn.Linear(config.model_width, 1)

        generator = torch.Generator(device="cpu").manual_seed(identity_seed)
        identity = torch.randn(
            stock_count + 1,
            config.stock_embedding_width,
            generator=generator,
        ) * 0.02
        identity[0].zero_()
        self.stock_identity = nn.Embedding.from_pretrained(
            identity,
            freeze=False,
            padding_idx=0,
        )
        self.identity_fusion = nn.Sequential(
            nn.Linear(
                config.model_width + config.stock_embedding_width,
                config.model_width,
            ),
            nn.LayerNorm(config.model_width),
            nn.GELU(),
        )

    def forward(
        self,
        alpha360: torch.Tensor,
        stock_ids: torch.Tensor,
        stock_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, stocks, _ = alpha360.shape
        sequence = alpha360_flat_to_sequence(alpha360)
        sequence = self.feature_projection(sequence)
        sequence = sequence + self.time_position[None, None, :, :]
        sequence = sequence.reshape(batch * stocks, 60, -1)
        flat_valid = stock_mask.reshape(-1)
        # Do not run padded stock slots through either market's temporal
        # Transformer.  Besides avoiding wasted work, this makes padding inert
        # by construction rather than relying on a later overwrite.
        flat_state = sequence.new_zeros((batch * stocks, sequence.shape[-1]))
        if flat_valid.any():
            valid_sequence = self.temporal_encoder(sequence[flat_valid])
            temporal_weight = torch.softmax(
                self.temporal_pool(valid_sequence), dim=-2
            )
            valid_state = (valid_sequence * temporal_weight).sum(dim=-2)
            flat_state[flat_valid] = valid_state
        state = flat_state.reshape(batch, stocks, -1)

        identity = self.stock_identity(stock_ids.long())
        state = self.identity_fusion(torch.cat((state, identity), dim=-1))
        # Padded market tokens remain inert before entering shared attention.
        return state.masked_fill(~stock_mask.unsqueeze(-1), 0.0)


class CrossMarketGaussianHead(nn.Module):
    """One independent Gaussian MLP head for one executable horizon."""

    def __init__(self, config: Alpha360CrossMarketConfig) -> None:
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
        with torch.autocast(device_type=raw.device.type, enabled=False):
            raw = raw.float()
            mean = raw[..., 0]
            std = F.softplus(raw[..., 1]) + self.minimum_std
        return mean, std


class Alpha360CrossMarketTransformer(nn.Module):
    """E6 model: independent A/US encoders and shared cross-market attention."""

    def __init__(
        self,
        a_stock_count: int,
        us_stock_count: int,
        config: Alpha360CrossMarketConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or Alpha360CrossMarketConfig()
        cfg = self.config
        if cfg.market_embedding_width < 1:
            raise ValueError("market_embedding_width must be positive")
        if cfg.model_width % cfg.attention_heads:
            raise ValueError("model_width must be divisible by attention_heads")

        # Distinct module instances and distinct initialization streams are
        # intentional: no temporal parameter is shared across markets.
        self.a_encoder = MarketTemporalEncoder(
            a_stock_count,
            cfg,
            identity_seed=cfg.identity_seed,
        )
        self.us_encoder = MarketTemporalEncoder(
            us_stock_count,
            cfg,
            identity_seed=cfg.identity_seed + 1,
        )

        self.market_embedding = nn.Embedding(2, cfg.market_embedding_width)
        self.shared_token_projection = nn.Sequential(
            nn.Linear(cfg.model_width + cfg.market_embedding_width, cfg.model_width),
            nn.LayerNorm(cfg.model_width),
            nn.GELU(),
        )
        cross_market_layer = nn.TransformerEncoderLayer(
            d_model=cfg.model_width,
            nhead=cfg.attention_heads,
            dim_feedforward=cfg.feedforward_width,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.cross_market_encoder = nn.TransformerEncoder(
            cross_market_layer,
            num_layers=cfg.cross_market_layers,
            enable_nested_tensor=False,
        )
        self.horizon_names = HORIZON_NAMES
        self.heads = nn.ModuleDict(
            {name: CrossMarketGaussianHead(cfg) for name in self.horizon_names}
        )

    @staticmethod
    def _validate_market_inputs(
        market: str,
        alpha360: torch.Tensor,
        stock_ids: torch.Tensor,
        stock_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if alpha360.ndim != 3 or alpha360.shape[-1] != 360:
            raise ValueError(f"{market}_alpha360 must have shape [B, N, 360]")
        batch, stocks, _ = alpha360.shape
        if stocks < 1:
            raise ValueError(f"{market}_alpha360 must contain at least one stock slot")
        if stock_ids.shape != (batch, stocks):
            raise ValueError(f"{market}_stock_ids must have shape [B, N]")
        if stock_ids.device != alpha360.device:
            raise ValueError(f"{market}_stock_ids must be on the same device as features")
        if stock_mask is None:
            valid = torch.ones(
                (batch, stocks),
                dtype=torch.bool,
                device=alpha360.device,
            )
        else:
            if stock_mask.shape != (batch, stocks):
                raise ValueError(f"{market}_stock_mask must have shape [B, N]")
            valid = stock_mask.to(device=alpha360.device, dtype=torch.bool)
        ids = stock_ids.long()
        if torch.any(ids[valid] <= 0):
            raise ValueError(f"{market} valid stock IDs must be positive")
        if torch.any(ids[~valid] != 0):
            raise ValueError(f"{market} padded stock IDs must be zero")
        return valid

    def _add_market_embedding(
        self,
        state: torch.Tensor,
        market_index: int,
        stock_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, stocks, _ = state.shape
        market_ids = torch.full(
            (batch, stocks),
            market_index,
            dtype=torch.long,
            device=state.device,
        )
        market = self.market_embedding(market_ids)
        token = self.shared_token_projection(torch.cat((state, market), dim=-1))
        # Projection biases and the market embedding would otherwise make a
        # padded state non-zero again before cross-market attention.
        return token.masked_fill(~stock_mask.unsqueeze(-1), 0.0)

    def forward(
        self,
        a_alpha360: torch.Tensor,
        us_alpha360: torch.Tensor,
        a_stock_ids: torch.Tensor,
        us_stock_ids: torch.Tensor,
        a_stock_mask: torch.Tensor | None = None,
        us_stock_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | tuple[str, ...]]:
        if a_alpha360.shape[0] != us_alpha360.shape[0]:
            raise ValueError("A and US inputs must have the same batch size")
        a_valid = self._validate_market_inputs(
            "a", a_alpha360, a_stock_ids, a_stock_mask
        )
        us_valid = self._validate_market_inputs(
            "us", us_alpha360, us_stock_ids, us_stock_mask
        )
        if not torch.all(a_valid.any(dim=1)):
            raise ValueError("each date must contain at least one valid A-share")

        a_state = self.a_encoder(a_alpha360, a_stock_ids, a_valid)
        us_state = self.us_encoder(us_alpha360, us_stock_ids, us_valid)
        a_token = self._add_market_embedding(a_state, 0, a_valid)
        us_token = self._add_market_embedding(us_state, 1, us_valid)

        tokens = torch.cat((a_token, us_token), dim=1)
        valid = torch.cat((a_valid, us_valid), dim=1)
        padding_mask = ~valid

        # Avoid all -inf attention logits if an entire sample is padding.
        all_padding = padding_mask.all(dim=1)
        safe_padding_mask = padding_mask.clone()
        if all_padding.any():
            safe_padding_mask[all_padding, 0] = False
            tokens = tokens.clone()
            tokens[all_padding, 0] = 0.0

        cross_state = self.cross_market_encoder(
            tokens,
            src_key_padding_mask=safe_padding_mask,
        )
        a_state = cross_state[:, : a_alpha360.shape[1]]
        a_state = a_state.masked_fill(~a_valid.unsqueeze(-1), 0.0)

        parameters = [self.heads[name](a_state) for name in self.horizon_names]
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
