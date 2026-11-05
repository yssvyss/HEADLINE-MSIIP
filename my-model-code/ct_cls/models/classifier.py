from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ct_cls.config import ModelConfig
from ct_cls.models.heads import MeanPoolHead, MLPPoolHead, QueryAttentionHead
from ct_cls.models.vision_encoder import (
    encode_slice_2d_tokens,
    encode_video_temporal_pool_tokens,
    encode_video_tokens,
    load_qwen_vision,
    resolve_dtype,
)


class CTVisionClassifier(nn.Module):
    """Qwen3-VL vision encoder + classification head (no LLM)."""

    def __init__(
        self,
        vision_encoder: nn.Module,
        head: nn.Module,
        token_dim: int,
        dtype: torch.dtype,
        vision_mode: str = "video",
        slice_token_pool: str = "mean",
    ) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.head = head
        self.token_dim = token_dim
        self.dtype = dtype
        self.vision_mode = vision_mode
        self.slice_token_pool = slice_token_pool
        self.num_labels: int | None = None

    @property
    def config(self):
        """Expose vision encoder config for Accelerate/DeepSpeed auto bucket sizing."""
        return self.vision_encoder.config

    def build_head(self, cfg: ModelConfig, num_labels: int) -> None:
        self.num_labels = num_labels
        self.vision_mode = cfg.vision_mode
        self.slice_token_pool = cfg.slice_token_pool
        if cfg.head_type == "mean_pool":
            self.head = MeanPoolHead(self.token_dim, num_labels)
        elif cfg.head_type == "mlp_pool":
            self.head = MLPPoolHead(
                self.token_dim,
                num_labels,
                mlp_hidden_size=cfg.head_mlp_hidden_size,
                dropout=cfg.head_dropout,
            )
        elif cfg.head_type == "query_attention":
            num_queries = cfg.num_queries or num_labels
            self.head = QueryAttentionHead(
                hidden_size=self.token_dim,
                num_queries=num_queries,
                num_labels=num_labels,
                num_layers=cfg.query_num_layers,
                num_heads=cfg.query_num_heads,
                dropout=cfg.query_dropout,
            )
        else:
            raise ValueError(f"Unsupported head_type: {cfg.head_type}")

        self.head.to(dtype=self.dtype)

    def encode_tokens(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        slice_counts: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.vision_mode == "video":
            return encode_video_tokens(self.vision_encoder, pixel_values, grid_thw)
        if self.vision_mode == "video_temporal_pool":
            return encode_video_temporal_pool_tokens(self.vision_encoder, pixel_values, grid_thw)
        if self.vision_mode == "slice_2d":
            if slice_counts is None:
                raise ValueError("slice_2d mode requires slice_counts")
            return encode_slice_2d_tokens(
                self.vision_encoder,
                pixel_values,
                grid_thw,
                slice_counts,
                slice_token_pool=self.slice_token_pool,
            )
        raise ValueError(f"Unsupported vision_mode: {self.vision_mode}")

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        slice_counts: list[int] | None = None,
    ) -> torch.Tensor:
        vision_tokens, vision_mask = self.encode_tokens(pixel_values, grid_thw, slice_counts)
        return self.head(vision_tokens, vision_mask)


def build_model(cfg: ModelConfig, num_labels: int | None = None) -> tuple[nn.Module, Any]:
    if cfg.vision_backend not in {"qwen3_vl", "qwen3_5"}:
        raise ValueError(f"Unsupported vision_backend for inference: {cfg.vision_backend}")

    vision_encoder, processor, token_dim, dtype = load_qwen_vision(cfg)
    model = CTVisionClassifier(
        vision_encoder=vision_encoder,
        head=nn.Identity(),
        token_dim=token_dim,
        dtype=dtype,
        vision_mode=cfg.vision_mode,
        slice_token_pool=cfg.slice_token_pool,
    )
    if num_labels is not None:
        model.build_head(cfg, num_labels)
    return model, processor
