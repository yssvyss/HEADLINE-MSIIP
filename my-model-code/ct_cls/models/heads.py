from __future__ import annotations

import torch
import torch.nn as nn


class MeanPoolHead(nn.Module):
    """对所有 vision token 做 masked mean pooling，接一个多标签分类头。"""

    def __init__(self, hidden_size: int, num_labels: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, vision_tokens: torch.Tensor, vision_mask: torch.Tensor) -> torch.Tensor:
        mask = vision_mask.unsqueeze(-1).to(dtype=vision_tokens.dtype)
        pooled = (vision_tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.classifier(pooled)


class MLPPoolHead(nn.Module):
    """Masked mean pooling + 2-layer MLP classifier."""

    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        mlp_hidden_size: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        inner = mlp_hidden_size or hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, num_labels),
        )

    def forward(self, vision_tokens: torch.Tensor, vision_mask: torch.Tensor) -> torch.Tensor:
        mask = vision_mask.unsqueeze(-1).to(dtype=vision_tokens.dtype)
        pooled = (vision_tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.classifier(pooled)


class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(
        self,
        queries: torch.Tensor,
        vision_tokens: torch.Tensor,
        vision_mask: torch.Tensor,
    ) -> torch.Tensor:
        key_padding_mask = ~vision_mask
        attn_out, _ = self.cross_attn(
            query=queries,
            key=vision_tokens,
            value=vision_tokens,
            key_padding_mask=key_padding_mask,
        )
        queries = self.norm1(queries + attn_out)
        queries = self.norm2(queries + self.ffn(queries))
        return queries


class QueryAttentionHead(nn.Module):
    """Q-Former 风格：可学习 query token 作为 Q，vision token 作为 K/V，每个 query 单独分类头。"""

    def __init__(
        self,
        hidden_size: int,
        num_queries: int,
        num_labels: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_queries != num_labels:
            raise ValueError(
                f"query_attention 当前约定 num_queries == num_labels，"
                f"got num_queries={num_queries}, num_labels={num_labels}"
            )
        self.num_queries = num_queries
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, hidden_size) * 0.02)
        self.layers = nn.ModuleList(
            [
                CrossAttentionBlock(hidden_size, num_heads=num_heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.classifiers = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(num_queries)])

    def forward(self, vision_tokens: torch.Tensor, vision_mask: torch.Tensor) -> torch.Tensor:
        batch_size = vision_tokens.shape[0]
        queries = self.query_tokens.expand(batch_size, -1, -1)
        for layer in self.layers:
            queries = layer(queries, vision_tokens, vision_mask)

        logits = torch.cat([head(queries[:, i, :]) for i, head in enumerate(self.classifiers)], dim=-1)
        return logits
