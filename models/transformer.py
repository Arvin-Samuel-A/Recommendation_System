import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import Config
from utils import get_device, resolve_dtype

logger = logging.getLogger(__name__)


class AttentionBackend:
    def __init__(self, config: Config):
        self.device = get_device(config)
        self.dtype = resolve_dtype(config)
        self.backend_name = self._resolve_backend(config.attention_backend)
        logger.info("Resolved attention backend: %s on %s", self.backend_name, self.device)

    def _resolve_backend(self, backend: str) -> str:
        if backend == "auto":
            if self.device.type == "cuda":
                return "flash_attention_4"
            elif self.device.type == "mps":
                return "flex_attention"
            else:
                return "sdpa"
        return backend

    def self_test(self) -> dict:
        B, H, S, D = 2, 4, 8, 16
        q = torch.randn(B, H, S, D, device=self.device, dtype=self.dtype)
        k = torch.randn(B, H, S, D, device=self.device, dtype=self.dtype)
        v = torch.randn(B, H, S, D, device=self.device, dtype=self.dtype)

        try:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            assert out.shape == (B, H, S, D), f"Unexpected shape: {out.shape}"
            assert torch.isfinite(out).all(), "Non-finite values in attention output"
        except Exception as e:
            raise RuntimeError(
                f"Attention backend self-test FAILED on {self.device} "
                f"with backend '{self.backend_name}': {e}\n"
                f"Cannot proceed with training."
            ) from e

        return {
            "backend": self.backend_name,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "status": "passed",
        }

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        if attn_mask is not None:
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, is_causal=False
            )
        return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)


class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float, backend: AttentionBackend):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim
        self.backend = backend

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        if padding_mask is not None:
            causal = torch.triu(
                torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1
            )
            pad_key = ~padding_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = ~(causal.unsqueeze(0) | pad_key)
            attn_mask = attn_mask.expand(B, self.num_heads, S, S)
            attn_float = torch.zeros_like(attn_mask, dtype=x.dtype)
            attn_float.masked_fill_(~attn_mask, float("-inf"))
            out = self.backend.forward(q, k, v, attn_mask=attn_float, is_causal=False)
        else:
            out = self.backend.forward(q, k, v, is_causal=True)

        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float, ffn_expansion: int, backend: AttentionBackend):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = CausalSelfAttention(hidden_dim, num_heads, dropout, backend)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(hidden_dim)
        ffn_dim = hidden_dim * ffn_expansion
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.dropout1(self.attn(self.norm1(x), padding_mask))
        x = x + self.dropout2(self.ffn(self.norm2(x)))
        return x


class TransformerModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_items = config.num_items
        self.vocab_size = config.num_items + 2

        device = get_device(config)
        dtype = resolve_dtype(config)

        self.backend = AttentionBackend(config)

        self.item_embedding = nn.Embedding(self.vocab_size, config.hidden_dim)
        self.position_embedding = nn.Embedding(config.max_seq_length, config.hidden_dim)
        self.embed_dropout = nn.Dropout(config.dropout)

        if config.use_metadata and config.feature_dim > 0:
            self.metadata_projection = nn.Linear(config.feature_dim, config.hidden_dim, bias=False)
        else:
            self.metadata_projection = None

        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.hidden_dim, config.num_heads, config.dropout,
                config.ffn_expansion, self.backend,
            )
            for _ in range(config.num_layers)
        ])

        self.final_norm = nn.LayerNorm(config.hidden_dim)
        self.normalize_embeddings = config.normalize_embeddings

        self.to(dtype=dtype, device=device)

    @property
    def item_emb_weight(self) -> torch.Tensor:
        return self.item_embedding.weight

    def forward(
        self,
        seq: torch.Tensor,
        metadata: Optional[torch.Tensor] = None,
        return_all_hidden: bool = False,
    ) -> torch.Tensor:
        B, S = seq.shape
        positions = torch.arange(S, device=seq.device)
        x = self.item_embedding(seq) + self.position_embedding(positions)

        if self.metadata_projection is not None and metadata is not None:
            x = x + self.metadata_projection(metadata.to(x.dtype))

        x = self.embed_dropout(x)

        padding_mask = (seq != 0)
        has_padding = (~padding_mask).any()

        for block in self.blocks:
            x = block(x, padding_mask if has_padding else None)

        x = self.final_norm(x)

        if return_all_hidden:
            return x

        pooled = x[:, -1, :]

        if self.normalize_embeddings:
            pooled = F.normalize(pooled, p=2, dim=-1)

        return pooled
