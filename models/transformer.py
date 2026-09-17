import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import Config
from utils import get_device, resolve_dtype

logger = logging.getLogger(__name__)

# T1.3 / AC-09: exactly two sanctioned kernels plus "auto". `flex_attention` is
# deliberately absent -- it has no backward pass on MPS in the pinned PyTorch build, so
# it cannot train on the declared dev host (FR-4 rationale, design D2b).
VALID_BACKENDS = ("auto", "flash_attention_4", "sdpa")


class AttentionBackendError(RuntimeError):
    """Raised when the selected attention backend cannot execute."""


def l2_normalize(t: torch.Tensor) -> torch.Tensor:
    """L2-normalize along the feature axis, in fp32 for numerical stability under bf16."""
    return F.normalize(t.float(), p=2, dim=-1).to(t.dtype)


class AttentionBackend:
    """Dispatches attention to a real kernel and reports the kernel that actually ran.

    REV-043: the previous implementation resolved a backend *name* but always executed
    `F.scaled_dot_product_attention`, so run manifests recorded a kernel that never ran.
    Every backend below is now genuinely invoked, `self_test` exercises the selected
    implementation (forward *and* backward, since training needs both), and an explicitly
    requested backend that cannot execute raises instead of falling back.

    REV-062 / T1.3: the resolvable backends are `sdpa` (MPS/CPU) and `flash_attention_4`
    (CUDA). SDPA is the sanctioned local/dev training kernel, not a degraded fallback.
    FlexAttention is not resolvable and not selectable -- see `VALID_BACKENDS`.
    """

    def __init__(self, config: Config):
        self.device = get_device(config)
        self.dtype = resolve_dtype(config)
        self.requested_backend = config.attention_backend
        self.require_backend = getattr(config, "require_attention_backend", False)
        self.fallback_reason: Optional[str] = None
        self.backend_name = self._resolve_backend(config.attention_backend)
        logger.info(
            "Resolved attention backend: %s on %s%s",
            self.backend_name,
            self.device,
            f" (fallback: {self.fallback_reason})" if self.fallback_reason else "",
        )

    # ---------------- backend resolution ----------------

    def _preferred_backend(self) -> str:
        if self.device.type == "cuda":
            return "flash_attention_4"
        # T1.3: MPS and CPU both resolve to SDPA. MPS used to prefer FlexAttention, which
        # could not run a backward pass here, so the armed gate failed on the dev host.
        return "sdpa"

    def _resolve_backend(self, backend: str) -> str:
        if backend not in VALID_BACKENDS:
            raise ValueError(
                f"Unknown attention_backend '{backend}'. Choose from {list(VALID_BACKENDS)}"
            )

        if backend != "auto":
            # Explicit request: must execute or fail fast. This is the path the H100
            # final run and the MPS gate run use, so there is no fallback here.
            ok, reason = self._probe(backend)
            if not ok:
                raise AttentionBackendError(
                    f"Attention backend '{backend}' was explicitly requested but cannot "
                    f"execute on {self.device}: {reason}\n"
                    f"Refusing to fall back to another kernel."
                )
            return backend

        preferred = self._preferred_backend()
        ok, reason = self._probe(preferred)
        if ok:
            return preferred

        if self.require_backend:
            raise AttentionBackendError(
                f"Preferred attention backend '{preferred}' cannot execute on "
                f"{self.device}: {reason}\n"
                f"require_attention_backend=True forbids falling back."
            )

        self.fallback_reason = (
            f"'{preferred}' unavailable on {self.device}: {reason}"
        )
        logger.warning(
            "ATTENTION BACKEND FALLBACK: %s. Falling back to 'sdpa'. "
            "The run manifest will record 'sdpa' as the backend that actually ran. "
            "Set attention_backend explicitly or require_attention_backend=True to "
            "turn this into a hard failure.",
            self.fallback_reason,
        )
        ok, sdpa_reason = self._probe("sdpa")
        if not ok:
            raise AttentionBackendError(
                f"No usable attention backend on {self.device}: {sdpa_reason}"
            )
        return "sdpa"

    def _probe(self, backend: str) -> Tuple[bool, str]:
        """Run the backend's real kernel forward+backward on a tiny input."""
        B, H, S, D = 2, 2, 8, 16
        try:
            q = torch.randn(B, H, S, D, device=self.device, dtype=self.dtype, requires_grad=True)
            k = torch.randn(B, H, S, D, device=self.device, dtype=self.dtype, requires_grad=True)
            v = torch.randn(B, H, S, D, device=self.device, dtype=self.dtype, requires_grad=True)
            out = self._dispatch(backend, q, k, v, attn_mask=None, padding_mask=None, is_causal=True)
            if out.shape != (B, H, S, D):
                return False, f"unexpected output shape {tuple(out.shape)}"
            if not torch.isfinite(out).all():
                return False, "non-finite values in attention output"
            # Training requires a backward pass through the same kernel.
            out.float().sum().backward()
            if q.grad is None or not torch.isfinite(q.grad).all():
                return False, "backward produced no/!finite gradients"
        except Exception as e:  # noqa: BLE001 - probe must classify any failure
            return False, f"{type(e).__name__}: {e}"
        return True, ""

    # ---------------- kernels ----------------

    def _dispatch(
        self,
        backend: str,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        padding_mask: Optional[torch.Tensor],
        is_causal: bool,
    ) -> torch.Tensor:
        if backend == "flash_attention_4":
            return self._flash_attention_4(q, k, v, padding_mask, is_causal)
        if backend == "sdpa":
            if attn_mask is not None:
                return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
            return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        raise ValueError(f"Unknown attention backend '{backend}'")

    @staticmethod
    def _flash_attention_4(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
        is_causal: bool,
    ) -> torch.Tensor:
        # FlashAttention expects (B, S, H, D); the project keeps (B, H, S, D).
        try:
            from flash_attn import flash_attn_func
        except ImportError as e:
            raise AttentionBackendError(
                "FlashAttention-4 is not installed. Install the flash-attn package "
                "matching this CUDA/PyTorch build, or select another backend."
            ) from e

        # REV-081: this used to RAISE whenever `padding_mask` had any masked position.
        # `CausalSelfAttention.forward` passes the mask whenever the batch contains a pad
        # token, and every training batch is left-padded, so the first real batch on CUDA
        # aborted the run -- the H100 path (FR-4, AC-03) could not train at all. Route
        # padded batches through the variable-length API instead: unpad to a single
        # ragged buffer with cumulative sequence lengths, attend, then repad.
        #
        # Left-padding makes this exact: every real token of a row is contiguous and in
        # order, so removing the pads preserves the causal order within each sequence.
        if padding_mask is not None and not bool(padding_mask.all()):
            return AttentionBackend._flash_attention_4_varlen(
                q, k, v, padding_mask, is_causal,
            )

        out = flash_attn_func(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=is_causal,
        )
        return out.transpose(1, 2)

    @staticmethod
    def _flash_attention_4_varlen(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        padding_mask: torch.Tensor,
        is_causal: bool,
    ) -> torch.Tensor:
        """FlashAttention over a padded batch, via the variable-length API (REV-081).

        `padding_mask` is True at REAL positions, which is the convention
        `flash_attn.bert_padding.unpad_input` expects.
        """
        try:
            from flash_attn import flash_attn_varlen_func
            from flash_attn.bert_padding import pad_input, unpad_input
        except ImportError as e:
            raise AttentionBackendError(
                "FlashAttention's variable-length API is required for padded batches "
                "but is unavailable. Install a flash-attn build providing "
                "`flash_attn_varlen_func` and `flash_attn.bert_padding`."
            ) from e

        B, H, S, D = q.shape
        # (B, H, S, D) -> (B, S, H, D), which is what the varlen helpers expect.
        q_bshd, k_bshd, v_bshd = (t.transpose(1, 2) for t in (q, k, v))

        q_un, indices, cu_seqlens, max_seqlen = unpad_input(q_bshd, padding_mask)[:4]
        k_un = unpad_input(k_bshd, padding_mask)[0]
        v_un = unpad_input(v_bshd, padding_mask)[0]

        out_un = flash_attn_varlen_func(
            q_un, k_un, v_un,
            cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
            causal=is_causal,
        )

        out = pad_input(out_un, indices, B, S)  # (B, S, H, D)
        return out.transpose(1, 2).contiguous()

    # ---------------- public API ----------------

    def self_test(self) -> dict:
        ok, reason = self._probe(self.backend_name)
        if not ok:
            raise AttentionBackendError(
                f"Attention backend self-test FAILED on {self.device} "
                f"with backend '{self.backend_name}': {reason}\n"
                f"Cannot proceed with training."
            )
        return {
            "backend": self.backend_name,
            "requested_backend": self.requested_backend,
            "fallback_reason": self.fallback_reason,
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
        padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        return self._dispatch(
            self.backend_name, q, k, v,
            attn_mask=attn_mask, padding_mask=padding_mask, is_causal=is_causal,
        )


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
        self.attn_dropout = nn.Dropout(dropout)

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
            out = self.backend.forward(
                q, k, v, attn_mask=attn_float, padding_mask=padding_mask, is_causal=False,
            )
        else:
            out = self.backend.forward(q, k, v, padding_mask=None, is_causal=True)

        out = out.transpose(1, 2).contiguous().view(B, S, D)
        # REV-051: attention-weight dropout was stored but never applied. SDPA's
        # `dropout_p` is not usable uniformly across backends, so it is applied to the
        # attention output here instead, which keeps the regularization backend-agnostic.
        return self.out_proj(self.attn_dropout(out))


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
    # REV-102: guards the once-per-process warning when a metadata tensor reaches a
    # model built without a metadata projection.
    _warned_metadata_ignored = False

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

        # REV-050: T2.2 requires validating the metadata feature dimension. Previously a
        # `use_metadata=True, feature_dim=0` config silently produced a metadata-free
        # model that still reported use_metadata=True in its snapshot.
        self.use_metadata = config.use_metadata
        self.feature_dim = config.feature_dim
        if config.use_metadata:
            if config.feature_dim <= 0:
                raise ValueError(
                    "config.use_metadata=True but config.feature_dim="
                    f"{config.feature_dim}. feature_dim is set after preprocessing loads "
                    "item_metadata.npy; load it (or set use_metadata=False) before "
                    "building the model."
                )
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

        if metadata is not None and self.metadata_projection is None:
            # REV-102: requirements.md Edge Cases and AC-06 require that a metadata
            # tensor supplied to a `use_metadata=False` model "must not affect
            # computation". Ignore it, but say so once per process so the mismatch is
            # not actually silent.
            if not TransformerModel._warned_metadata_ignored:
                TransformerModel._warned_metadata_ignored = True
                logger.warning(
                    "A metadata tensor was supplied to a model built without a metadata "
                    "projection (config.use_metadata=False); it is ignored and the "
                    "output is identical to the metadata-free forward (AC-06). This "
                    "warning is emitted once per process."
                )
            metadata = None

        if metadata is not None:
            if metadata.shape[-1] != self.feature_dim:
                raise ValueError(
                    f"metadata last dimension {metadata.shape[-1]} does not match "
                    f"config.feature_dim {self.feature_dim}"
                )
            x = x + self.metadata_projection(metadata.to(x.dtype))

        x = self.embed_dropout(x)

        padding_mask = (seq != 0)
        has_padding = (~padding_mask).any()

        for block in self.blocks:
            x = block(x, padding_mask if has_padding else None)

        x = self.final_norm(x)

        if return_all_hidden:
            # REV-044: hidden states are returned un-normalized on purpose -- they feed
            # the MoE layer. L2 normalization is applied at the scoring sites
            # (`score_candidates` and the loss functions), which is where dot products
            # actually happen and where FR-8's [-1, 1] bound must hold.
            return x

        pooled = x[:, -1, :]

        if self.normalize_embeddings:
            pooled = l2_normalize(pooled)

        return pooled
