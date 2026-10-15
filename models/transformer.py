import logging
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import Config
from utils import get_device, resolve_dtype, resolve_param_dtype

logger = logging.getLogger(__name__)

# Exactly two sanctioned kernels plus "auto" (specs/ablation-run FR-2, owner decision
# 2026-10-15). The former FlashAttention backend is gone: it was misnamed "4" and called
# the FA2 `flash_attn` API. `flex_attention` stays absent -- it has no backward pass on MPS in the pinned
# PyTorch build, so it cannot train on the declared dev host (transformer-backbone D2b).
VALID_BACKENDS = ("auto", "flash_attention_3", "sdpa")

#: FlashAttention-3 kernels take half precision only.
FA3_DTYPES = (torch.bfloat16, torch.float16)


class AttentionBackendError(RuntimeError):
    """Raised when the selected attention backend cannot execute."""


def l2_normalize(t: torch.Tensor) -> torch.Tensor:
    """L2-normalize along the feature axis, in fp32 for numerical stability under bf16."""
    return F.normalize(t.float(), p=2, dim=-1).to(t.dtype)


def preferred_backend(device_type: str, capability: Optional[Tuple[int, int]]) -> str:
    """The hardware-preferred kernel (specs/ablation-run FR-2, design D2).

    Hopper CUDA (compute capability 9.x) -> FlashAttention-3. Any other CUDA GPU, MPS and
    CPU -> SDPA, which is a sanctioned kernel there, not a fallback.
    """
    if device_type == "cuda" and capability is not None and capability[0] == 9:
        return "flash_attention_3"
    return "sdpa"


# ---------------- variable-length (unpad -> kernel -> repad) core ----------------
#
# FlashAttention-3 ships no `bert_padding`, and every training batch is left-padded, so the
# varlen path is the hot path (design D2). These functions are pure torch, differentiable,
# and make no assumption about WHERE the pads are; under left padding each row's real
# tokens are contiguous and in order, which is what makes causal attention over the ragged
# buffer equal to causal attention with a key-padding mask.


@dataclass
class VarlenMeta:
    """Ragged-batch metadata, computed once per `padding_mask` object."""

    indices: Optional[torch.Tensor]  # (N_real,) int64 flat positions b*S+s; None if all_real
    cu_seqlens: torch.Tensor         # (B+1,) int32 cumulative real lengths, on the mask's device
    max_seqlen: int                  # host int, exact
    all_real: bool                   # True -> no pad anywhere; callers take the dense path


def varlen_meta(padding_mask: torch.Tensor) -> VarlenMeta:
    """Metadata for a `(B, S)` mask that is True at REAL positions.

    One host sync for the lengths (gives `max_seqlen` and `all_real`) and, only when a pad
    exists, a second for the real-token indices.
    """
    B, S = padding_mask.shape
    seqlens = padding_mask.sum(dim=1, dtype=torch.int32)
    seqlens_host = seqlens.cpu()
    max_seqlen = int(seqlens_host.max()) if B else 0
    all_real = bool((seqlens_host == S).all())
    cu_seqlens = torch.zeros(B + 1, dtype=torch.int32, device=padding_mask.device)
    cu_seqlens[1:] = seqlens.cumsum(0, dtype=torch.int32)
    indices = None
    if not all_real:
        indices = torch.nonzero(padding_mask.flatten(), as_tuple=False).flatten()
    return VarlenMeta(indices, cu_seqlens, max_seqlen, all_real)


def unpad(x_bshd: torch.Tensor, meta: VarlenMeta) -> torch.Tensor:
    """`(B, S, H, D)` -> `(N_real, H, D)`, keeping real tokens in row-major order."""
    B, S = x_bshd.shape[:2]
    return x_bshd.reshape(B * S, *x_bshd.shape[2:]).index_select(0, meta.indices)


def repad(x_un: torch.Tensor, meta: VarlenMeta, B: int, S: int) -> torch.Tensor:
    """`(N_real, H, D)` -> `(B, S, H, D)` with exact zeros at pad positions."""
    out = torch.zeros(B * S, *x_un.shape[1:], dtype=x_un.dtype, device=x_un.device)
    return out.index_copy(0, meta.indices, x_un).reshape(B, S, *x_un.shape[1:])


def _fa3_output(result):
    """FA3 versions return either `out` or `(out, softmax_lse)`."""
    if isinstance(result, (tuple, list)):
        return result[0]
    return result


def varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    padding_mask: torch.Tensor,
    is_causal: bool,
    varlen_fn: Callable,
    meta: Optional[VarlenMeta] = None,
) -> torch.Tensor:
    """Attention over a padded `(B, H, S, D)` batch through a variable-length kernel.

    `varlen_fn` is FA3's `flash_attn_varlen_func` in production and a pure-torch reference
    in tests. Causality reaches it as the `causal` flag (REV-122).
    """
    B, H, S, D = q.shape
    if meta is None:
        meta = varlen_meta(padding_mask)
    # (B, H, S, D) -> (B, S, H, D): a view back to the projections' contiguous layout.
    q_un, k_un, v_un = (unpad(t.transpose(1, 2), meta) for t in (q, k, v))
    out_un = _fa3_output(varlen_fn(
        q_un, k_un, v_un,
        cu_seqlens_q=meta.cu_seqlens, cu_seqlens_k=meta.cu_seqlens,
        max_seqlen_q=meta.max_seqlen, max_seqlen_k=meta.max_seqlen,
        causal=is_causal,
    ))
    return repad(out_un, meta, B, S).transpose(1, 2)


def _import_fa3() -> Tuple[Callable, Callable]:
    """Lazily import FlashAttention-3. Reached only from the FA3 probe/dispatch."""
    try:
        import flash_attn_interface
    except ImportError as e:
        raise AttentionBackendError(
            "FlashAttention-3 (flash_attn_interface) is not installed on this machine. "
            "It is built from the flash-attention repo's hopper/ directory for Hopper "
            "GPUs; see scripts/install_fa3.sh. Select --attention_backend sdpa elsewhere."
        ) from e
    return flash_attn_interface.flash_attn_func, flash_attn_interface.flash_attn_varlen_func


class AttentionBackend:
    """Dispatches attention to a real kernel and reports the kernel that actually ran.

    REV-043: the previous implementation resolved a backend *name* but always executed
    `F.scaled_dot_product_attention`, so run manifests recorded a kernel that never ran.
    Every backend below is now genuinely invoked, `self_test` exercises the selected
    implementation (forward *and* backward, since training needs both), and an explicitly
    requested backend that cannot execute raises instead of falling back.

    specs/ablation-run FR-2: the resolvable backends are `sdpa` (MPS, CPU, non-Hopper CUDA)
    and `flash_attention_3` (Hopper CUDA). FlexAttention is not resolvable and not
    selectable -- see `VALID_BACKENDS`.
    """

    def __init__(self, config: Config):
        self.device = get_device(config)
        self.dtype = resolve_dtype(config)
        # D2 "Probe": the startup gate exercises the kernel instance training uses.
        self.head_dim = max(1, config.hidden_dim // max(1, config.num_heads))
        self.requested_backend = config.attention_backend
        self.require_backend = getattr(config, "require_attention_backend", False)
        self.fallback_reason: Optional[str] = None
        # D2 "One metadata computation per forward": every block receives the SAME
        # padding_mask object, so its varlen metadata is computed once per forward. The
        # cache holds a strong reference, so the identity cannot be reused by another
        # mask while cached.
        self._varlen_cache: Optional[Tuple[torch.Tensor, VarlenMeta]] = None
        self.backend_name = self._resolve_backend(config.attention_backend)
        logger.info(
            "Resolved attention backend: %s on %s%s",
            self.backend_name,
            self.device,
            f" (fallback: {self.fallback_reason})" if self.fallback_reason else "",
        )

    # ---------------- backend resolution ----------------

    def _preferred_backend(self) -> str:
        capability = None
        if self.device.type == "cuda":
            capability = torch.cuda.get_device_capability(self.device)
        return preferred_backend(self.device.type, capability)

    def _resolve_backend(self, backend: str) -> str:
        if backend not in VALID_BACKENDS:
            raise ValueError(
                f"Unknown attention_backend '{backend}'. Choose from {list(VALID_BACKENDS)}"
            )

        if backend != "auto":
            # Explicit request: must execute or fail fast. This is the path the H100
            # run and the MPS gate run use, so there is no fallback here.
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
        """Run the backend's real kernel forward+backward on tiny inputs.

        REV-122: the probe covers BOTH branches -- unpadded and left-padded -- because
        every training batch is left-padded. specs/ablation-run D2: at the model's real
        head dimension (80 at t-base), not a hardcoded 16, so the gate exercises the kernel
        instance training uses.
        """
        B, H, S, D = 2, 2, 8, self.head_dim
        # The probe draws from its OWN generator: it must not consume the global RNG,
        # which model construction (weight init) and dropout read next. Otherwise the
        # initial weights would depend on the probe's shapes and on how many backends
        # `auto` probed before settling (specs/ablation-run T5.1 BUILD note).
        gen = torch.Generator(device="cpu").manual_seed(0)

        def _rand():
            t = torch.randn(B, H, S, D, generator=gen)
            return t.to(device=self.device, dtype=self.dtype).requires_grad_(True)

        try:
            for case, padding_mask in (("unpadded", None), ("left-padded", self._probe_padding_mask(B, S))):
                q, k, v = _rand(), _rand(), _rand()
                out = self._dispatch(
                    backend, q, k, v, padding_mask=padding_mask, is_causal=True,
                )
                if out.shape != (B, H, S, D):
                    return False, f"[{case}] unexpected output shape {tuple(out.shape)}"
                if not torch.isfinite(out).all():
                    return False, f"[{case}] non-finite values in attention output"
                # Training requires a backward pass through the same kernel.
                out.float().sum().backward()
                if q.grad is None or not torch.isfinite(q.grad).all():
                    return False, f"[{case}] backward produced no/!finite gradients"
        except Exception as e:  # noqa: BLE001 - probe must classify any failure
            return False, f"{type(e).__name__}: {e}"
        finally:
            self._varlen_cache = None
        return True, ""

    def _probe_padding_mask(self, B: int, S: int) -> torch.Tensor:
        """Left-padded key-padding mask (True at REAL positions), as training produces."""
        mask = torch.ones(B, S, device=self.device, dtype=torch.bool)
        mask[:, : S // 2] = False
        return mask

    # ---------------- kernels ----------------

    def _dispatch(
        self,
        backend: str,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
        is_causal: bool,
    ) -> torch.Tensor:
        # REV-122: the caller passes causal INTENT (`is_causal`) and the padding mask, and
        # each backend formats its own mask from that pair -- design D2 point 3. Causality
        # must never travel in a channel only one backend reads.
        if backend == "flash_attention_3":
            return self._flash_attention_3(q, k, v, padding_mask, is_causal)
        if backend == "sdpa":
            attn_mask = self._sdpa_additive_mask(q, padding_mask, is_causal)
            if attn_mask is not None:
                # SDPA forbids `attn_mask` and `is_causal=True` together, so the causal
                # half is folded into the additive mask here and the flag is cleared.
                return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
            return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        raise ValueError(f"Unknown attention backend '{backend}'")

    @staticmethod
    def _sdpa_additive_mask(
        q: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
        is_causal: bool,
    ) -> Optional[torch.Tensor]:
        """Build SDPA's attention mask (boolean, True = attend) from `(padding_mask, is_causal)`.

        Returns `None` when there is no key padding, so the unpadded path keeps using
        SDPA's native `is_causal` fast path. `padding_mask` is True at REAL positions.
        """
        if padding_mask is None:
            return None
        B, H, S, _ = q.shape
        blocked = ~padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S) True where padded
        if is_causal:
            causal = torch.triu(
                torch.ones(S, S, device=q.device, dtype=torch.bool), diagonal=1
            )
            blocked = blocked | causal.unsqueeze(0).unsqueeze(0)
        # A boolean (B, 1, S, S) mask (True = attend) is equivalent to the additive -inf
        # mask in every SDPA backend, but broadcasts over heads instead of materialising
        # a (B, H, S, S) float tensor per layer, and leaves SDPA free to pick its fused
        # memory-efficient kernel on CUDA.
        return ~blocked

    def _varlen_meta_for(self, padding_mask: torch.Tensor) -> VarlenMeta:
        cached = self._varlen_cache
        if cached is not None and cached[0] is padding_mask:
            return cached[1]
        meta = varlen_meta(padding_mask)
        self._varlen_cache = (padding_mask, meta)
        return meta

    def _flash_attention_3(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
        is_causal: bool,
    ) -> torch.Tensor:
        """FlashAttention-3 (`flash_attn_interface`), dense or variable-length (design D2)."""
        if q.dtype not in FA3_DTYPES:
            raise AttentionBackendError(
                f"FlashAttention-3 takes bf16/fp16 inputs only, got {q.dtype}. Under "
                f"mixed precision q/k/v come out of autocast in the compute dtype; with "
                f"pure weights use --dtype bf16."
            )
        dense_fn, varlen_fn = _import_fa3()
        if padding_mask is not None:
            meta = self._varlen_meta_for(padding_mask)
            if not meta.all_real:
                return varlen_attention(
                    q, k, v, padding_mask, is_causal, varlen_fn, meta=meta,
                )
        # FlashAttention takes (B, S, H, D); the project keeps (B, H, S, D).
        out = _fa3_output(dense_fn(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=is_causal,
        ))
        return out.transpose(1, 2)

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
        padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        """Attend with causal INTENT, not a pre-formatted mask.

        REV-122: callers pass `(padding_mask, is_causal)` and each backend formats the
        mask its kernel actually reads. There is deliberately no `attn_mask` parameter --
        an additive mask is SDPA dialect, and accepting one here is what let causality
        silently vanish on the (former) FlashAttention path.
        """
        return self._dispatch(
            self.backend_name, q, k, v,
            padding_mask=padding_mask, is_causal=is_causal,
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

        # REV-122: this attention is causal unconditionally -- `is_causal=True` on BOTH
        # branches. It previously passed `is_causal=False` on the padded branch and relied
        # on a hand-built additive mask to carry causality, which only SDPA reads; the
        # FlashAttention branch therefore attended bidirectionally on every left-padded
        # batch. Causal
        # intent now travels as the flag, and each backend formats its own mask.
        out = self.backend.forward(q, k, v, padding_mask=padding_mask, is_causal=True)

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
        dtype = resolve_param_dtype(config)

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
        # specs/ablation-run FR-7 / AC-18: a shape check (no device sync). Without it a
        # wider input indexes the position table out of range -- an opaque IndexError on
        # CPU, a context-poisoning device assert on CUDA.
        max_positions = self.position_embedding.num_embeddings
        if S > max_positions:
            raise ValueError(
                f"input has {S} positions but the model was built with "
                f"max_seq_length={max_positions}; narrow the data to the model width "
                f"(load_tensorised(..., max_seq_length=...) / load_examples(...))."
            )
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
