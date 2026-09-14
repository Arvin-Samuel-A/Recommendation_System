# Code Quality

## Dtype Discipline

- All model and data computation uses `torch.bfloat16`. No FP16, no autocast, no GradScaler.
- `resolve_dtype(config)` must raise if dtype is not `"bf16"`.
- Never silently change dtype if a numerical issue occurs — fail explicitly.

## Device Handling

- Always use `get_device(config)` → `torch.device`. Never call `.cuda()` or `.to("cuda")` directly.
- `config.device = "auto"` resolves CUDA > MPS > CPU.
- Device synchronization is mandatory before latency timing: `torch.mps.synchronize()` on MPS, `torch.cuda.synchronize(device)` on CUDA.

## Attention Backend Discipline

- MPS → FlexAttention. CUDA/H100 → FlashAttention-4. `"auto"` resolves by hardware.
- Each backend must pass a startup self-test before training.
- Final H100 run must fail fast if FA4 cannot initialize — no silent fallback.
- MPS smoke test must fail fast if FlexAttention cannot execute — no silent fallback.
- Record PyTorch version, device, dtype, and resolved attention backend in every run manifest.

## Configuration as Single Source of Truth

- `configs/config.py` (`Config` dataclass) holds all hyperparameters.
- No hardcoded hyperparameters in model/training/evaluation code.
- CLI flags override config values; log which values were overridden and original vs. new.

## Numerical Safety

- Assert finite losses, finite GradNorm loss, and finite active lambdas on every training step.
- Gradient clipping: `clip_grad_norm_(model.parameters(), max_norm=1.0)` on every step.
- `optimizer.zero_grad(set_to_none=True)` — use `set_to_none` for memory efficiency.

## Error Handling

- Missing Kaggle credentials with no local dataset → fail with clear auth message; never substitute another dataset.
- Missing checkpoint at evaluate stage → print skip message with remedy command, return — never raise exception.
- Negative pool exhaustion → fail with clear dataset/configuration error.
- `contrastive_interval = 0` is invalid; use `use_contrastive = False` to disable.

## Anti-Patterns (Forbidden)

- No external recommendation libraries (RecBole, Surprise, pytorch-fm, etc.)
- No sinusoidal positional embeddings (learnable `nn.Embedding` only)
- No GradScaler or FP16 mixed-precision policy
- No `num_items` as mask token (that collides with the last real item ID)
- No test data for model selection, threshold tuning, or diagnostic decisions
- No silent dtype/backend substitution
- No worker initialization when `num_workers=0`
- No per-epoch negative pool regeneration
- No cross-model state sharing between independently trained models
