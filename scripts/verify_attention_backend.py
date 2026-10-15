"""Verify an attention backend on THIS machine: self-test, then numeric parity vs fp32 SDPA.

specs/ablation-run FR-3 / design D3. Builds `models.transformer.AttentionBackend` with the
backend named explicitly (no fallback), runs its startup self-test, then for every head
dimension and sequence length compares the backend's bf16 output and q/k/v gradients against
an fp32 SDPA reference with a causal key-padding mask, on a left-padded batch (rows of real
length S, S/2 and 1) and on an unpadded batch. Only real positions are compared.

Writes a JSON report and exits
    0  every case within tolerance
    1  a parity failure (the backend runs but disagrees with the reference)
    2  the backend is unavailable on this machine (e.g. FlashAttention-3 not installed)

Examples:
    uv run python scripts/verify_attention_backend.py --backend sdpa --model t-base
    python scripts/verify_attention_backend.py --backend flash_attention_3 --model t-base \\
        --report logs/verify_fa3.json
"""
import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from configs.config import MODEL_CONFIGS, Config  # noqa: E402
from models.transformer import AttentionBackend, AttentionBackendError  # noqa: E402

#: bf16 against an fp32 reference (D3); recorded in the report.
ATOL = 2e-2
RTOL = 2e-2

EXIT_OK, EXIT_PARITY, EXIT_UNAVAILABLE = 0, 1, 2


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--backend", required=True, choices=["sdpa", "flash_attention_3"])
    p.add_argument("--model", default="t-base", choices=sorted(MODEL_CONFIGS))
    p.add_argument("--head-dims", default="64,80",
                   help="Comma-separated head dimensions to check (default 64,80).")
    p.add_argument("--seq-lens", default="50,200",
                   help="Comma-separated sequence lengths to check (default 50,200).")
    p.add_argument("--device", default="auto", help="auto (CUDA > MPS > CPU), cuda, mps, cpu")
    p.add_argument("--report", default=None,
                   help="JSON report path (default logs/verify_attention_<backend>.json).")
    return p.parse_args(argv)


def _ints(text):
    return [int(x) for x in str(text).split(",") if x.strip()]


def _environment(device):
    cuda = device.type == "cuda"
    return {
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if cuda
            else platform.processor() or platform.machine() or device.type
        ),
        "capability": list(torch.cuda.get_device_capability(device)) if cuda else None,
        "python": platform.python_version(),
    }


def _backend_module(backend_name):
    if backend_name != "flash_attention_3":
        return {"name": "torch.nn.functional.scaled_dot_product_attention"}
    import flash_attn_interface

    return {
        "name": "flash_attn_interface",
        "file": getattr(flash_attn_interface, "__file__", None),
        "version": getattr(flash_attn_interface, "__version__", None),
    }


def _reference(q, k, v, padding_mask):
    """fp32 SDPA, causal, with key padding (True = real)."""
    S = q.shape[2]
    blocked = torch.triu(torch.ones(S, S, dtype=torch.bool, device=q.device), diagonal=1)
    blocked = blocked[None, None]
    if padding_mask is not None:
        blocked = blocked | ~padding_mask[:, None, None, :]
    return F.scaled_dot_product_attention(q, k, v, attn_mask=~blocked)


def run_case(backend, device, num_heads, head_dim, seq_len, padded, seed=0):
    gen = torch.Generator(device="cpu").manual_seed(seed + 7919 * head_dim + seq_len)
    B, H, S, D = 3, num_heads, seq_len, head_dim
    lengths = [S, max(1, S // 2), 1] if padded else [S, S, S]
    mask = torch.zeros(B, S, dtype=torch.bool)
    for i, length in enumerate(lengths):
        mask[i, S - length:] = True
    mask = mask.to(device)
    base = [torch.randn(B, H, S, D, generator=gen) for _ in range(3)]
    weight = torch.randn(B, H, S, D, generator=gen).to(device)
    real = mask[:, None, :, None].expand(B, H, S, D)
    weight = weight * real  # only real positions feed the "loss"

    q, k, v = (t.to(device=device, dtype=torch.bfloat16).requires_grad_(True) for t in base)
    out = backend.forward(q, k, v, padding_mask=mask if padded else None, is_causal=True)
    (out.float() * weight).sum().backward()

    rq, rk, rv = (
        t.detach().to(device=device, dtype=torch.float32).requires_grad_(True)
        for t in (q, k, v)
    )
    ref = _reference(rq, rk, rv, mask if padded else None)
    (ref * weight).sum().backward()

    out_diff = (out.float() - ref).abs()[real].max().item()
    grad_diffs = []
    ok = bool(torch.allclose(out.float()[real], ref[real], atol=ATOL, rtol=RTOL))
    for got, exp in ((q.grad, rq.grad), (k.grad, rk.grad), (v.grad, rv.grad)):
        got_r, exp_r = got.float()[real], exp[real]
        grad_diffs.append((got_r - exp_r).abs().max().item())
        ok = ok and bool(torch.allclose(got_r, exp_r, atol=ATOL, rtol=RTOL))
        ok = ok and bool(torch.isfinite(got_r).all())
    return {
        "head_dim": head_dim,
        "seq_len": seq_len,
        "padded": padded,
        "lengths": lengths,
        "max_abs_out_diff": out_diff,
        "max_abs_grad_diff": max(grad_diffs),
        "passed": ok,
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = Config()
    cfg.apply_model_config(args.model)
    cfg.device = args.device
    cfg.dtype = "bf16"
    cfg.attention_backend = args.backend
    cfg.require_attention_backend = True

    from utils import get_device

    device = get_device(cfg)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend,
        "model": args.model,
        "head_dims": _ints(args.head_dims),
        "seq_lens": _ints(args.seq_lens),
        "tolerance": {"atol": ATOL, "rtol": RTOL, "compute_dtype": "bf16",
                      "reference": "fp32 SDPA, causal + key padding"},
        **_environment(device),
    }
    report_path = args.report or os.path.join(
        ROOT, "logs", f"verify_attention_{args.backend}.json",
    )

    def write(status_code):
        report["exit_code"] = status_code
        os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"report: {report_path}")
        return status_code

    try:
        backend = AttentionBackend(cfg)
        report["self_test"] = backend.self_test()
        report["backend_module"] = _backend_module(backend.backend_name)
    except AttentionBackendError as exc:
        reason = str(exc)
        if args.backend == "flash_attention_3":
            reason = f"FlashAttention-3 is unavailable on this machine: {reason}"
        print(reason, file=sys.stderr)
        report["passed"] = False
        report["unavailable_reason"] = reason
        report["cases"] = []
        return write(EXIT_UNAVAILABLE)

    cases = []
    for head_dim in report["head_dims"]:
        for seq_len in report["seq_lens"]:
            for padded in (True, False):
                case = run_case(backend, device, cfg.num_heads, head_dim, seq_len, padded)
                cases.append(case)
                print(
                    f"head_dim={head_dim:>3} seq_len={seq_len:>4} padded={padded!s:<5} "
                    f"out_diff={case['max_abs_out_diff']:.3e} "
                    f"grad_diff={case['max_abs_grad_diff']:.3e} "
                    f"{'PASS' if case['passed'] else 'FAIL'}"
                )
    report["cases"] = cases
    report["passed"] = all(c["passed"] for c in cases)
    print(f"backend={backend.backend_name} device={device} passed={report['passed']}")
    return write(EXIT_OK if report["passed"] else EXIT_PARITY)


if __name__ == "__main__":
    sys.exit(main())
