"""
fused_rmsnorm_linear.py — Fused RMSNorm + Linear Projection (Triton)

Standard transformer pattern, executed twice per layer:
  x_norm = RMSNorm(x)           # reads x, writes x_norm to HBM
  y      = x_norm @ W_linear.T  # reads x_norm + W_linear, writes y

Naive HBM traffic per (norm + linear) pair:
  reads:  x × 1 (RMSNorm) + x_norm × 1 (Linear) + W_norm × 1 + W_linear × 1
  writes: x_norm × 1 (RMSNorm) + y × 1 (Linear)
  total:  3·M·K + K + N·K + M·N  (in fp16 elements)

Fused HBM traffic:
  reads:  x × 2 (pass-1: compute rms; pass-2: normalize + GEMM) + W_norm × 1 + W_linear × 1
  writes: y × 1
  total:  2·M·K + K + N·K + M·N

Savings: M·K elements (eliminates the x_norm intermediate write + read).
For Qwen2.5-7B prefill at seq=512, batch=4  (M=2048, K=3584):
  per pair:    2048 × 3584 × 2 B ≈ 14.7 MB
  per layer:   4 pairs (pre-attn Q/K/V + pre-FFN gate/up) ≈ 58.7 MB
  32 layers:   ~1.88 GB of HBM traffic eliminated per forward pass

Kernel design:
  Grid:  (⌈M/BLOCK_M⌉, ⌈N/BLOCK_N⌉)
  Block: BLOCK_M × BLOCK_N output tile
  Pass 1 (inner loop over K): accumulate per-row sum-of-squares → inv_rms [BLOCK_M]
  Pass 2 (inner loop over K): normalize x tile (fp32), cast fp16, tl.dot with W tile
  x_norm never written to HBM — stays in registers across both passes.

Run: python fused_rmsnorm_linear.py   (from kernels/)
"""

import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

DEVICE = "cuda"
DTYPE  = torch.float16
EPS    = 1e-6


# ── Triton kernel ──────────────────────────────────────────────────────────────

@triton.jit
def _fused_rmsnorm_linear_kernel(
    X,           # [M, K]  fp16  input activations
    W_norm,      # [K]     fp16  RMSNorm scale
    W_linear,    # [N, K]  fp16  linear projection weight
    Y,           # [M, N]  fp16  output
    M, N, K,
    stride_xm, stride_xk,          # X strides
    stride_wn, stride_wk,          # W_linear strides  (row-major: stride_wn=K, stride_wk=1)
    stride_ym, stride_yn,          # Y strides
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    One program handles a [BLOCK_M, BLOCK_N] output tile.

    Pass 1 — compute per-row inv_rms:
      Loop over K in BLOCK_K tiles, load X, accumulate sum(x²) per row.
      X is read from HBM once here.

    Pass 2 — normalize + GEMM:
      Loop over K in BLOCK_K tiles:
        - Load X tile again (second HBM read of X)
        - Normalize in fp32: x_norm = x * inv_rms * w_norm
        - Cast x_norm to fp16
        - tl.dot(x_norm_tile [BLOCK_M, BLOCK_K],
                 tl.trans(W_tile) [BLOCK_K, BLOCK_N]) → accumulate into acc
      x_norm never written to HBM.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_offs = m_start + tl.arange(0, BLOCK_M)
    n_offs = n_start + tl.arange(0, BLOCK_N)
    m_mask = m_offs < M
    n_mask = n_offs < N

    # ── Pass 1: accumulate per-row sum-of-squares ─────────────────────────────
    sum_sq = tl.zeros([BLOCK_M], dtype=tl.float32)

    for k in tl.range(0, K, BLOCK_K):
        k_offs = k + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K
        X_tile = tl.load(
            X + m_offs[:, None] * stride_xm + k_offs[None, :] * stride_xk,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)                          # [BLOCK_M, BLOCK_K]
        sum_sq += tl.sum(X_tile * X_tile, axis=1) # [BLOCK_M]

    inv_rms = tl.rsqrt(sum_sq / K + eps)          # [BLOCK_M]

    # ── Pass 2: normalize + tiled GEMM ───────────────────────────────────────
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in tl.range(0, K, BLOCK_K):
        k_offs = k + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K

        # Load X and normalize (fp32 for precision)
        X_tile = tl.load(
            X + m_offs[:, None] * stride_xm + k_offs[None, :] * stride_xk,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)                                        # [BLOCK_M, BLOCK_K]

        W_norm_tile = tl.load(
            W_norm + k_offs, mask=k_mask, other=0.0,
        ).to(tl.float32)                                        # [BLOCK_K]

        X_norm_tile = (X_tile * inv_rms[:, None] * W_norm_tile[None, :]).to(tl.float16)
        # [BLOCK_M, BLOCK_K] fp16 — ready for tensor core

        # Load W_linear tile [BLOCK_N, BLOCK_K] (coalesced: K is fast dim)
        W_tile = tl.load(
            W_linear + n_offs[:, None] * stride_wn + k_offs[None, :] * stride_wk,
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float16)                                        # [BLOCK_N, BLOCK_K]

        # GEMM: X_norm [BLOCK_M, BLOCK_K] @ W^T [BLOCK_K, BLOCK_N] → [BLOCK_M, BLOCK_N]
        acc = tl.dot(X_norm_tile, tl.trans(W_tile), acc=acc, out_dtype=tl.float32)

    # Write output
    tl.store(
        Y + m_offs[:, None] * stride_ym + n_offs[None, :] * stride_yn,
        acc.to(tl.float16),
        mask=m_mask[:, None] & n_mask[None, :],
    )


def fused_rmsnorm_linear(
    x: torch.Tensor,        # [M, K]  fp16
    w_norm: torch.Tensor,   # [K]     fp16  RMSNorm scale
    w_linear: torch.Tensor, # [N, K]  fp16  linear weight
    eps: float = EPS,
    BLOCK_M: int = 16,
    BLOCK_N: int = 64,
    BLOCK_K: int = 64,
) -> torch.Tensor:
    assert x.is_contiguous() and x.dtype == torch.float16
    assert w_linear.is_contiguous() and w_linear.dtype == torch.float16
    M, K = x.shape
    N    = w_linear.shape[0]
    assert w_linear.shape[1] == K
    y = torch.empty((M, N), device=x.device, dtype=torch.float16)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _fused_rmsnorm_linear_kernel[grid](
        x, w_norm, w_linear, y,
        M, N, K,
        x.stride(0),        x.stride(1),
        w_linear.stride(0), w_linear.stride(1),
        y.stride(0),        y.stride(1),
        eps,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4,
    )
    return y


# ── PyTorch reference (naive, two kernels) ────────────────────────────────────

def rmsnorm_linear_torch(x, w_norm, w_linear, eps=EPS):
    x_f32   = x.float()
    inv_rms = torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + eps)
    x_norm  = (x_f32 * inv_rms * w_norm.float()).half()
    return F.linear(x_norm, w_linear)          # x_norm @ w_linear.T


# ── Correctness ───────────────────────────────────────────────────────────────

def check_correctness():
    torch.manual_seed(42)
    M, K, N = 64, 3584, 3584
    x        = torch.randn(M, K, device=DEVICE, dtype=DTYPE)
    w_norm   = torch.ones(K,     device=DEVICE, dtype=DTYPE)   # ones: cleaner reference
    w_linear = torch.randn(N, K, device=DEVICE, dtype=DTYPE)

    ref  = rmsnorm_linear_torch(x, w_norm, w_linear)
    ours = fused_rmsnorm_linear(x, w_norm, w_linear)

    abs_err = (ref.float() - ours.float()).abs().max().item()
    # fp16 GEMM: output scale ~ sqrt(K) * w_scale ~ 60; tolerance = 0.5% of output range
    tol = max(0.1, 0.005 * ref.float().abs().max().item())
    ok  = abs_err < tol
    print(f"Correctness  M={M} K={K} N={N}:  max_err={abs_err:.4f}  tol={tol:.4f}  "
          f"[{'PASS' if ok else 'FAIL'}]")
    return ok


# ── Benchmark helper ──────────────────────────────────────────────────────────

def bench(fn, *args, warmup=5, runs=50):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / runs * 1e6   # µs


def hbm_bytes_naive(M, K, N):
    """3·M·K + K + N·K + M·N  (fp16 = 2 B)"""
    return (3 * M * K + K + N * K + M * N) * 2


def hbm_bytes_fused(M, K, N):
    """2·M·K + K + N·K + M·N  (fp16 = 2 B)"""
    return (2 * M * K + K + N * K + M * N) * 2


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not check_correctness():
        return
    print()

    # Qwen2.5-7B relevant dimensions
    # K=3584 (hidden), N=3584 (Q-proj), N=18944 (gate/up-proj in FFN)
    K = 3584

    configs = [
        # (label,   N,     BLOCK_M, BLOCK_N, BLOCK_K, description)
        ("Q-proj",  3584,  16, 64, 64,  "attention Q projection"),
        ("FFN-gate",18944, 16, 64, 64,  "FFN gate/up projection"),
    ]

    # Token counts: decode (small M) vs prefill (large M)
    M_VALUES = [1, 4, 16, 64, 256, 512, 1024, 2048]

    all_records = {}

    for label, N, BM, BN, BK, desc in configs:
        print(f"── {label} ({desc})  K={K} N={N} ─────────────────────────────")
        print(f"{'M (tokens)':>12}  {'naive (µs)':>11}  {'fused (µs)':>11}"
              f"  {'speedup':>8}  {'HBM saved (MB)':>15}")
        print("-" * 72)

        records = []
        torch.manual_seed(0)
        w_norm   = torch.randn(K,    device=DEVICE, dtype=DTYPE)
        w_linear = torch.randn(N, K, device=DEVICE, dtype=DTYPE).contiguous()

        for M in M_VALUES:
            x = torch.randn(M, K, device=DEVICE, dtype=DTYPE)

            t_naive = bench(rmsnorm_linear_torch, x, w_norm, w_linear)
            t_fused = bench(fused_rmsnorm_linear, x, w_norm, w_linear,
                            EPS, BM, BN, BK)
            speedup = t_naive / t_fused

            saved_mb = (hbm_bytes_naive(M, K, N) - hbm_bytes_fused(M, K, N)) / 1e6

            print(f"{M:>12}  {t_naive:>10.2f}µs  {t_fused:>10.2f}µs"
                  f"  {speedup:>7.2f}x  {saved_mb:>14.1f} MB")

            records.append(dict(M=M, naive_us=t_naive, fused_us=t_fused,
                                speedup=speedup, saved_mb=saved_mb, N=N, K=K))

        all_records[label] = records
        print()

    # Save JSON
    with open("../results/fused_rmsnorm_linear.json", "w") as f:
        json.dump(all_records, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (label, records) in zip(axes, all_records.items()):
        Ms       = [r["M"]       for r in records]
        speedups = [r["speedup"] for r in records]
        ax.plot(Ms, speedups, "o-", color="C2", linewidth=2, markersize=6)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Number of tokens (M)")
        ax.set_ylabel("Speedup vs naive (RMSNorm + F.linear)")
        ax.set_title(f"{label}  K={records[0]['K']} N={records[0]['N']}")
        ax.grid(True, alpha=0.3)
        for x_val, y_val in zip(Ms, speedups):
            ax.annotate(f"{y_val:.2f}x", (x_val, y_val),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8)

    plt.suptitle(
        "Fused RMSNorm + Linear  |  RTX 5090  |  Speedup over PyTorch eager",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig("../results/fused_rmsnorm_linear.png", dpi=150)
    print("Saved: results/fused_rmsnorm_linear.json  results/fused_rmsnorm_linear.png")


if __name__ == "__main__":
    main()
