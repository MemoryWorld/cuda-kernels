"""
RMSNorm — fused Triton kernel vs PyTorch baseline.

PyTorch eager mode runs two HBM passes:
  Pass 1: read X  → compute mean(x²)  → write nothing
  Pass 2: read X  → normalize + scale  → write Y
  Plus separate kernel launches for each elementwise op.

This Triton kernel fuses everything into ONE pass:
  Read X once, compute RMS in SRAM, normalize + scale, write Y once.

Memory traffic reduction: ~2x (two reads → one read)
"""

import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import triton
import triton.language as tl

DEVICE = "cuda"
DTYPE  = torch.float16
EPS    = 1e-6

# Qwen2.5-7B hidden dims used in benchmarks
HIDDEN_DIMS = [896, 1536, 2048, 3072, 4096]
BATCH_SEQS  = [(1, 512), (1, 2048), (4, 512), (4, 2048)]


# ─────────────────────────────────────────────────────────────────────
# Triton kernel
# ─────────────────────────────────────────────────────────────────────

@triton.jit
def _rms_norm_kernel(
    X, W, Y,
    stride_row,          # stride between rows (= hidden_dim)
    N,                   # hidden dimension
    eps,
    BLOCK_N: tl.constexpr,
):
    """
    One program = one row (one token's hidden vector).
    Loads the entire row into SRAM, computes RMS, normalises, stores.
    """
    row  = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    # Load row into registers (fp32 for numerical stability)
    x = tl.load(X + row * stride_row + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)

    # RMS = sqrt( mean(x²) + eps )  — entirely in SRAM
    var = tl.sum(x * x, axis=0) / N
    rms = tl.rsqrt(var + eps)          # reciprocal sqrt

    # Normalise and scale
    y = x * rms * w

    tl.store(Y + row * stride_row + cols, y.to(tl.float16), mask=mask)


def rms_norm_triton(x: torch.Tensor, w: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """
    x : (rows, hidden_dim)  fp16
    w : (hidden_dim,)       fp16
    """
    assert x.is_contiguous() and x.dtype == torch.float16
    rows, N = x.shape
    y = torch.empty_like(x)

    # BLOCK_N must be a power of 2 and >= N
    BLOCK_N = triton.next_power_of_2(N)

    _rms_norm_kernel[(rows,)](
        x, w, y,
        x.stride(0),
        N, eps,
        BLOCK_N=BLOCK_N,
        num_warps=min(max(BLOCK_N // 256, 1), 16),
    )
    return y


# ─────────────────────────────────────────────────────────────────────
# PyTorch reference
# ─────────────────────────────────────────────────────────────────────

def rms_norm_torch(x: torch.Tensor, w: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    x_f32 = x.float()
    rms   = torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + eps)
    return (x_f32 * rms * w.float()).to(x.dtype)


# ─────────────────────────────────────────────────────────────────────
# Correctness
# ─────────────────────────────────────────────────────────────────────

def check_correctness():
    torch.manual_seed(0)
    x = torch.randn(512, 4096, device=DEVICE, dtype=DTYPE)
    w = torch.ones(4096,       device=DEVICE, dtype=DTYPE)
    ref  = rms_norm_torch(x, w)
    ours = rms_norm_triton(x, w)
    err  = (ref.float() - ours.float()).abs().max().item()
    ok   = err < 5e-3   # fp16 epsilon ~9.77e-4; allow 5x margin
    print(f"RMSNorm correctness:  max_err={err:.6f}  [{'PASS' if ok else 'FAIL'}]")
    return ok


# ─────────────────────────────────────────────────────────────────────
# Benchmark
# ─────────────────────────────────────────────────────────────────────

def bench(fn, *args, warmup=5, runs=50):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / runs * 1e6   # µs


def hbm_bytes_torch(rows, N):
    """2 reads of X + 1 read of W + 1 write of Y  (fp16 = 2 B)"""
    return (2 * rows * N + N + rows * N) * 2


def hbm_bytes_triton(rows, N):
    """1 read of X + 1 read of W + 1 write of Y"""
    return (rows * N + N + rows * N) * 2


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    if not check_correctness():
        return
    print()

    records = []

    # Sweep hidden dim at fixed batch×seq
    B, T = 4, 512
    rows  = B * T

    print(f"Sweep hidden_dim  (batch={B}, seq={T}  →  rows={rows})")
    print(f"{'hidden':>8}  {'torch (µs)':>12}  {'triton (µs)':>13}  {'speedup':>8}  "
          f"{'BW torch':>10}  {'BW triton':>11}")
    print("-" * 72)

    for N in HIDDEN_DIMS:
        x = torch.randn(rows, N, device=DEVICE, dtype=DTYPE)
        w = torch.ones(N,       device=DEVICE, dtype=DTYPE)

        t_torch  = bench(rms_norm_torch,  x, w)
        t_triton = bench(rms_norm_triton, x, w)
        speedup  = t_torch / t_triton

        bw_torch  = hbm_bytes_torch(rows, N)  / (t_torch  * 1e-6) / 1e9
        bw_triton = hbm_bytes_triton(rows, N) / (t_triton * 1e-6) / 1e9

        print(f"{N:>8}  {t_torch:>11.2f}µs  {t_triton:>12.2f}µs  "
              f"{speedup:>7.2f}x  {bw_torch:>9.1f}GB/s  {bw_triton:>10.1f}GB/s")

        records.append(dict(hidden=N, torch_us=t_torch, triton_us=t_triton,
                            speedup=speedup, bw_torch=bw_torch, bw_triton=bw_triton))

    with open("../results/rmsnorm.json", "w") as f:
        json.dump(records, f, indent=2)

    # Plot
    hiddens   = [r["hidden"]    for r in records]
    speedups  = [r["speedup"]   for r in records]
    bw_t      = [r["bw_torch"]  for r in records]
    bw_tr     = [r["bw_triton"] for r in records]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.bar([str(h) for h in hiddens], speedups, color="C2")
    ax1.axhline(1.0, color="gray", linestyle="--")
    ax1.set_xlabel("Hidden dimension")
    ax1.set_ylabel("Speedup vs PyTorch")
    ax1.set_title("RMSNorm — Triton speedup over PyTorch eager")
    ax1.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(speedups):
        ax1.text(i, v + 0.02, f"{v:.2f}x", ha="center", fontsize=9)

    ax2.plot([str(h) for h in hiddens], bw_t,  "o-", label="PyTorch eager",  color="C3")
    ax2.plot([str(h) for h in hiddens], bw_tr, "^-", label="Triton fused",   color="C2", linewidth=2)
    ax2.axhline(1792, color="gray", linestyle=":", label="RTX 5090 peak BW (1792 GB/s)")
    ax2.set_xlabel("Hidden dimension")
    ax2.set_ylabel("HBM bandwidth (GB/s)")
    ax2.set_title("Effective memory bandwidth utilization")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"Fused RMSNorm Kernel  |  RTX 5090  |  batch={B} seq={T}", fontsize=11)
    plt.tight_layout()
    plt.savefig("../results/rmsnorm.png", dpi=150)
    print("\nSaved: results/rmsnorm.json  results/rmsnorm.png")


if __name__ == "__main__":
    main()
