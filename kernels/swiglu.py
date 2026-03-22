"""
SwiGLU activation — fused Triton kernel vs PyTorch baseline.

SwiGLU is the FFN activation in every modern LLM: LLaMA, Qwen, Mistral…
  FFN(x) = (silu(gate) ⊙ up) @ W_down
  silu(x) = x · σ(x)   where σ is sigmoid

PyTorch eager mode splits into separate ops:
  1. silu(gate)  — reads gate, writes temp
  2. temp * up   — reads temp + up, writes out
  Two kernel launches, two reads of the output of step 1.

Triton fused kernel:
  Reads gate and up ONCE, computes silu(gate)*up, writes output ONCE.
  Eliminates the intermediate tensor write/read entirely.
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

DEVICE  = "cuda"
DTYPE   = torch.float16

# Qwen2.5-7B FFN intermediate sizes  (ffn_dim = 4 * hidden, approx)
# Actual Qwen2.5 uses SwiGLU with ffn_hidden = 18944 for 7B
# We benchmark across representative sizes
FFN_DIMS = [4096, 8192, 11008, 14336, 18944]
BATCH_SEQS = [(1, 512), (4, 512), (1, 2048), (4, 2048)]


# ─────────────────────────────────────────────────────────────────────
# Triton kernel
# ─────────────────────────────────────────────────────────────────────

@triton.jit
def _swiglu_kernel(
    GATE, UP, OUT,
    N,
    BLOCK_N: tl.constexpr,
):
    """
    One program = one BLOCK_N chunk of the flattened (rows × ffn_dim) tensor.
    Fuses silu(gate) and elementwise multiply with up in one pass.
    """
    pid  = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N

    gate = tl.load(GATE + offs, mask=mask).to(tl.float32)
    up   = tl.load(UP   + offs, mask=mask).to(tl.float32)

    # silu(gate) = gate · sigmoid(gate)
    silu_gate = gate * tl.sigmoid(gate)

    # fused multiply
    out = silu_gate * up

    tl.store(OUT + offs, out.to(tl.float16), mask=mask)


def swiglu_triton(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """
    gate, up : (..., ffn_dim)  fp16
    returns  : (..., ffn_dim)  fp16
    """
    assert gate.dtype == torch.float16 and gate.is_contiguous()
    out     = torch.empty_like(gate)
    N       = gate.numel()
    BLOCK_N = min(triton.next_power_of_2(gate.shape[-1]), 4096)
    grid    = (triton.cdiv(N, BLOCK_N),)

    _swiglu_kernel[grid](gate, up, out, N, BLOCK_N=BLOCK_N, num_warps=8)
    return out


# ─────────────────────────────────────────────────────────────────────
# PyTorch reference
# ─────────────────────────────────────────────────────────────────────

def swiglu_torch(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(gate) * up


# ─────────────────────────────────────────────────────────────────────
# Correctness
# ─────────────────────────────────────────────────────────────────────

def check_correctness():
    torch.manual_seed(0)
    gate = torch.randn(512, 18944, device=DEVICE, dtype=DTYPE)
    up   = torch.randn(512, 18944, device=DEVICE, dtype=DTYPE)

    ref  = swiglu_torch(gate, up)
    ours = swiglu_triton(gate, up)
    err  = (ref.float() - ours.float()).abs().max().item()
    ok   = err < 1e-2
    print(f"SwiGLU correctness:  max_err={err:.6f}  [{'PASS' if ok else 'FAIL'}]")
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


def hbm_bytes_torch(rows, ffn):
    """read gate + write silu_out + read silu_out + read up + write out (fp16)"""
    return (rows * ffn * 5) * 2


def hbm_bytes_triton(rows, ffn):
    """read gate + read up + write out (fp16)"""
    return (rows * ffn * 3) * 2


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    if not check_correctness():
        return
    print()

    records = []
    B, T = 4, 512
    rows = B * T

    print(f"Sweep ffn_dim  (batch={B}, seq={T}  →  rows={rows})")
    print(f"{'ffn_dim':>8}  {'torch (µs)':>12}  {'triton (µs)':>13}  "
          f"{'speedup':>8}  {'BW torch':>10}  {'BW triton':>11}")
    print("-" * 76)

    for D in FFN_DIMS:
        gate = torch.randn(rows, D, device=DEVICE, dtype=DTYPE)
        up   = torch.randn(rows, D, device=DEVICE, dtype=DTYPE)

        t_torch  = bench(swiglu_torch,  gate, up)
        t_triton = bench(swiglu_triton, gate, up)
        speedup  = t_torch / t_triton

        bw_torch  = hbm_bytes_torch(rows, D)  / (t_torch  * 1e-6) / 1e9
        bw_triton = hbm_bytes_triton(rows, D) / (t_triton * 1e-6) / 1e9

        print(f"{D:>8}  {t_torch:>11.2f}µs  {t_triton:>12.2f}µs  "
              f"{speedup:>7.2f}x  {bw_torch:>9.1f}GB/s  {bw_triton:>10.1f}GB/s")

        records.append(dict(ffn_dim=D, torch_us=t_torch, triton_us=t_triton,
                            speedup=speedup, bw_torch=bw_torch, bw_triton=bw_triton))

    with open("../results/swiglu.json", "w") as f:
        json.dump(records, f, indent=2)

    # Plot
    dims     = [str(r["ffn_dim"])  for r in records]
    speedups = [r["speedup"]       for r in records]
    bw_t     = [r["bw_torch"]      for r in records]
    bw_tr    = [r["bw_triton"]     for r in records]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.bar(dims, speedups, color="C2")
    ax1.axhline(1.0, color="gray", linestyle="--")
    ax1.set_xlabel("FFN intermediate dimension")
    ax1.set_ylabel("Speedup vs PyTorch eager")
    ax1.set_title("SwiGLU — Triton speedup over PyTorch eager")
    ax1.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(speedups):
        ax1.text(i, v + 0.02, f"{v:.2f}x", ha="center", fontsize=9)

    ax2.plot(dims, bw_t,  "o-", label="PyTorch eager",  color="C3")
    ax2.plot(dims, bw_tr, "^-", label="Triton fused",   color="C2", linewidth=2)
    ax2.axhline(1792, color="gray", linestyle=":", label="RTX 5090 peak BW (1792 GB/s)")
    ax2.set_xlabel("FFN intermediate dimension")
    ax2.set_ylabel("HBM bandwidth (GB/s)")
    ax2.set_title("Effective memory bandwidth utilization")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"Fused SwiGLU Kernel  |  RTX 5090  |  batch={B} seq={T}", fontsize=11)
    plt.tight_layout()
    plt.savefig("../results/swiglu.png", dpi=150)
    print("\nSaved: results/swiglu.json  results/swiglu.png")


if __name__ == "__main__":
    main()
