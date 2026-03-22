"""
RoPE (Rotary Position Embedding) — fused Triton kernel vs PyTorch baseline.

RoPE is applied to Q and K at every attention layer of every modern LLM
(LLaMA, Qwen, Mistral, DeepSeek…). At inference time it runs thousands
of times per request.

PyTorch eager mode:
  cos/sin lookup → reshape → elementwise multiply → concat  (4–5 ops)
  Each op launches a separate kernel and reads/writes Q/K from HBM.

Triton fused kernel:
  ONE kernel reads Q (or K), loads precomputed cos/sin, rotates, writes output.
  HBM traffic: 1 read + 1 write instead of 4-5 passes.

Uses the "half-split" rotation convention (Qwen / GPT-NeoX style):
  x1 = x[..., :D/2], x2 = x[..., D/2:]
  out = cat(x1*cos - x2*sin,  x1*sin + x2*cos, dim=-1)
"""

import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import triton
import triton.language as tl

DEVICE   = "cuda"
DTYPE    = torch.float16
N_HEADS  = 32
HEAD_DIM = 128        # Qwen2.5-7B
MAX_SEQ  = 8192
SEQ_LENS = [512, 1024, 2048, 4096, 8192]
BATCH    = 1


# ─────────────────────────────────────────────────────────────────────
# Precompute cos/sin table  (shared across Q and K)
# ─────────────────────────────────────────────────────────────────────

def build_cos_sin(seq_len: int, head_dim: int, base: float = 10000.0):
    """Returns cos, sin tensors of shape (seq_len, head_dim // 2)  fp32."""
    half  = head_dim // 2
    inv_θ = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
    pos   = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(pos, inv_θ)            # (seq_len, half)
    return freqs.cos().to(DEVICE), freqs.sin().to(DEVICE)


# ─────────────────────────────────────────────────────────────────────
# Triton kernel
# ─────────────────────────────────────────────────────────────────────

@triton.jit
def _rope_fwd_kernel(
    X, COS, SIN, Y,
    B, H, T, D_half,
    stride_xb, stride_xh, stride_xt, stride_xd,
    BLOCK_D: tl.constexpr,
):
    """
    One program = one (batch, head, token) triple.
    Loads x1 and x2 halves, applies rotation, stores output.
    """
    pid     = tl.program_id(0)
    n_bh    = B * H
    pid_t   = pid  % T
    pid_bh  = pid // T
    b       = pid_bh // H
    h       = pid_bh  % H

    offs = tl.arange(0, BLOCK_D)
    mask = offs < D_half

    base_ptr = X + b * stride_xb + h * stride_xh + pid_t * stride_xt

    # Load first half x1 and second half x2
    x1 = tl.load(base_ptr + offs            * stride_xd, mask=mask).to(tl.float32)
    x2 = tl.load(base_ptr + (offs + D_half) * stride_xd, mask=mask).to(tl.float32)

    # Load precomputed cos/sin for this token position
    cos = tl.load(COS + pid_t * D_half + offs, mask=mask)
    sin = tl.load(SIN + pid_t * D_half + offs, mask=mask)

    # Rotate: (x1, x2) → (x1·cos − x2·sin,  x1·sin + x2·cos)
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos

    out_ptr = Y + b * stride_xb + h * stride_xh + pid_t * stride_xt
    tl.store(out_ptr + offs            * stride_xd, out1.to(tl.float16), mask=mask)
    tl.store(out_ptr + (offs + D_half) * stride_xd, out2.to(tl.float16), mask=mask)


def rope_triton(
    x: torch.Tensor,        # (B, H, T, D)  fp16
    cos: torch.Tensor,      # (T, D//2)     fp32
    sin: torch.Tensor,      # (T, D//2)     fp32
) -> torch.Tensor:
    B, H, T, D = x.shape
    D_half  = D // 2
    y       = torch.empty_like(x)
    BLOCK_D = triton.next_power_of_2(D_half)
    grid    = (B * H * T,)

    _rope_fwd_kernel[grid](
        x, cos, sin, y,
        B, H, T, D_half,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        BLOCK_D=BLOCK_D,
        num_warps=4,
    )
    return y


# ─────────────────────────────────────────────────────────────────────
# PyTorch reference  (half-split convention)
# ─────────────────────────────────────────────────────────────────────

def rope_torch(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    B, H, T, D = x.shape
    D_half = D // 2
    x1 = x[..., :D_half].float()
    x2 = x[..., D_half:].float()
    # cos/sin: (T, D_half) → broadcast over (B, H, T, D_half)
    c = cos[None, None, :, :]
    s = sin[None, None, :, :]
    out1 = x1 * c - x2 * s
    out2 = x1 * s + x2 * c
    return torch.cat([out1, out2], dim=-1).to(x.dtype)


# ─────────────────────────────────────────────────────────────────────
# Correctness
# ─────────────────────────────────────────────────────────────────────

def check_correctness():
    torch.manual_seed(0)
    T   = 512
    x   = torch.randn(BATCH, N_HEADS, T, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    cos, sin = build_cos_sin(T, HEAD_DIM)

    ref  = rope_torch(x, cos, sin)
    ours = rope_triton(x, cos, sin)
    err  = (ref.float() - ours.float()).abs().max().item()
    ok   = err < 5e-3   # fp16 epsilon ~9.77e-4; allow 5x margin
    print(f"RoPE correctness:  max_err={err:.6f}  [{'PASS' if ok else 'FAIL'}]")
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


def hbm_bytes(B, H, T, D):
    """Read X (fp16) + read cos/sin (fp32 halves) + write Y (fp16)."""
    qkv_bytes  = B * H * T * D * 2       # fp16 read
    cs_bytes   = T * (D // 2) * 4 * 2    # fp32 cos+sin read
    out_bytes  = B * H * T * D * 2       # fp16 write
    return qkv_bytes + cs_bytes + out_bytes


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    if not check_correctness():
        return
    print()

    records = []
    cos_max, sin_max = build_cos_sin(max(SEQ_LENS), HEAD_DIM)

    print(f"Sweep seq_len  (batch={BATCH}, heads={N_HEADS}, head_dim={HEAD_DIM})")
    print(f"{'seq':>6}  {'torch (µs)':>12}  {'triton (µs)':>13}  {'speedup':>8}  {'BW (GB/s)':>10}")
    print("-" * 60)

    for T in SEQ_LENS:
        x   = torch.randn(BATCH, N_HEADS, T, HEAD_DIM, device=DEVICE, dtype=DTYPE)
        cos = cos_max[:T]
        sin = sin_max[:T]

        t_torch  = bench(rope_torch,  x, cos, sin)
        t_triton = bench(rope_triton, x, cos, sin)
        speedup  = t_torch / t_triton
        bw       = hbm_bytes(BATCH, N_HEADS, T, HEAD_DIM) / (t_triton * 1e-6) / 1e9

        print(f"{T:>6}  {t_torch:>11.2f}µs  {t_triton:>12.2f}µs  "
              f"{speedup:>7.2f}x  {bw:>9.1f}GB/s")

        records.append(dict(seq=T, torch_us=t_torch, triton_us=t_triton,
                            speedup=speedup, bw_gb=bw))

    with open("../results/rope.json", "w") as f:
        json.dump(records, f, indent=2)

    # Plot
    seqs     = [r["seq"]       for r in records]
    speedups = [r["speedup"]   for r in records]
    bws      = [r["bw_gb"]     for r in records]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(seqs, speedups, "o-", color="C2", linewidth=2)
    ax1.axhline(1.0, color="gray", linestyle="--")
    ax1.set_xlabel("Sequence length")
    ax1.set_ylabel("Speedup vs PyTorch eager")
    ax1.set_title("RoPE — Triton speedup over PyTorch eager")
    ax1.grid(True, alpha=0.3)

    ax2.plot(seqs, bws, "^-", color="C2", linewidth=2, label="Triton fused")
    ax2.axhline(1792, color="gray", linestyle=":", label="RTX 5090 peak BW (1792 GB/s)")
    ax2.set_xlabel("Sequence length")
    ax2.set_ylabel("Effective HBM bandwidth (GB/s)")
    ax2.set_title("Bandwidth utilization (Triton kernel)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(
        f"Fused RoPE Kernel  |  RTX 5090  |  "
        f"batch={BATCH}  heads={N_HEADS}  head_dim={HEAD_DIM}",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig("../results/rope.png", dpi=150)
    print("\nSaved: results/rope.json  results/rope.png")


if __name__ == "__main__":
    main()
