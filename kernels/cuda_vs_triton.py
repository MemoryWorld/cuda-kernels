"""
cuda_vs_triton.py — PyTorch eager vs Triton vs CUDA C++

Benchmarks all three implementations for RMSNorm, RoPE, and SwiGLU.

Triton (Driver API) and PyTorch CUDA C++ extensions (Runtime API) cannot
share the same CUDA context on Blackwell (sm_120) without deadlocking.
Fix: run each backend in an isolated subprocess; main process merges results.

Usage:
  python cuda_vs_triton.py            # full benchmark + plot
  python cuda_vs_triton.py --worker triton
  python cuda_vs_triton.py --worker cuda
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Shared config ──────────────────────────────────────────────────────────────

DEVICE = "cuda"
DTYPE_STR = "float16"
EPS = 1e-6

RMSNORM_HIDDEN = [896, 1536, 2048, 3072, 3584, 4096]
ROPE_SEQS      = [512, 1024, 2048, 4096, 8192]
SWIGLU_FFNS    = [4096, 8192, 11008, 14336, 18944]

WARMUP = 10
RUNS   = 100


# ── Worker: Triton-only ────────────────────────────────────────────────────────

def worker_triton():
    import torch
    import torch.nn.functional as F

    sys.path.insert(0, str(Path(__file__).parent))
    from rmsnorm import rms_norm_triton
    from rope    import rope_triton, build_cos_sin
    from swiglu  import swiglu_triton

    dtype = torch.float16

    def bench(fn, *args):
        for _ in range(WARMUP):
            fn(*args)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(RUNS):
            fn(*args)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / RUNS * 1e6

    # RMSNorm
    rmsnorm = []
    for N in RMSNORM_HIDDEN:
        x = torch.randn(4 * 512, N, device=DEVICE, dtype=dtype)
        w = torch.ones(N, device=DEVICE, dtype=dtype)
        us = bench(rms_norm_triton, x, w, EPS)
        rmsnorm.append({"hidden": N, "us": us})

    # RoPE
    rope = []
    cos_max, sin_max = build_cos_sin(max(ROPE_SEQS), 128)
    for T in ROPE_SEQS:
        x   = torch.randn(1, 32, T, 128, device=DEVICE, dtype=dtype)
        cos = cos_max[:T]
        sin = sin_max[:T]
        us = bench(rope_triton, x, cos, sin)
        rope.append({"seq": T, "us": us})

    # SwiGLU
    swiglu = []
    for F_DIM in SWIGLU_FFNS:
        gate = torch.randn(4 * 512, F_DIM, device=DEVICE, dtype=dtype).contiguous()
        up   = torch.randn(4 * 512, F_DIM, device=DEVICE, dtype=dtype).contiguous()
        us = bench(swiglu_triton, gate, up)
        swiglu.append({"ffn_dim": F_DIM, "us": us})

    # Correctness spot-check
    torch.manual_seed(42)
    x_rms = torch.randn(8, 3584, device=DEVICE, dtype=dtype)
    w_rms = torch.ones(3584, device=DEVICE, dtype=dtype)
    x_f32 = x_rms.float()
    ref_rms = (x_f32 * torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + EPS) * w_rms.float()).half()
    err_rms = (ref_rms.float() - rms_norm_triton(x_rms, w_rms, EPS).float()).abs().max().item()

    print(json.dumps({
        "rmsnorm": rmsnorm,
        "rope": rope,
        "swiglu": swiglu,
        "correctness": {"rmsnorm_err": err_rms},
    }))


# ── Worker: CUDA C++-only ──────────────────────────────────────────────────────

def worker_cuda():
    import torch
    import torch.nn.functional as F

    cuda_dir = str(Path(__file__).parent / "cuda")
    sys.path.insert(0, cuda_dir)
    import rmsnorm_cuda_ext
    import rope_cuda_ext
    import swiglu_cuda_ext

    # Inline build_cos_sin to avoid importing rope.py which pulls in Triton
    def build_cos_sin(seq_len, head_dim, base=10000.0):
        half = head_dim // 2
        inv_theta = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
        pos = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(pos, inv_theta)
        return torch.cos(freqs).to("cuda"), torch.sin(freqs).to("cuda")

    dtype = torch.float16

    def bench(fn, *args):
        for _ in range(WARMUP):
            fn(*args)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(RUNS):
            fn(*args)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / RUNS * 1e6

    # RMSNorm
    rmsnorm = []
    for N in RMSNORM_HIDDEN:
        x = torch.randn(4 * 512, N, device=DEVICE, dtype=dtype)
        w = torch.ones(N, device=DEVICE, dtype=dtype)
        us = bench(rmsnorm_cuda_ext.rms_norm_cuda, x, w, EPS)
        rmsnorm.append({"hidden": N, "us": us})

    # RoPE
    rope = []
    cos_max, sin_max = build_cos_sin(max(ROPE_SEQS), 128)
    for T in ROPE_SEQS:
        x   = torch.randn(1, 32, T, 128, device=DEVICE, dtype=dtype)
        cos = cos_max[:T]
        sin = sin_max[:T]
        us = bench(rope_cuda_ext.rope_cuda, x, cos, sin)
        rope.append({"seq": T, "us": us})

    # SwiGLU
    swiglu = []
    for F_DIM in SWIGLU_FFNS:
        gate = torch.randn(4 * 512, F_DIM, device=DEVICE, dtype=dtype).contiguous()
        up   = torch.randn(4 * 512, F_DIM, device=DEVICE, dtype=dtype).contiguous()
        us = bench(swiglu_cuda_ext.swiglu_cuda, gate, up)
        swiglu.append({"ffn_dim": F_DIM, "us": us})

    # Correctness spot-check
    torch.manual_seed(42)
    x_rms = torch.randn(8, 3584, device=DEVICE, dtype=dtype)
    w_rms = torch.ones(3584, device=DEVICE, dtype=dtype)
    x_f32 = x_rms.float()
    ref_rms = (x_f32 * torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + EPS) * w_rms.float()).half()
    err_rms = (ref_rms.float() - rmsnorm_cuda_ext.rms_norm_cuda(x_rms, w_rms, EPS).float()).abs().max().item()

    print(json.dumps({
        "rmsnorm": rmsnorm,
        "rope": rope,
        "swiglu": swiglu,
        "correctness": {"rmsnorm_err": err_rms},
    }))


# ── Worker: PyTorch eager-only ─────────────────────────────────────────────────

def worker_torch():
    import torch
    import torch.nn.functional as F

    # Inline build_cos_sin to avoid importing rope.py which pulls in Triton
    def build_cos_sin(seq_len, head_dim, base=10000.0):
        half = head_dim // 2
        inv_theta = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
        pos = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(pos, inv_theta)
        return torch.cos(freqs).to("cuda"), torch.sin(freqs).to("cuda")

    dtype = torch.float16

    def rms_norm_torch(x, w):
        x2d = x.reshape(-1, x.shape[-1]).float()
        return (x2d * torch.rsqrt(x2d.pow(2).mean(-1, keepdim=True) + EPS) * w.float()).half().reshape(x.shape)

    def rope_torch(x, cos, sin):
        B, H, T, D = x.shape
        D_half = D // 2
        x1 = x[..., :D_half].float()
        x2 = x[..., D_half:].float()
        c = cos[None, None, :, :]
        s = sin[None, None, :, :]
        return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).half()

    def swiglu_torch(gate, up):
        return (F.silu(gate.float()) * up.float()).half()

    def bench(fn, *args):
        for _ in range(WARMUP):
            fn(*args)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(RUNS):
            fn(*args)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / RUNS * 1e6

    rmsnorm = []
    for N in RMSNORM_HIDDEN:
        x = torch.randn(4 * 512, N, device=DEVICE, dtype=dtype)
        w = torch.ones(N, device=DEVICE, dtype=dtype)
        us = bench(rms_norm_torch, x, w)
        rmsnorm.append({"hidden": N, "us": us})

    rope = []
    cos_max, sin_max = build_cos_sin(max(ROPE_SEQS), 128)
    for T in ROPE_SEQS:
        x   = torch.randn(1, 32, T, 128, device=DEVICE, dtype=dtype)
        cos = cos_max[:T]
        sin = sin_max[:T]
        us = bench(rope_torch, x, cos, sin)
        rope.append({"seq": T, "us": us})

    swiglu = []
    for F_DIM in SWIGLU_FFNS:
        gate = torch.randn(4 * 512, F_DIM, device=DEVICE, dtype=dtype).contiguous()
        up   = torch.randn(4 * 512, F_DIM, device=DEVICE, dtype=dtype).contiguous()
        us = bench(swiglu_torch, gate, up)
        swiglu.append({"ffn_dim": F_DIM, "us": us})

    print(json.dumps({"rmsnorm": rmsnorm, "rope": rope, "swiglu": swiglu}))


# ── Main: spawn subprocesses, merge, plot ──────────────────────────────────────

def run_worker(mode):
    """Spawn a subprocess running this script in --worker <mode> and return parsed JSON."""
    env_prefix = ""
    if mode == "cuda":
        env_prefix = (
            "LD_LIBRARY_PATH="
            "/home/torch/miniconda3/lib/python3.13/site-packages/torch/lib"
            ":$LD_LIBRARY_PATH "
        )
    cmd = f"{env_prefix}python -u {__file__} --worker {mode}"
    print(f"  spawning worker: {mode} ...", flush=True)
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True,
        cwd=str(Path(__file__).parent),
    )
    if result.returncode != 0:
        print(f"  [ERROR] {mode} worker failed:\n{result.stderr}", flush=True)
        sys.exit(1)
    # Last line of stdout is the JSON
    lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
    return json.loads(lines[-1])


def merge_and_print(torch_data, triton_data, cuda_data):
    """Print aligned table and return merged records."""
    rmsnorm_recs, rope_recs, swiglu_recs = [], [], []

    print(f"\n── RMSNorm  (batch=4, seq=512) ──")
    print(f"{'hidden':>8}  {'torch(µs)':>10}  {'triton(µs)':>11}  {'cuda(µs)':>9}"
          f"  {'triton/torch':>13}  {'cuda/torch':>11}")
    print("─" * 76)
    for t, tr, c in zip(torch_data["rmsnorm"], triton_data["rmsnorm"], cuda_data["rmsnorm"]):
        sp_tr = t["us"] / tr["us"]
        sp_cu = t["us"] / c["us"]
        print(f"{t['hidden']:8d}  {t['us']:>9.2f}µs  {tr['us']:>10.2f}µs  {c['us']:>8.2f}µs"
              f"  {sp_tr:>12.2f}x  {sp_cu:>10.2f}x")
        rmsnorm_recs.append(dict(hidden=t["hidden"], torch_us=t["us"], triton_us=tr["us"],
                                 cuda_us=c["us"], sp_triton=sp_tr, sp_cuda=sp_cu))

    print(f"\n── RoPE  (batch=1, heads=32, head_dim=128) ──")
    print(f"{'seq':>6}  {'torch(µs)':>10}  {'triton(µs)':>11}  {'cuda(µs)':>9}"
          f"  {'triton/torch':>13}  {'cuda/torch':>11}")
    print("─" * 76)
    for t, tr, c in zip(torch_data["rope"], triton_data["rope"], cuda_data["rope"]):
        sp_tr = t["us"] / tr["us"]
        sp_cu = t["us"] / c["us"]
        print(f"{t['seq']:6d}  {t['us']:>9.2f}µs  {tr['us']:>10.2f}µs  {c['us']:>8.2f}µs"
              f"  {sp_tr:>12.2f}x  {sp_cu:>10.2f}x")
        rope_recs.append(dict(seq=t["seq"], torch_us=t["us"], triton_us=tr["us"],
                              cuda_us=c["us"], sp_triton=sp_tr, sp_cuda=sp_cu))

    print(f"\n── SwiGLU  (batch=4, seq=512) ──")
    print(f"{'ffn_dim':>8}  {'torch(µs)':>10}  {'triton(µs)':>11}  {'cuda(µs)':>9}"
          f"  {'triton/torch':>13}  {'cuda/torch':>11}")
    print("─" * 76)
    for t, tr, c in zip(torch_data["swiglu"], triton_data["swiglu"], cuda_data["swiglu"]):
        sp_tr = t["us"] / tr["us"]
        sp_cu = t["us"] / c["us"]
        print(f"{t['ffn_dim']:8d}  {t['us']:>9.2f}µs  {tr['us']:>10.2f}µs  {c['us']:>8.2f}µs"
              f"  {sp_tr:>12.2f}x  {sp_cu:>10.2f}x")
        swiglu_recs.append(dict(ffn_dim=t["ffn_dim"], torch_us=t["us"], triton_us=tr["us"],
                                cuda_us=c["us"], sp_triton=sp_tr, sp_cuda=sp_cu))

    return rmsnorm_recs, rope_recs, swiglu_recs


def plot_all(rmsnorm_rec, rope_rec, swiglu_rec):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    def speedup_bars(ax, records, x_key, title, xlabel):
        xs        = [r[x_key]       for r in records]
        sp_triton = [r["sp_triton"] for r in records]
        sp_cuda   = [r["sp_cuda"]   for r in records]
        idx = range(len(xs))
        w = 0.35
        ax.bar([i - w/2 for i in idx], sp_triton, w, label="Triton", color="C2")
        ax.bar([i + w/2 for i in idx], sp_cuda,   w, label="CUDA C++", color="C1")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        ax.set_xticks(list(idx))
        ax.set_xticklabels([str(x) for x in xs], fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Speedup vs PyTorch eager")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    speedup_bars(axes[0], rmsnorm_rec, "hidden",  "RMSNorm speedup",  "hidden_dim")
    speedup_bars(axes[1], rope_rec,    "seq",     "RoPE speedup",     "seq_len")
    speedup_bars(axes[2], swiglu_rec,  "ffn_dim", "SwiGLU speedup",   "ffn_dim")

    plt.suptitle(
        "Triton vs CUDA C++  |  RTX 5090  |  Speedup over PyTorch eager",
        fontsize=12,
    )
    plt.tight_layout()
    out = Path(__file__).parent.parent / "results" / "cuda_vs_triton.png"
    plt.savefig(out, dpi=150)
    print(f"\nSaved: {out}")


def main():
    print("Running workers in isolated subprocesses (avoids CUDA context conflict)...")
    print()

    print("[1/3] PyTorch eager baseline")
    torch_data = run_worker("torch")

    print("[2/3] Triton kernels")
    triton_data = run_worker("triton")

    print("[3/3] CUDA C++ kernels")
    cuda_data = run_worker("cuda")

    # Correctness report
    print()
    print("=" * 50)
    print("Correctness (vs PyTorch eager, max abs err)")
    print("=" * 50)
    tc = triton_data.get("correctness", {})
    cc = cuda_data.get("correctness", {})
    print(f"  RMSNorm  Triton={tc.get('rmsnorm_err', '?'):.2e}  CUDA={cc.get('rmsnorm_err', '?'):.2e}")

    print()
    print("=" * 50)
    print("Benchmarks (µs per call, 100 runs)")
    print("=" * 50)
    rmsnorm_rec, rope_rec, swiglu_rec = merge_and_print(torch_data, triton_data, cuda_data)

    # Save JSON
    out_json = Path(__file__).parent.parent / "results" / "cuda_vs_triton.json"
    with open(out_json, "w") as f:
        json.dump({"rmsnorm": rmsnorm_rec, "rope": rope_rec, "swiglu": swiglu_rec}, f, indent=2)
    print(f"\nSaved: {out_json}")

    plot_all(rmsnorm_rec, rope_rec, swiglu_rec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=["triton", "cuda", "torch"], default=None)
    args = parser.parse_args()

    if args.worker == "triton":
        worker_triton()
    elif args.worker == "cuda":
        worker_cuda()
    elif args.worker == "torch":
        worker_torch()
    else:
        main()
