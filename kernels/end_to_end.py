"""
end_to_end.py — Qwen2.5-7B prefill latency: PyTorch eager vs Triton fused kernels.

Patches applied to the live model:
  • RMSNorm  — 65 instances (input_layernorm + post_attention_layernorm per layer × 32
                             + final model.norm)
  • SwiGLU   — 32 instances (one per MLP layer)

RoPE is excluded: it runs inside the attention forward and interleaves with
Q/K projection + GQA reshaping; patching it cleanly requires deeper surgery
with negligible additional gain given RMSNorm + SwiGLU already cover all
elementwise/reduction ops.

Benchmark: prefill (single forward pass, no generation loop).
  • No KV cache, no sampling — pure encoder-style forward pass.
  • Measures wall-time ms per forward pass at various sequence lengths.

Run from cuda-kernels/kernels/:
    python end_to_end.py
"""

import json
import sys
import time
import types
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

# ── Import our Triton kernels from the same directory ─────────────────
sys.path.insert(0, str(Path(__file__).parent))
from rmsnorm import rms_norm_triton   # (rows, N) fp16 → (rows, N) fp16
from swiglu import swiglu_triton      # (gate, up) → fused silu(gate)*up fp16

MODEL_ID  = "Qwen/Qwen2.5-7B-Instruct"
DEVICE    = "cuda"
SEQ_LENS  = [128, 256, 512, 1024, 2048]
WARMUP    = 3
RUNS      = 10


# ─────────────────────────────────────────────────────────────────────
# Triton wrapper modules
# ─────────────────────────────────────────────────────────────────────

class TritonRMSNorm(nn.Module):
    """Drop-in replacement for Qwen2RMSNorm using our fused Triton kernel."""

    def __init__(self, orig: nn.Module):
        super().__init__()
        self.weight           = orig.weight
        self.variance_epsilon = orig.variance_epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        orig_dtype = x.dtype
        # Kernel requires (rows, hidden) contiguous fp16
        x2d = x.reshape(-1, orig_shape[-1])
        if not x2d.is_contiguous():
            x2d = x2d.contiguous()
        if x2d.dtype != torch.float16:
            x2d = x2d.half()
        w = self.weight if self.weight.dtype == torch.float16 else self.weight.half()
        out = rms_norm_triton(x2d, w, self.variance_epsilon)
        return out.reshape(orig_shape).to(orig_dtype)


def _patched_mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Fused SwiGLU: replaces act_fn(gate_proj(x)) * up_proj(x) with one Triton pass."""
    gate = self.gate_proj(x)
    up   = self.up_proj(x)
    orig_dtype = gate.dtype
    g = gate.contiguous() if gate.is_contiguous() else gate.contiguous()
    u = up.contiguous()   if up.is_contiguous()   else up.contiguous()
    if g.dtype != torch.float16:
        g = g.half()
        u = u.half()
    hidden = swiglu_triton(g, u)
    if hidden.dtype != orig_dtype:
        hidden = hidden.to(orig_dtype)
    return self.down_proj(hidden)


# ─────────────────────────────────────────────────────────────────────
# Patching helpers
# ─────────────────────────────────────────────────────────────────────

def patch_model(model) -> tuple[int, int]:
    """
    Replace all RMSNorm and MLP SwiGLU ops in-place.
    Returns (n_rmsnorm_patched, n_swiglu_patched).
    """
    n_rms = 0
    for layer in model.model.layers:
        layer.input_layernorm         = TritonRMSNorm(layer.input_layernorm)
        layer.post_attention_layernorm = TritonRMSNorm(layer.post_attention_layernorm)
        n_rms += 2
        layer.mlp.forward = types.MethodType(_patched_mlp_forward, layer.mlp)
    model.model.norm = TritonRMSNorm(model.model.norm)
    n_rms += 1
    n_swiglu = len(model.model.layers)
    return n_rms, n_swiglu


# ─────────────────────────────────────────────────────────────────────
# Benchmark
# ─────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def bench(model, input_ids: torch.Tensor, warmup: int = WARMUP, runs: int = RUNS) -> float:
    """Returns mean forward-pass latency in ms."""
    for _ in range(warmup):
        model(input_ids)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        model(input_ids)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / runs * 1e3


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading {MODEL_ID} …")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map="cuda"
    ).eval()

    n_layers = model.config.num_hidden_layers
    h_dim    = model.config.hidden_size
    ffn_dim  = model.config.intermediate_size
    print(f"  layers={n_layers}  hidden={h_dim}  ffn_dim={ffn_dim}\n")

    # Pre-generate all inputs so shape differences don't affect timing
    inputs = {
        seq: torch.randint(100, 50_000, (1, seq), device=DEVICE)
        for seq in SEQ_LENS
    }

    # ── Phase 1: baseline (PyTorch eager) ────────────────────────────
    print("── Baseline (PyTorch eager) ──────────────────────────────")
    baselines: dict[int, float] = {}
    for seq in SEQ_LENS:
        ms = bench(model, inputs[seq])
        baselines[seq] = ms
        print(f"  seq={seq:5d}: {ms:8.2f} ms")

    # ── Patch ────────────────────────────────────────────────────────
    print("\nPatching model with Triton kernels …")
    n_rms, n_swiglu = patch_model(model)
    print(f"  Patched {n_rms} RMSNorm instances + {n_swiglu} SwiGLU instances")

    # Warm up Triton JIT compilation across all shapes
    print("Warming up Triton JIT (compiles on first call) …")
    for seq in SEQ_LENS:
        for _ in range(2):
            model(inputs[seq])
    torch.cuda.synchronize()
    print()

    # ── Phase 2: patched ──────────────────────────────────────────────
    print("── Patched (Triton fused) ────────────────────────────────")
    patched: dict[int, float] = {}
    for seq in SEQ_LENS:
        ms = bench(model, inputs[seq])
        patched[seq] = ms
        print(f"  seq={seq:5d}: {ms:8.2f} ms")

    # ── Summary table ─────────────────────────────────────────────────
    print(f"\n{'seq':>6}  {'baseline (ms)':>14}  {'patched (ms)':>13}  {'speedup':>8}")
    print("─" * 50)
    records = []
    for seq in SEQ_LENS:
        speedup = baselines[seq] / patched[seq]
        print(f"{seq:6d}  {baselines[seq]:14.2f}  {patched[seq]:13.2f}  {speedup:7.3f}x")
        records.append(dict(seq=seq, baseline_ms=baselines[seq],
                            patched_ms=patched[seq], speedup=speedup))

    # ── Save JSON ─────────────────────────────────────────────────────
    out_json = Path("../results/end_to_end.json")
    out_json.parent.mkdir(exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({
            "model": MODEL_ID,
            "n_layers": n_layers,
            "hidden_dim": h_dim,
            "ffn_dim": ffn_dim,
            "patched_ops": {"rmsnorm": n_rms, "swiglu": n_swiglu},
            "results": records,
        }, f, indent=2)

    # ── Plot ──────────────────────────────────────────────────────────
    seqs     = [r["seq"]         for r in records]
    base_ms  = [r["baseline_ms"] for r in records]
    patch_ms = [r["patched_ms"]  for r in records]
    speedups = [r["speedup"]     for r in records]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(seqs, base_ms,  "o-", color="C0", linewidth=2, label="PyTorch eager")
    ax1.plot(seqs, patch_ms, "s-", color="C2", linewidth=2, label="Triton fused (RMSNorm+SwiGLU)")
    ax1.set_xlabel("Sequence length (prefill tokens)")
    ax1.set_ylabel("Forward-pass latency (ms)")
    ax1.set_title("Qwen2.5-7B prefill latency")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="Baseline (1×)")
    ax2.plot(seqs, speedups, "o-", color="C2", linewidth=2, label="Triton speedup")
    ax2.set_xlabel("Sequence length")
    ax2.set_ylabel("Speedup vs PyTorch eager")
    ax2.set_title("Prefill speedup from kernel fusion")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(
        f"End-to-End Triton Fusion  |  Qwen2.5-7B  |  RTX 5090\n"
        f"Patched: {n_rms}× RMSNorm + {n_swiglu}× SwiGLU  (RoPE excluded)",
        fontsize=10,
    )
    plt.tight_layout()
    out_png = Path("../results/end_to_end.png")
    plt.savefig(out_png, dpi=150)
    print(f"\nSaved: {out_json}  {out_png}")


if __name__ == "__main__":
    main()
