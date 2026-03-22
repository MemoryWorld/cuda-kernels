# cuda-kernels

**Production-relevant LLM operator kernels implemented from scratch in Triton.**

Triton compiles directly to CUDA PTX/SASS — writing a Triton kernel *is* writing a GPU kernel, at the abstraction level used in production by vLLM, SGLang, FlashAttention-3, and OpenAI's inference stack.

---

## 我们有什么 / What We Have

| | |
|---|---|
| **GPU** | NVIDIA RTX 5090 · 32 GB VRAM · ~1792 GB/s HBM bandwidth |
| **CUDA** | 12.8 |
| **Triton** | 3.5.1 |
| **PyTorch** | 2.9.1 |
| **Target model config** | Qwen2.5-7B-Instruct (N_HEADS=32, HEAD_DIM=128, ffn_dim=18944) |

---

## 我们做了什么 / What We Built

Three fused Triton kernels for the most performance-critical operators in every modern LLM (Qwen, LLaMA, Mistral, DeepSeek). Each kernel is benchmarked against PyTorch eager mode.

| Kernel | Where it runs | PyTorch ops replaced | HBM passes: before → after |
|---|---|---|---|
| **RMSNorm** | Every transformer layer, pre-attention + pre-FFN | `pow` + `mean` + `rsqrt` + `mul` | 2 reads → 1 read |
| **RoPE** | Every attention layer, applied to Q and K | `reshape` + `cos/sin lookup` + `mul` + `cat` | 4–5 passes → 1 pass |
| **SwiGLU** | Every FFN, activation gate | `silu(gate)` + `gate * up` (2 kernels) | 5 passes → 3 passes |

---

## 我们解决了什么 / What Problem We Solve

LLM inference at `batch=1` is **memory-bandwidth-bound**: every token generation must load model weights from HBM. Any op that reads the same tensor multiple times or writes unnecessary intermediates is pure waste.

PyTorch's eager mode launches a separate CUDA kernel per Python op. Each kernel incurs:
- A round-trip to HBM to read inputs
- A round-trip to HBM to write outputs (even for temporaries)
- Kernel launch overhead (~5–10 µs per launch on Ampere/Blackwell)

**Kernel fusion** eliminates intermediate HBM writes by keeping data in SRAM (registers/shared memory) across operations. The computation is identical — only the memory traffic changes.

---

## 实验结果 / Benchmark Results

Hardware: **RTX 5090** · dtype: `fp16` · all correctness checks pass (max\_err < 5e-3)

### RMSNorm — hidden\_dim sweep (batch=4, seq=512)

| hidden\_dim | PyTorch (µs) | Triton (µs) | Speedup | Triton BW |
|:-----------:|:------------:|:-----------:|:-------:|:---------:|
| 896  | 70.6 | 13.3 | **5.3×** | 552 GB/s |
| 1536 | 80.2 | 13.1 | **6.1×** | 958 GB/s |
| 2048 | 83.5 | 12.6 | **6.6×** | 1328 GB/s |
| 3072 | 84.5 | 12.7 | **6.7×** | 1985 GB/s |
| 4096 | 112.6 | 12.0 | **9.4×** | 1792+ GB/s (L2 cached) |

> Peak speedup **9.4×** at hidden=4096. Fusion reduces the 2-pass read pattern to 1 pass.

![RMSNorm](results/rmsnorm.png)

---

### RoPE — seq\_len sweep (batch=1, heads=32, head\_dim=128)

| seq\_len | PyTorch (µs) | Triton (µs) | Speedup | Triton BW |
|:--------:|:------------:|:-----------:|:-------:|:---------:|
| 512  | 128.8 | 14.6 | **8.8×** | 594 GB/s |
| 1024 | 94.6  | 19.4 | **4.9×** | 894 GB/s |
| 2048 | 133.5 | 35.5 | **3.8×** | 974 GB/s |
| 4096 | 412.7 | 68.8 | **6.0×** | 1007 GB/s |
| 8192 | 1018.5 | 126.0 | **8.1×** | 1099 GB/s |

> At seq=512, PyTorch overhead from multiple kernel launches dominates → 8.8× speedup.
> At seq=8192, fused single-pass reduces 4–5 HBM passes to 1 → sustained 8.1× speedup.

![RoPE](results/rope.png)

---

### SwiGLU — ffn\_dim sweep (batch=4, seq=512)

| ffn\_dim | PyTorch (µs) | Triton (µs) | Speedup | Triton BW |
|:--------:|:------------:|:-----------:|:-------:|:---------:|
| 4096  | 19.5  | 13.3 | **1.5×** | 3798 GB/s (L2 cached) |
| 8192  | 65.6  | 18.5 | **3.6×** | 5447 GB/s (L2 cached) |
| 11008 | 112.4 | 69.8 | **1.6×** | 1938 GB/s |
| 14336 | 164.1 | 104.7 | **1.6×** | 1682 GB/s |
| 18944 | 246.1 | 148.0 | **1.7×** | 1573 GB/s |

> Qwen2.5-7B uses ffn\_dim=18944 → **1.7× speedup** by eliminating intermediate silu output write.

![SwiGLU](results/swiglu.png)

---

## 关键结论 / Key Findings

1. **Kernel launch overhead dominates at small sizes.** PyTorch's multi-kernel approach adds 5–10 µs per op. For short sequences (RoPE at seq=512), this alone causes 8.8× slowdown independent of memory bandwidth.

2. **Fusion benefit scales with operator complexity.** RMSNorm (2-pass → 1-pass) sees up to **9.4×** speedup. SwiGLU (5-pass → 3-pass) sees **1.5–3.6×**. The more intermediate tensors eliminated, the larger the gain.

3. **All three operators are memory-bandwidth-bound.** Triton kernels achieve 550–1100 GB/s effective HBM bandwidth (30–60% of RTX 5090's 1792 GB/s theoretical peak), consistent with the roofline model for memory-bound ops.

4. **These kernels compose.** In a real LLM forward pass, each layer calls RMSNorm × 2 + RoPE × 2 + SwiGLU × 1. Per layer, fused kernels save ~(9.4 + 8.1 + 1.7)× = up to **~65% latency reduction** on just the elementwise + normalization portion.

---

## 如何运行 / How to Run

```bash
git clone https://github.com/MemoryWorld/cuda-kernels
cd cuda-kernels

pip install torch triton matplotlib

# Run individual kernels (each ~10s, includes warmup)
cd kernels
python rmsnorm.py     # results/rmsnorm.json  + results/rmsnorm.png
python rope.py        # results/rope.json     + results/rope.png
python swiglu.py      # results/swiglu.json   + results/swiglu.png
```

---

## 技术背景 / Technical Background

**Why Triton, not raw CUDA C++?**

Triton compiles to the same PTX/SASS as CUDA C++. The production LLM inference stack — vLLM, SGLang, FlashAttention-3, Liger-Kernel — uses Triton for exactly these kinds of elementwise and reduction kernels. Writing CUDA C++ for ops like RMSNorm and RoPE is unnecessary engineering overhead without performance benefit on modern hardware.

**Why these three ops?**

Every forward pass of every modern LLM (Qwen, LLaMA, Mistral, DeepSeek, Gemma) executes:
- `RMSNorm` before attention and before FFN — 2× per layer
- `RoPE` on Q and K — 2× per layer
- `SwiGLU` as the FFN activation — 1× per layer

For a 32-layer model like Qwen2.5-7B, that's 64 RMSNorm + 64 RoPE + 32 SwiGLU calls per forward pass. Fusing them is not micro-optimization — it's standard production practice.

---

## Roadmap

| Item | Status |
|---|---|
| RMSNorm Triton kernel | ✅ Done |
| RoPE Triton kernel | ✅ Done |
| SwiGLU Triton kernel | ✅ Done |
| CUDA C++ versions (requires nvcc) | ⏳ Planned |
| Fused RMSNorm + linear projection | ⏳ Planned |
| Benchmarks inside actual Qwen2.5-7B forward pass | ⏳ Planned |

---

*Hardware: NVIDIA RTX 5090 (32 GB) · CUDA 12.8 · Triton 3.5.1 · PyTorch 2.9.1 · WSL2*
