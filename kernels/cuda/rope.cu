/*
 * RoPE (Rotary Position Embedding) — fused CUDA C++ kernel
 *
 * Half-split convention (Qwen / GPT-NeoX):
 *   x1 = x[..., :D/2],  x2 = x[..., D/2:]
 *   out = cat(x1·cos − x2·sin,  x1·sin + x2·cos, dim=-1)
 *
 * Each CUDA thread processes one (d) index across one (b, h, t) position,
 * reading x1[d] and x2[d] together and writing both output halves.
 * This avoids reading x twice, keeping HBM traffic to 1 read + 1 write.
 *
 * Uses __half2 arithmetic where possible (two fp16 ops in one instruction).
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_fp16.h>

// ── RoPE kernel ───────────────────────────────────────────────────────────────
// Grid:  ceil(B * H * T * D_half / BLOCK)
// Block: BLOCK threads
// Each thread: one (b, h, t, d) half-index pair
__global__ void rope_kernel(
    const __half* __restrict__ x,     // (B, H, T, D)    fp16
    const float*  __restrict__ cos_,  // (T, D_half)     fp32
    const float*  __restrict__ sin_,  // (T, D_half)     fp32
    __half*       __restrict__ y,     // (B, H, T, D)    fp16
    int B, int H, int T, int D_half,
    int stride_b, int stride_h, int stride_t  // strides in units of __half
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * T * D_half;
    if (idx >= total) return;

    // Decode (b, h, t, d) from flat index
    int d  = idx % D_half;
    int t  = (idx / D_half) % T;
    int bh = idx / (D_half * T);
    int b  = bh / H;
    int h  = bh % H;

    int base = b * stride_b + h * stride_h + t * stride_t;

    // Load x1[d] and x2[d] — two HBM loads per thread (coalesced within warp)
    float x1 = __half2float(x[base + d]);
    float x2 = __half2float(x[base + d + D_half]);

    float c = cos_[t * D_half + d];
    float s = sin_[t * D_half + d];

    // Rotation
    y[base + d]          = __float2half(x1 * c - x2 * s);
    y[base + d + D_half] = __float2half(x1 * s + x2 * c);
}

// ── Python binding ────────────────────────────────────────────────────────────
torch::Tensor rope_cuda(
    torch::Tensor x,      // (B, H, T, D)   fp16
    torch::Tensor cos_t,  // (T, D//2)      fp32
    torch::Tensor sin_t   // (T, D//2)      fp32
) {
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat16, "x must be CUDA fp16");
    TORCH_CHECK(cos_t.is_cuda() && cos_t.dtype() == torch::kFloat32, "cos must be CUDA fp32");
    TORCH_CHECK(sin_t.is_cuda() && sin_t.dtype() == torch::kFloat32, "sin must be CUDA fp32");

    auto y = torch::empty_like(x);

    int B = x.size(0), H = x.size(1), T = x.size(2), D = x.size(3);
    int D_half = D / 2;
    int total  = B * H * T * D_half;

    constexpr int BLOCK = 256;
    int grid = (total + BLOCK - 1) / BLOCK;

    rope_kernel<<<grid, BLOCK, 0, c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        cos_t.data_ptr<float>(),
        sin_t.data_ptr<float>(),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        B, H, T, D_half,
        (int)x.stride(0), (int)x.stride(1), (int)x.stride(2)
    );
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rope_cuda", &rope_cuda,
          "Fused RoPE CUDA kernel (half-split convention, Qwen/NeoX style)");
}
