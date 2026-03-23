/*
 * SwiGLU — fused CUDA C++ kernel
 *
 * SwiGLU(x) = silu(gate) ⊙ up
 *           = gate · sigmoid(gate) · up
 *
 * PyTorch eager: two kernel launches — silu(gate) writes a temp tensor,
 *   then temp * up writes the output. Two HBM reads of gate, two writes.
 *
 * This kernel: reads gate and up ONCE, fuses silu + multiply, writes once.
 *   HBM: 2 reads + 1 write (vs 3 reads + 2 writes in eager).
 *
 * Uses __half2 (vectorised fp16): processes 2 elements per thread,
 * doubling effective memory bandwidth throughput for small ffn_dims.
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_fp16.h>

// ── silu ──────────────────────────────────────────────────────────────────────
__device__ __forceinline__ float silu_f(float x) {
    return x / (1.0f + __expf(-x));   // __expf: fast single-precision exp
}

// ── SwiGLU kernel (vectorised with half2) ─────────────────────────────────────
// Processes 2 fp16 elements per thread using __half2 arithmetic.
// Requires ffn_dim to be even (always true in practice).
__global__ void swiglu_kernel_h2(
    const __half2* __restrict__ gate,   // (N/2,) packed fp16×2
    const __half2* __restrict__ up,     // (N/2,) packed fp16×2
    __half2*       __restrict__ out,    // (N/2,) packed fp16×2
    int N2  // total __half2 elements = numel / 2
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N2) return;

    __half2 g2 = gate[idx];
    __half2 u2 = up[idx];

    // Unpack to float2 for computation
    float2 gf = __half22float2(g2);
    float2 uf = __half22float2(u2);

    // Fused silu × up  (scalar ops, fully unrolled by compiler)
    float2 of;
    of.x = silu_f(gf.x) * uf.x;
    of.y = silu_f(gf.y) * uf.y;

    out[idx] = __float22half2_rn(of);   // round-to-nearest pack
}

// Scalar fallback for odd total sizes (rare in practice)
__global__ void swiglu_kernel_scalar(
    const __half* __restrict__ gate,
    const __half* __restrict__ up,
    __half*       __restrict__ out,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float g = __half2float(gate[idx]);
    float u = __half2float(up[idx]);
    out[idx] = __float2half(silu_f(g) * u);
}

// ── Python binding ────────────────────────────────────────────────────────────
torch::Tensor swiglu_cuda(torch::Tensor gate, torch::Tensor up) {
    TORCH_CHECK(gate.is_cuda() && gate.dtype() == torch::kFloat16, "gate must be CUDA fp16");
    TORCH_CHECK(up.is_cuda()   && up.dtype()   == torch::kFloat16, "up must be CUDA fp16");
    TORCH_CHECK(gate.is_contiguous() && up.is_contiguous(), "inputs must be contiguous");

    auto out = torch::empty_like(gate);
    int N = gate.numel();

    constexpr int BLOCK = 256;

    if (N % 2 == 0) {
        // Vectorised path: half2 — 2 elements per thread
        int N2   = N / 2;
        int grid = (N2 + BLOCK - 1) / BLOCK;
        swiglu_kernel_h2<<<grid, BLOCK, 0, c10::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __half2*>(gate.data_ptr<at::Half>()),
            reinterpret_cast<const __half2*>(up.data_ptr<at::Half>()),
            reinterpret_cast<__half2*>(out.data_ptr<at::Half>()),
            N2
        );
    } else {
        // Scalar fallback
        int grid = (N + BLOCK - 1) / BLOCK;
        swiglu_kernel_scalar<<<grid, BLOCK, 0, c10::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __half*>(gate.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(up.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
            N
        );
    }
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("swiglu_cuda", &swiglu_cuda,
          "Fused SwiGLU CUDA kernel: half2 vectorised silu(gate)*up");
}
