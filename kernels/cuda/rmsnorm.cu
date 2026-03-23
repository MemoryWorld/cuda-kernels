/*
 * RMSNorm — fused CUDA C++ kernel
 *
 * Algorithm (1-pass, shared memory):
 *   1. Load entire row (one token's hidden vector) into shared memory  → 1 HBM read
 *   2. Compute sum-of-squares entirely from SRAM using two-level reduction:
 *        - per-thread partial sum → warp reduction (__shfl_down_sync)
 *        - warp sums → one more warp reduction in shared memory
 *   3. Broadcast rms_inv to all threads
 *   4. Normalise using shared-memory data, scale, write output           → 1 HBM write
 *
 * Net HBM traffic: 1 read (x) + 1 read (w) + 1 write (y)  — same as Triton.
 * Shared memory per block: N × sizeof(__half)  (3584×2 = 7 KB for Qwen2.5-7B)
 *
 * Compile: loaded via torch.utils.cpp_extension.load in cuda_vs_triton.py
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_fp16.h>

// ── Warp-level sum reduction ──────────────────────────────────────────────────
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// ── RMSNorm kernel ────────────────────────────────────────────────────────────
// Grid:  (rows,)     — one block per token
// Block: (BLOCK,)    — BLOCK threads collaborate on one row
// Smem:  N × fp16   — full row cached in SRAM
template <int BLOCK>
__global__ void rms_norm_kernel(
    const __half* __restrict__ x,   // (rows, N)  fp16  input
    const __half* __restrict__ w,   // (N,)       fp16  scale
    __half*       __restrict__ y,   // (rows, N)  fp16  output
    int N, float eps
) {
    // Shared memory layout:
    //   [0 .. N-1]        : fp16 row data
    //   [N .. N + BLOCK/32 - 1] : warp partial sums (fp32, reinterpret)
    extern __shared__ __half smem[];
    float* warp_sums = reinterpret_cast<float*>(smem + N);  // BLOCK/32 floats

    const int row = blockIdx.x;
    const __half* xr = x + row * N;
    __half*       yr = y + row * N;

    // ── Step 1: load row into shared memory ──────────────────────────────────
    for (int i = threadIdx.x; i < N; i += BLOCK)
        smem[i] = xr[i];
    __syncthreads();

    // ── Step 2: compute sum of squares ───────────────────────────────────────
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < N; i += BLOCK) {
        float xi = __half2float(smem[i]);
        sum_sq += xi * xi;
    }

    // Warp-level reduction
    sum_sq = warp_reduce_sum(sum_sq);
    if (threadIdx.x % 32 == 0)
        warp_sums[threadIdx.x / 32] = sum_sq;
    __syncthreads();

    // Block-level reduction (first warp finalises)
    // Only BLOCK/32 threads enter — use exact mask to avoid deadlock on sm_70+
    if (threadIdx.x < BLOCK / 32) {
        float s = warp_sums[threadIdx.x];
        // mask covers exactly the (BLOCK/32) active lanes in this warp
        constexpr unsigned NWARPS = BLOCK / 32;
        constexpr unsigned MASK   = (NWARPS >= 32) ? 0xffffffff : ((1u << NWARPS) - 1u);
        #pragma unroll
        for (int offset = NWARPS / 2; offset > 0; offset >>= 1)
            s += __shfl_down_sync(MASK, s, offset);
        if (threadIdx.x == 0)
            warp_sums[0] = rsqrtf(s / N + eps);
    }
    __syncthreads();

    float rms_inv = warp_sums[0];

    // ── Step 3: normalise and write ──────────────────────────────────────────
    for (int i = threadIdx.x; i < N; i += BLOCK) {
        float xi = __half2float(smem[i]);
        float wi = __half2float(w[i]);
        yr[i] = __float2half(xi * rms_inv * wi);
    }
}

// ── Python binding ────────────────────────────────────────────────────────────
torch::Tensor rms_norm_cuda(
    torch::Tensor x,    // (rows, N)  fp16, contiguous
    torch::Tensor w,    // (N,)       fp16
    double eps
) {
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat16, "x must be CUDA fp16");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(w.is_cuda() && w.dtype() == torch::kFloat16, "w must be CUDA fp16");

    auto y = torch::empty_like(x);
    const int rows = x.size(0);
    const int N    = x.size(1);

    constexpr int BLOCK = 256;
    // Shared mem: N fp16 for row data + (BLOCK/32) fp32 for warp sums
    const size_t smem_bytes = N * sizeof(__half) + (BLOCK / 32) * sizeof(float);

    rms_norm_kernel<BLOCK><<<rows, BLOCK, smem_bytes, c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(w.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        N, static_cast<float>(eps)
    );
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rms_norm_cuda", &rms_norm_cuda,
          "Fused RMSNorm: 1-pass SRAM reduction, warp __shfl_down_sync");
}
