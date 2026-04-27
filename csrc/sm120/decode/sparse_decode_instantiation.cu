// CUDA translation unit: pulls in the kernel template and provides the
// launch entry point seen by the C++-only api/sparse_decode.cpp.

#include "sm120/decode/sparse_decode.h"
#include "sm120/decode/sparse_decode_kernel.cuh"

namespace dsv4_kernel {
namespace sm120 {

void launch_dsv4_sparse_decode_v32(const SparseAttnDecodeParams &params) {
    int num_head_blocks = ceil_div(params.h_q, BLOCK_M_HEADS);
    dim3 grid(static_cast<unsigned>(params.b * params.s_q),
              static_cast<unsigned>(num_head_blocks), 1);
    dim3 block(NUM_THREADS, 1, 1);
    size_t smem_size = sizeof(SmemLayout);
    // cudaFuncSetAttribute is a runtime API, not a stream op — it can't be
    // captured into a CUDA graph. Run it exactly once before any capture.
    static const bool _smem_attr_set = []() {
        DSV4_CHECK_CUDA(cudaFuncSetAttribute(
            dsv4_sparse_decode_kernel<BLOCK_M_HEADS>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(sizeof(SmemLayout))));
        return true;
    }();
    (void)_smem_attr_set;
    dsv4_sparse_decode_kernel<BLOCK_M_HEADS>
        <<<grid, block, smem_size, params.stream>>>(params);
    DSV4_CHECK_CUDA_LAUNCH();
}

}  // namespace sm120
}  // namespace dsv4_kernel
