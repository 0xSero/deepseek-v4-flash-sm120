#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace dsv4_kernel {

static constexpr float LOG_2_E = 1.44269504f;

#define DSV4_CHECK_CUDA(call)                                                  \
    do {                                                                       \
        cudaError_t _status = (call);                                          \
        if (_status != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(_status));                              \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

#define DSV4_CHECK_CUDA_LAUNCH() DSV4_CHECK_CUDA(cudaGetLastError())

template <typename T>
__host__ __device__ inline T ceil_div(T a, T b) {
    return (a + b - T{1}) / b;
}

}  // namespace dsv4_kernel
