// Forward declaration callable from plain C++ translation units.
#pragma once

#include "common/params.h"

namespace dsv4_kernel {
namespace sm120 {

void launch_dsv4_sparse_decode_v32(const SparseAttnDecodeParams &params);

}  // namespace sm120
}  // namespace dsv4_kernel
