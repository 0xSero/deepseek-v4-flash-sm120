// Minimal CUTLASS shim.
//
// The SM_120 sparse-decode kernel only needs the `cutlass::bfloat16_t`
// type — it is layout-compatible with CUDA's `__nv_bfloat16`.  Shipping
// the full 150 MB CUTLASS tree only to get one typedef is excessive, so
// we just alias it here.  If you'd rather build against the real CUTLASS,
// drop its `include/` directory on the include path BEFORE this header
// is seen; the `#ifndef CUTLASS_BFLOAT16_H` guard then makes the real
// definition win.
#pragma once

#ifndef CUTLASS_BFLOAT16_H
#define CUTLASS_BFLOAT16_H
#include <cuda_bf16.h>

namespace cutlass {
using bfloat16_t = __nv_bfloat16;
}  // namespace cutlass
#endif  // CUTLASS_BFLOAT16_H
