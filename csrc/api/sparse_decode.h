#pragma once

#include <ATen/core/Tensor.h>
#include <optional>
#include <tuple>

namespace dsv4_kernel {

// Python-visible signature identical to flash_mla.cuda.sparse_decode_fwd.
std::tuple<at::Tensor, at::Tensor,
           std::optional<at::Tensor>, std::optional<at::Tensor>>
sparse_decode_fwd(
    const at::Tensor &q,
    const at::Tensor &kv,
    const at::Tensor &indices,
    const std::optional<at::Tensor> &topk_length,
    const std::optional<at::Tensor> &attn_sink,
    std::optional<at::Tensor> tile_scheduler_metadata,
    std::optional<at::Tensor> num_splits,
    const std::optional<at::Tensor> &extra_kv,
    const std::optional<at::Tensor> &extra_indices,
    const std::optional<at::Tensor> &extra_topk_length,
    int d_v,
    double sm_scale);

}  // namespace dsv4_kernel
