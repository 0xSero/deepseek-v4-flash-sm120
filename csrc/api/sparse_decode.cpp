#include "api/sparse_decode.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdlib>
#include <limits>

#include "common/defines.h"
#include "common/params.h"
#include "sm120/decode/sparse_decode.h"

namespace dsv4_kernel {

namespace {
inline int strideToInt(int64_t s) {
    TORCH_CHECK(s <= std::numeric_limits<int>::max(), "stride overflow");
    return static_cast<int>(s);
}
template <typename T>
T *maybe_data_ptr(const std::optional<at::Tensor> &t) {
    return t.has_value() ? reinterpret_cast<T *>(t->data_ptr()) : nullptr;
}
}  // namespace

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
    double sm_scale) {
    // ----- Basic shape/dtype checks (mirror upstream flash_mla) -----
    TORCH_CHECK(q.is_cuda() && kv.is_cuda(), "q and kv must be CUDA tensors");
    TORCH_CHECK(q.dim() == 4, "q must be 4-D [b, s_q, h_q, d_qk]");
    TORCH_CHECK(kv.dim() == 4, "kv must be 4-D [num_blocks, page_block, h_kv, bytes]");
    TORCH_CHECK(indices.dim() == 3, "indices must be 3-D [b, s_q, topk]");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16, "q must be bf16");
    TORCH_CHECK(kv.scalar_type() == at::kFloat8_e4m3fn ||
                    kv.scalar_type() == at::kByte ||
                    kv.scalar_type() == at::kChar,
                "kv must be fp8_e4m3 / uint8 / int8 (packed V32 layout)");
    TORCH_CHECK(indices.scalar_type() == at::kInt, "indices must be int32");

    const int b = q.size(0);
    const int s_q = q.size(1);
    const int h_q = q.size(2);
    const int d_qk = q.size(3);
    const int num_blocks = kv.size(0);
    const int page_block_size = kv.size(1);
    const int h_kv = kv.size(2);
    const int k_bytes = kv.size(3);
    const int topk = indices.size(2);

    TORCH_CHECK(h_kv == 1, "Only MQA (h_kv=1) is supported for sparse decode");
    TORCH_CHECK(d_qk == 512,
                "DeepSeek-V4-Flash expects d_qk=512 "
                "(448 NoPE + 64 RoPE) but got ", d_qk);
    TORCH_CHECK(k_bytes == 584,
                "DeepSeek-V4-Flash expects 584 bytes / token "
                "(448 FP8 NoPE + 7 UE8M0 scales + 1 pad + 64 BF16 RoPE), got ",
                k_bytes);
    TORCH_CHECK(d_v == 512, "Only d_v=512 supported");
    TORCH_CHECK(s_q == 1,
                "Only s_q=1 supported on SM_120 path currently; MTP batches "
                "should be flattened into `b`.");

    // Extra KV sanity.
    int extra_num_blocks = 0, extra_page_block_size = 0, extra_topk = 0;
    if (extra_kv.has_value()) {
        TORCH_CHECK(extra_indices.has_value(),
                    "extra_indices required when extra_kv is provided");
        extra_num_blocks = extra_kv->size(0);
        extra_page_block_size = extra_kv->size(1);
        extra_topk = extra_indices->size(-1);
    }

    // Allocate outputs.
    at::cuda::CUDAGuard g(q.device());
    auto opts = q.options();
    auto out = at::empty({b, s_q, h_q, d_v}, opts);
    auto lse = at::empty({b, s_q, h_q}, opts.dtype(at::kFloat));

    // Pack params.
    SparseAttnDecodeParams p{};
    p.b = b;
    p.s_q = s_q;
    p.h_q = h_q;
    p.h_kv = h_kv;
    p.d_qk = d_qk;
    p.d_v = d_v;
    p.sm_scale = static_cast<float>(sm_scale);
    p.sm_scale_div_log2 = p.sm_scale * LOG_2_E;
    p.num_blocks = num_blocks;
    p.page_block_size = page_block_size;
    p.topk = topk;
    p.model_type = ModelType::V32;

    p.q = reinterpret_cast<cutlass::bfloat16_t *>(q.data_ptr());
    p.kv = reinterpret_cast<cutlass::bfloat16_t *>(kv.data_ptr());
    p.indices = indices.data_ptr<int>();
    p.topk_length = topk_length.has_value() ? topk_length->data_ptr<int>() : nullptr;
    p.attn_sink = attn_sink.has_value() ? attn_sink->data_ptr<float>() : nullptr;

    p.lse = lse.data_ptr<float>();
    p.out = reinterpret_cast<cutlass::bfloat16_t *>(out.data_ptr());

    p.extra_num_blocks = extra_num_blocks;
    p.extra_page_block_size = extra_page_block_size;
    p.extra_topk = extra_topk;
    p.extra_kv = maybe_data_ptr<cutlass::bfloat16_t>(extra_kv);
    p.extra_indices = maybe_data_ptr<int>(extra_indices);
    p.extra_topk_length = maybe_data_ptr<int>(extra_topk_length);

    p.stride_q_b = strideToInt(q.stride(0));
    p.stride_q_s_q = strideToInt(q.stride(1));
    p.stride_q_h_q = strideToInt(q.stride(2));
    p.stride_kv_block = strideToInt(kv.stride(0));
    p.stride_kv_row = strideToInt(kv.stride(1));
    p.stride_indices_b = strideToInt(indices.stride(0));
    p.stride_indices_s_q = strideToInt(indices.stride(1));
    p.stride_lse_b = strideToInt(lse.stride(0));
    p.stride_lse_s_q = strideToInt(lse.stride(1));
    p.stride_o_b = strideToInt(out.stride(0));
    p.stride_o_s_q = strideToInt(out.stride(1));
    p.stride_o_h_q = strideToInt(out.stride(2));
    p.stride_extra_kv_block =
        extra_kv.has_value() ? strideToInt(extra_kv->stride(0)) : 0;
    p.stride_extra_kv_row =
        extra_kv.has_value() ? strideToInt(extra_kv->stride(1)) : 0;
    p.stride_extra_indices_b =
        extra_indices.has_value() ? strideToInt(extra_indices->stride(0)) : 0;
    p.stride_extra_indices_s_q =
        extra_indices.has_value() ? strideToInt(extra_indices->stride(1)) : 0;

    p.stream = at::cuda::getCurrentCUDAStream();

    // No split-KV in the SM_120 path yet.  Return empty tile scheduler tensors
    // that match the flash_mla API shape so callers can keep reusing them.
    if (!tile_scheduler_metadata.has_value()) {
        tile_scheduler_metadata =
            at::empty({1, DecodingSchedMetaSize / 4}, opts.dtype(at::kInt));
        num_splits = at::empty({b + 1}, opts.dtype(at::kInt));
    }
    p.tile_scheduler_metadata_ptr =
        reinterpret_cast<DecodingSchedMeta *>(
            tile_scheduler_metadata->data_ptr());
    p.num_splits_ptr = num_splits->data_ptr<int>();
    p.num_sm_parts = 1;

    Arch arch;
    if (arch.is_sm120() || std::getenv("DSV4_KERNEL_FORCE") != nullptr) {
        sm120::launch_dsv4_sparse_decode_v32(p);
    } else {
        TORCH_CHECK(false,
                    "dsv4_kernel.sparse_decode_fwd only implements SM_120 "
                    "(Blackwell workstation). Current device major=",
                    arch.major, " minor=", arch.minor);
    }

    return std::make_tuple(out, lse.transpose(1, 2).contiguous(),
                           tile_scheduler_metadata, num_splits);
}

}  // namespace dsv4_kernel
