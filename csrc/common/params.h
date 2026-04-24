// DeepSeek-V4 Flash MLA sparse decode: params & shared types
// Compatible with the upstream flash_mla sparse_decode_fwd API surface so our
// SM_120 kernel can be dropped into `flash_mla.cuda.sparse_decode_fwd` paths.

#pragma once

#include <cuda_runtime.h>
#include "common/cutlass_shim.h"

namespace dsv4_kernel {

enum class ModelType : int {
    V32 = 0,     // head_dim_qk=576, scale tile=128
    MODEL1 = 1,  // head_dim_qk=512, scale tile=64
};

// Decoding schedule metadata (one entry per SM-part).
// Field ordering matches upstream flash_mla::DecodingSchedMeta so the
// tile_scheduler tensor is binary compatible.
struct __align__(4 * 8) DecodingSchedMeta {
    int begin_req_idx;
    int end_req_idx;
    int begin_block_idx;
    int end_block_idx;
    int begin_split_idx;
    int is_first_req_splitted;
    int is_last_req_splitted;
    int _pad[1];
};
static constexpr int DecodingSchedMetaSize = sizeof(DecodingSchedMeta);
static_assert(DecodingSchedMetaSize % 4 == 0,
              "DecodingSchedMeta must be a multiple of int32");

struct SparseAttnDecodeParams {
    int b;
    int s_q;
    int h_q;
    int h_kv;
    int d_qk;  // 576 for V32, 512 for MODEL1
    int d_v;   // always 512
    float sm_scale;
    float sm_scale_div_log2;
    int num_blocks;
    int page_block_size;
    int topk;
    ModelType model_type;

    cutlass::bfloat16_t* __restrict__ q;   // [b, s_q, h_q, d_qk]
    cutlass::bfloat16_t* __restrict__ kv;  // [num_blocks, page_block_size, h_kv, bytes_per_token]
    int* __restrict__ indices;             // [b, s_q, topk]
    int* __restrict__ topk_length;         // [b] (may be nullptr)
    float* __restrict__ attn_sink;         // [h_q] (may be nullptr)

    float* __restrict__ lse;                 // [b, s_q, h_q]
    cutlass::bfloat16_t* __restrict__ out;   // [b, s_q, h_q, d_v]

    int extra_num_blocks;
    int extra_page_block_size;
    int extra_topk;
    cutlass::bfloat16_t* __restrict__ extra_kv;
    int* __restrict__ extra_indices;
    int* __restrict__ extra_topk_length;

    int stride_q_b;
    int stride_q_s_q;
    int stride_q_h_q;
    int stride_kv_block;
    int stride_kv_row;
    int stride_indices_b;
    int stride_indices_s_q;
    int stride_lse_b;
    int stride_lse_s_q;
    int stride_o_b;
    int stride_o_s_q;
    int stride_o_h_q;
    int stride_extra_kv_block;
    int stride_extra_kv_row;
    int stride_extra_indices_b;
    int stride_extra_indices_s_q;

    cudaStream_t stream;

    // Split-KV bookkeeping (combine kernel writes the final output when
    // num_sm_parts > 1). Kept for API parity with flash_mla; our initial
    // SM_120 implementation processes an entire request per CTA and thus
    // always emits num_splits == 1, writing directly to `out` / `lse`.
    float* __restrict__ lse_accum;
    float* __restrict__ o_accum;
    int stride_lse_accum_split;
    int stride_lse_accum_s_q;
    int stride_o_accum_split;
    int stride_o_accum_s_q;
    int stride_o_accum_h_q;
    DecodingSchedMeta* __restrict__ tile_scheduler_metadata_ptr;
    int* __restrict__ num_splits_ptr;
    int num_sm_parts;
};

// Minimal runtime arch probe (matches upstream naming enough for dispatch).
struct Arch {
    int major;
    int minor;
    int num_sms;

    Arch() {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, dev);
        major = prop.major;
        minor = prop.minor;
        num_sms = prop.multiProcessorCount;
    }

    bool is_sm90a() const { return major == 9 && minor == 0; }
    bool is_sm100f() const { return major == 10; }
    bool is_sm120() const { return major == 12; }  // Blackwell workstation (RTX Pro 6000 / RTX 50xx)
};

}  // namespace dsv4_kernel
