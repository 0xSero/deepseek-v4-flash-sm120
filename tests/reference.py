"""Pure-PyTorch reference for the DeepSeek-V4-Flash sparse decode kernel.

Used both as a unit-test oracle and for debugging the SM_120 kernel.
Semantics match sglang's `flash_mla_with_kvcache_entrypoint` call for
DeepSeek-V4-Flash (sglang.deepseek_v4_memory_pool + flash_mla MODEL1).

Per-page layout (page_block_size = P):
  [0 .. P*576)              nope_rope section
      per token (576 B):
           0..447  FP8 e4m3 NoPE        (448 B)
         448..575  BF16 RoPE            (64 values = 128 B)
  [P*576 .. P*(576+8))      scale section
      per token  :  7 UE8M0 bytes + 1 pad (8 bytes)
                    scale_i = 2**(byte_i - 127), covers NoPE[i*64 .. (i+1)*64)

Q / output : BF16 [b, s_q, h_q, 512]  (448 NoPE || 64 RoPE)

Attention: softmax(QK^T * sm_scale) over the gathered `indices`, with
optional `topk_length` mask and optional `attn_sink` (log-domain mix).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

HEAD_DIM_NOPE = 448
HEAD_DIM_ROPE = 64
HEAD_DIM_QK   = HEAD_DIM_NOPE + HEAD_DIM_ROPE  # 512
HEAD_DIM_V    = HEAD_DIM_QK                     # 512
QUANT_TILE         = 64
NUM_ACTIVE_SCALES  = HEAD_DIM_NOPE // QUANT_TILE   # 7
NUM_SCALE_SLOTS    = 8                              # 7 active + 1 pad
NOPE_BYTES         = HEAD_DIM_NOPE                  # 448
ROPE_BYTES         = HEAD_DIM_ROPE * 2              # 128
NOPE_ROPE_BYTES    = NOPE_BYTES + ROPE_BYTES        # 576
BYTES_PER_TOKEN    = NOPE_ROPE_BYTES + NUM_SCALE_SLOTS  # 584 (cosmetic)


def _ue8m0_to_scale(b: torch.Tensor) -> torch.Tensor:
    """uint8 tensor -> fp32 scale tensor via 2**(b - 127)."""
    return torch.pow(2.0, b.to(torch.float32) - 127.0)


def _unpack_nope_rope_scale(nope_bytes: torch.Tensor,
                             rope_bytes: torch.Tensor,
                             scale_u8: torch.Tensor) -> torch.Tensor:
    """Assemble the dequantised BF16 [N, 512] rows from the separated
    nope / rope / scale sections produced by sglang's quant_k_cache_v4."""
    n = nope_bytes.size(0)
    nope_fp8 = nope_bytes.contiguous().view(dtype=torch.float8_e4m3fn)
    rope = rope_bytes.contiguous().view(dtype=torch.bfloat16).view(n, HEAD_DIM_ROPE)

    scales = _ue8m0_to_scale(scale_u8[:, :NUM_ACTIVE_SCALES]).view(n, NUM_ACTIVE_SCALES)
    nope = nope_fp8.float().view(n, NUM_ACTIVE_SCALES, QUANT_TILE)
    nope = nope * scales.unsqueeze(-1)
    nope = nope.view(n, HEAD_DIM_NOPE).to(torch.bfloat16)
    return torch.cat([nope, rope], dim=-1)


def _pages_to_bf16(kv_pages: torch.Tensor, page_block: int) -> torch.Tensor:
    """Decode every token of every page into a dense [num_blocks*page, 512] BF16
    tensor.  `kv_pages` is a uint8 tensor of shape
      [num_blocks, page_block*NOPE_ROPE_BYTES + page_block*NUM_SCALE_SLOTS]
    i.e. the raw per-page byte arena, with nope_rope section first and
    scale section second (flash_mla MODEL1 layout)."""
    num_blocks = kv_pages.size(0)
    total_tokens = num_blocks * page_block

    nope_rope_section = kv_pages[:, : page_block * NOPE_ROPE_BYTES]
    scale_section = kv_pages[:, page_block * NOPE_ROPE_BYTES :
                             page_block * NOPE_ROPE_BYTES +
                             page_block * NUM_SCALE_SLOTS]

    nope_rope = nope_rope_section.contiguous().view(total_tokens, NOPE_ROPE_BYTES)
    nope_bytes = nope_rope[:, :NOPE_BYTES]
    rope_bytes = nope_rope[:, NOPE_BYTES:]
    scale_u8 = scale_section.contiguous().view(total_tokens, NUM_SCALE_SLOTS)
    return _unpack_nope_rope_scale(nope_bytes, rope_bytes, scale_u8)


def sparse_decode_reference(
    q: torch.Tensor,                 # [b, s_q, h_q, 512] bf16
    kv_pages: torch.Tensor,          # [num_blocks, page_bytes] uint8, MODEL1 layout
    indices: torch.Tensor,           # [b, s_q, topk] int32
    topk_length: Optional[torch.Tensor],
    attn_sink: Optional[torch.Tensor],
    sm_scale: float,
    page_block: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    b, s_q, h_q, d_qk = q.shape
    topk = indices.size(-1)
    assert d_qk == HEAD_DIM_QK
    assert kv_pages.dtype == torch.uint8
    expected_bytes = page_block * NOPE_ROPE_BYTES + page_block * NUM_SCALE_SLOTS
    assert kv_pages.size(-1) >= expected_bytes, (
        f"kv_pages last-dim {kv_pages.size(-1)} < required {expected_bytes}"
    )

    kv_bf16 = _pages_to_bf16(kv_pages, page_block)  # [N, 512]

    out = torch.zeros(b, s_q, h_q, HEAD_DIM_V, dtype=torch.bfloat16, device=q.device)
    lse_out = torch.empty(b, s_q, h_q, dtype=torch.float32, device=q.device)

    for bi in range(b):
        for si in range(s_q):
            idx = indices[bi, si]  # [topk]
            length = int(topk_length[bi].item()) if topk_length is not None else topk
            length = max(length, 1)
            valid = (idx >= 0) & (idx < kv_bf16.size(0))
            valid_idx = torch.where(valid, idx, torch.zeros_like(idx))
            k = kv_bf16[valid_idx.long()]  # [topk, 512]
            k = torch.where(valid.unsqueeze(-1), k, torch.zeros_like(k))
            pos = torch.arange(topk, device=q.device)
            length_mask = pos < length
            q_bs = q[bi, si].float()
            logits = (q_bs @ k.float().transpose(0, 1)) * sm_scale
            logits = torch.where(length_mask.unsqueeze(0), logits,
                                 torch.full_like(logits, float("-inf")))
            lse = torch.logsumexp(logits, dim=-1)
            p = torch.softmax(logits, dim=-1)
            v = k.float()  # MLA: V == K (full 512)
            o = p @ v
            if attn_sink is not None:
                sink = attn_sink.float()
                sink_scale = 1.0 / (1.0 + torch.exp(sink - lse))
                o = o * sink_scale.unsqueeze(-1)
                m = torch.maximum(lse, sink)
                lse = m + torch.log(torch.exp(lse - m) + torch.exp(sink - m))
            out[bi, si] = o.to(torch.bfloat16)
            lse_out[bi, si] = lse
    return out, lse_out


def make_fake_batch(
    b: int = 2,
    h_q: int = 64,
    num_blocks: int = 4,
    page_block: int = 64,
    topk: int = 256,
    seed: int = 0,
    device: str = "cuda",
):
    """Build a randomised batch whose packed KV arena matches the sglang
    DSv4-Flash layout (MODEL1): per page
        [page_block * 576 B nope_rope][page_block * 8 B scales]

    Returns
    -------
    q           : BF16 [b, 1, h_q, 512]
    kv_pages    : uint8 [num_blocks, page_block * (576 + 8)]
                  (raw per-page arena; pass to kernel + reference)
    kv_view     : uint8 [num_blocks, page_block, 1, 584]
                  (cosmetic sglang-style 4-D view sharing storage)
    idx         : int32 [b, 1, topk]
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(b, 1, h_q, HEAD_DIM_QK, dtype=torch.bfloat16,
                    device=device, generator=gen) * 0.1

    # NoPE bytes (FP8 e4m3fn).
    nope_bf16 = torch.randn(num_blocks, page_block, HEAD_DIM_NOPE,
                            dtype=torch.bfloat16, device=device, generator=gen) * 0.5
    nope_fp8 = nope_bf16.to(torch.float8_e4m3fn)
    nope_bytes = nope_fp8.view(dtype=torch.uint8)           # [B, P, 448]

    # RoPE bytes (BF16 -> 128 B per token).
    rope_bf16 = torch.randn(num_blocks, page_block, HEAD_DIM_ROPE,
                            dtype=torch.bfloat16, device=device, generator=gen) * 0.05
    rope_bytes = rope_bf16.view(dtype=torch.uint8).view(num_blocks, page_block,
                                                        ROPE_BYTES)

    # Concatenate per-token nope|rope -> 576 bytes, then flatten page.
    nope_rope = torch.cat([nope_bytes, rope_bytes], dim=-1)  # [B, P, 576]
    assert nope_rope.size(-1) == NOPE_ROPE_BYTES
    nope_rope_section = nope_rope.contiguous().view(num_blocks,
                                                    page_block * NOPE_ROPE_BYTES)

    # Scale section: 7 UE8M0 bytes + 1 pad, per token, laid out per page.
    scale_exp = (125 + torch.randint(0, 4,
                                     (num_blocks, page_block, NUM_ACTIVE_SCALES),
                                     device=device, generator=gen,
                                     dtype=torch.int32)).to(torch.uint8)
    pad = torch.zeros(num_blocks, page_block, 1, dtype=torch.uint8, device=device)
    scales = torch.cat([scale_exp, pad], dim=-1)             # [B, P, 8]
    assert scales.size(-1) == NUM_SCALE_SLOTS
    scale_section = scales.contiguous().view(num_blocks,
                                             page_block * NUM_SCALE_SLOTS)

    # Full per-page arena: nope_rope then scales.
    kv_pages = torch.cat([nope_rope_section, scale_section], dim=-1).contiguous()
    # Cosmetic 4-D view that matches sglang's .view(num_blocks, P, 1, 584).
    # Memory backing is NOT interleaved per-token; only the *shape* matches.
    kv_view = kv_pages.view(num_blocks, page_block, 1, BYTES_PER_TOKEN)

    total_tokens = num_blocks * page_block
    idx = torch.randint(-1, total_tokens, (b, 1, topk), dtype=torch.int32,
                        device=device, generator=gen)
    return q, kv_pages, kv_view, idx
