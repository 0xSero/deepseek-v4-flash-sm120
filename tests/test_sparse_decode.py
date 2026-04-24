"""Correctness regression for the SM_120 sparse decode kernel."""
from __future__ import annotations

import math
import os

import pytest
import torch

from deepseek_v4_kernel import sparse_decode_fwd

from reference import make_fake_batch, sparse_decode_reference


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA device required",
)


def _maybe_skip_not_sm120():
    major, _ = torch.cuda.get_device_capability(0)
    if major != 12:
        pytest.skip(f"SM_120 required; got {torch.cuda.get_device_capability(0)}")


@pytest.mark.parametrize("h_q", [64, 128])
@pytest.mark.parametrize("topk", [64, 256, 2048])
def test_matches_reference(h_q: int, topk: int):
    _maybe_skip_not_sm120()
    torch.manual_seed(0)
    page_block = 64
    q, kv_pages, kv_view, idx = make_fake_batch(
        b=2, h_q=h_q, num_blocks=8, page_block=page_block, topk=topk,
    )
    sm_scale = 1.0 / math.sqrt(512)

    out_ref, lse_ref = sparse_decode_reference(
        q, kv_pages, idx, topk_length=None, attn_sink=None,
        sm_scale=sm_scale, page_block=page_block,
    )
    out, lse, _, _ = sparse_decode_fwd(
        q, kv_view, idx,
        None, None, None, None, None, None, None,
        512, sm_scale,
    )
    torch.testing.assert_close(out.float(), out_ref.float(), atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(lse.squeeze(-1), lse_ref.squeeze(1),
                               atol=1e-3, rtol=1e-3)


def test_topk_length_mask():
    _maybe_skip_not_sm120()
    torch.manual_seed(1)
    page_block = 64
    q, kv_pages, kv_view, idx = make_fake_batch(
        b=2, h_q=64, num_blocks=4, page_block=page_block, topk=512,
    )
    tk = torch.tensor([64, 32], dtype=torch.int32, device="cuda")
    sm_scale = 1.0 / math.sqrt(512)
    out_ref, _ = sparse_decode_reference(
        q, kv_pages, idx, topk_length=tk, attn_sink=None,
        sm_scale=sm_scale, page_block=page_block,
    )
    out, _, _, _ = sparse_decode_fwd(q, kv_view, idx, tk, None, None, None,
                                     None, None, None, 512, sm_scale)
    torch.testing.assert_close(out.float(), out_ref.float(), atol=5e-3, rtol=5e-3)
