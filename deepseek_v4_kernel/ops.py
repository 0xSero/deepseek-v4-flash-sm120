"""Python wrappers with a `flash_mla.cuda.sparse_decode_fwd`-compatible API."""
from __future__ import annotations

from typing import Optional, Tuple

import torch

try:
    from . import cuda as _cuda  # type: ignore[attr-defined]
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "deepseek_v4_kernel.cuda not built. Run `pip install -e .` inside the "
        "deepseek-v4-kernel project directory."
    ) from exc


def sparse_decode_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    topk_length: Optional[torch.Tensor],
    attn_sink: Optional[torch.Tensor],
    tile_scheduler_metadata: Optional[torch.Tensor],
    num_splits: Optional[torch.Tensor],
    extra_kv: Optional[torch.Tensor],
    extra_indices: Optional[torch.Tensor],
    extra_topk_length: Optional[torch.Tensor],
    d_v: int,
    sm_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """See `flash_mla.cuda.sparse_decode_fwd` for semantics."""
    return _cuda.sparse_decode_fwd(
        q,
        kv,
        indices,
        topk_length,
        attn_sink,
        tile_scheduler_metadata,
        num_splits,
        extra_kv,
        extra_indices,
        extra_topk_length,
        int(d_v),
        float(sm_scale),
    )
