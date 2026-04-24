"""Runtime patch that replaces flash_mla's sparse decode path with our SM_120
kernel for DeepSeek-V4-Flash.

Two public entry points are swapped out so all the call-sites used by sglang
and vllm land in our code:

  1. ``flash_mla.flash_mla_with_kvcache`` (upstream pip wheel; sglang path)
  2. ``vllm.third_party.flashmla.flash_mla_interface.flash_mla_with_kvcache``

On non-SM_120 GPUs :func:`install` is a no-op — the wrapper falls through to
the original function.

Usage:
    import deepseek_v4_kernel
    deepseek_v4_kernel.patch_flash_mla()

For sglang, setting ``PYTHONSTARTUP`` (or installing a ``sitecustomize.py``
hook) ensures every forked worker runs the patch before the first model
forward pass.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable, Optional, Tuple

import torch

from .ops import sparse_decode_fwd as _dsv4_sparse_decode_fwd

_log = logging.getLogger("deepseek_v4_kernel.patch_flash_mla")
_INSTALLED: bool = False


def _current_is_sm120() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(0)
    return major == 12


def _make_wrapper(original: Callable, schedmeta_cls: Any) -> Callable:
    """Return a drop-in replacement for ``flash_mla_with_kvcache``."""

    def flash_mla_with_kvcache(
        q: torch.Tensor,
        k_cache: torch.Tensor,
        block_table: Optional[torch.Tensor],
        cache_seqlens: Optional[torch.Tensor],
        head_dim_v: int,
        tile_scheduler_metadata: Any,
        num_splits: Any = None,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        is_fp8_kvcache: bool = False,
        indices: Optional[torch.Tensor] = None,
        attn_sink: Optional[torch.Tensor] = None,
        extra_k_cache: Optional[torch.Tensor] = None,
        extra_indices_in_kvcache: Optional[torch.Tensor] = None,
        topk_length: Optional[torch.Tensor] = None,
        extra_topk_length: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Only take over the sparse FP8 decode path on SM_120; everything
        # else defers to the upstream implementation.
        is_sparse = indices is not None
        _decision = (
            is_sparse,
            _current_is_sm120(),
            bool(is_fp8_kvcache),
            int(q.element_size()) if isinstance(q, torch.Tensor) else None,
        )
        if os.environ.get("DSV4_KERNEL_TRACE", "0") == "1":
            _log.warning("dsv4 gating: %s q=%s kcache=%s",
                         _decision,
                         tuple(q.shape) if isinstance(q, torch.Tensor) else q,
                         tuple(k_cache.shape) if isinstance(k_cache, torch.Tensor) else k_cache)
        if all(_decision[:3]) and _decision[3] == 2:
            try:
                if softmax_scale is None:
                    softmax_scale = q.shape[-1] ** (-0.5)
                # Accept schedmeta object or plain tensor tuple.
                sched_meta_tensor = None
                num_splits_tensor = None
                if schedmeta_cls is not None and isinstance(
                    tile_scheduler_metadata, schedmeta_cls
                ):
                    sched_meta_tensor = tile_scheduler_metadata.tile_scheduler_metadata
                    num_splits_tensor = tile_scheduler_metadata.num_splits
                elif isinstance(tile_scheduler_metadata, torch.Tensor):
                    sched_meta_tensor = tile_scheduler_metadata
                    num_splits_tensor = num_splits
                out, lse, new_meta, new_splits = _dsv4_sparse_decode_fwd(
                    q,
                    k_cache,
                    indices,
                    topk_length,
                    attn_sink,
                    sched_meta_tensor,
                    num_splits_tensor,
                    extra_k_cache,
                    extra_indices_in_kvcache,
                    extra_topk_length,
                    head_dim_v,
                    float(softmax_scale),
                )
                if schedmeta_cls is not None and isinstance(
                    tile_scheduler_metadata, schedmeta_cls
                ):
                    tile_scheduler_metadata.tile_scheduler_metadata = new_meta
                    tile_scheduler_metadata.num_splits = new_splits
                    if not tile_scheduler_metadata.have_initialized:
                        # Populate the config record so subsequent calls hit
                        # the fast assertion path in the original interface.
                        tile_scheduler_metadata.have_initialized = True
                        tile_scheduler_metadata.config = schedmeta_cls.Config(
                            q.shape[0],
                            q.shape[1],
                            q.shape[2],
                            k_cache.shape[1],
                            k_cache.shape[2],
                            causal,
                            is_fp8_kvcache,
                            indices.shape[-1],
                            extra_k_cache.shape[1] if extra_k_cache is not None else None,
                            extra_indices_in_kvcache.shape[-1] if extra_indices_in_kvcache is not None else None,
                        )
                return out, lse
            except Exception as exc:  # pragma: no cover
                if os.environ.get("DSV4_KERNEL_STRICT", "0") == "1":
                    raise
                _log.warning(
                    "deepseek_v4_kernel sparse decode failed (%s); "
                    "falling back to upstream flash_mla.",
                    exc,
                )

        # Fallback: original flash_mla code path.
        return original(
            q=q,
            k_cache=k_cache,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            head_dim_v=head_dim_v,
            tile_scheduler_metadata=tile_scheduler_metadata,
            num_splits=num_splits,
            softmax_scale=softmax_scale,
            causal=causal,
            is_fp8_kvcache=is_fp8_kvcache,
            indices=indices,
            attn_sink=attn_sink,
            extra_k_cache=extra_k_cache,
            extra_indices_in_kvcache=extra_indices_in_kvcache,
            topk_length=topk_length,
            extra_topk_length=extra_topk_length,
        )

    flash_mla_with_kvcache.__wrapped__ = original  # type: ignore[attr-defined]
    return flash_mla_with_kvcache


def _patch_flash_mla_pkg() -> bool:
    try:
        import flash_mla  # type: ignore
        from flash_mla import flash_mla_interface as _fmi  # type: ignore
    except ImportError:
        return False
    original = getattr(_fmi, "flash_mla_with_kvcache", None)
    if original is None:
        return False
    schedmeta_cls = getattr(_fmi, "FlashMLASchedMeta", None)
    wrapped = _make_wrapper(original, schedmeta_cls)
    _fmi.flash_mla_with_kvcache = wrapped
    # Top-level re-export used by sglang's adapter.
    flash_mla.flash_mla_with_kvcache = wrapped  # type: ignore[attr-defined]
    return True


def _patch_vllm_pkg() -> bool:
    try:
        from vllm.third_party.flashmla import (  # type: ignore
            flash_mla_interface as _vfmi,
        )
    except ImportError:
        return False
    original = getattr(_vfmi, "flash_mla_with_kvcache", None)
    if original is None:
        return False
    schedmeta_cls = getattr(_vfmi, "FlashMLASchedMeta", None)
    wrapped = _make_wrapper(original, schedmeta_cls)
    _vfmi.flash_mla_with_kvcache = wrapped
    try:
        from vllm.third_party import flashmla as _vflash  # type: ignore
        if hasattr(_vflash, "flash_mla_with_kvcache"):
            _vflash.flash_mla_with_kvcache = wrapped  # type: ignore[attr-defined]
    except ImportError:
        pass
    return True


def _patch_sglang_indexer_fallbacks() -> None:
    """Workaround for sglang's DSv4 C4-indexer fallbacks that assert a 1-D
    ``seq_lens`` even though the caller (``forward_c4_indexer``) unconditionally
    unsqueezes it to 2-D to match the deep_gemm (SM_100-only) signature.  Happens
    the first time EAGLE / any multi-token-per-request batch is dispatched on
    SM_120; squeezing the trailing 1-dim is a safe no-op for all paths."""
    try:
        from sglang.srt.layers.attention.nsa import tilelang_kernel as _tk  # type: ignore
    except ImportError:
        _tk = None
    try:
        from sglang.srt.layers.attention.compressed import indexer as _idx  # type: ignore
    except ImportError:
        _idx = None

    def _wrap(fn_owner, fn_name):
        if fn_owner is None:
            return
        original = getattr(fn_owner, fn_name, None)
        if original is None or getattr(original, "__dsv4_patched__", False):
            return

        def wrapper(q_fp8, kvcache_fp8, weight, seq_lens, *args, **kwargs):
            if (
                isinstance(seq_lens, torch.Tensor)
                and seq_lens.dim() == 2
                and seq_lens.shape[-1] == 1
            ):
                seq_lens = seq_lens.squeeze(-1).contiguous()
            return original(q_fp8, kvcache_fp8, weight, seq_lens, *args, **kwargs)

        wrapper.__wrapped__ = original  # type: ignore[attr-defined]
        wrapper.__dsv4_patched__ = True  # type: ignore[attr-defined]
        setattr(fn_owner, fn_name, wrapper)

    _wrap(_tk, "tilelang_fp8_paged_mqa_logits")
    _wrap(_idx, "fp8_paged_mqa_logits_torch")


def install() -> None:
    """Install the patch. Idempotent."""
    global _INSTALLED
    if _INSTALLED:
        return
    patched_any = False
    if _patch_flash_mla_pkg():
        patched_any = True
    if _patch_vllm_pkg():
        patched_any = True
    _patch_sglang_indexer_fallbacks()
    if not patched_any:
        _log.warning(
            "deepseek_v4_kernel: neither flash_mla nor vllm.third_party.flashmla "
            "was importable; nothing patched."
        )
        return
    _INSTALLED = True
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
    else:
        major, minor = (-1, -1)
    _log.info(
        "deepseek_v4_kernel.patch_flash_mla installed (device SM %d.%d).",
        major,
        minor,
    )
