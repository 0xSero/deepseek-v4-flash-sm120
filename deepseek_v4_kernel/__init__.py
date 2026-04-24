"""DeepSeek-V4-Flash SM_120 sparse decode kernels."""
from __future__ import annotations

from .ops import sparse_decode_fwd  # noqa: F401
from ._patch import install as patch_flash_mla  # noqa: F401

__all__ = ["sparse_decode_fwd", "patch_flash_mla"]
