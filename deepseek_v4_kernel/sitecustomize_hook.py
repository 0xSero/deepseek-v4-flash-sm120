"""Drop-in ``sitecustomize.py`` body.

Place this file as ``sitecustomize.py`` on ``PYTHONPATH`` (or inside any
``site-packages``) and every subsequent Python interpreter start will apply
the flash_mla patch automatically — crucial for sglang, which forks worker
processes that would otherwise miss the monkey-patch.

Exceptions are deliberately swallowed so a broken kernel install never
prevents the interpreter from starting.
"""
import os
import sys

if os.environ.get("DSV4_KERNEL_DISABLE", "0") not in ("1", "true", "yes"):
    try:
        import deepseek_v4_kernel  # type: ignore
        deepseek_v4_kernel.patch_flash_mla()
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(
            f"[deepseek_v4_kernel/sitecustomize] patch skipped: {exc}\n"
        )
