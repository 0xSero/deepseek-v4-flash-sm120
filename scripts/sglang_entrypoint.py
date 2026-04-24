"""sglang entry-point shim.

Run inside the sglang Docker container after our kernel is mounted at
``/dsv4`` (read-only).  Applies the flash_mla patch **before** sglang forks
its worker processes, then forwards the remaining argv to sglang's
``launch_server`` entrypoint.

No files are written into the container image; everything stays in-memory.
"""
from __future__ import annotations

import os
import runpy
import sys


def main() -> None:
    # Make sure our package is importable even if PYTHONPATH isn't forwarded.
    extra_path = "/dsv4"
    if os.path.isdir(extra_path) and extra_path not in sys.path:
        sys.path.insert(0, extra_path)

    try:
        import deepseek_v4_kernel  # type: ignore

        deepseek_v4_kernel.patch_flash_mla()
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(
            f"[deepseek_v4_kernel] patch skipped: {exc}\n"
        )

    # Hand off to sglang.launch_server with the remaining CLI args.
    sys.argv[0] = "sglang.launch_server"
    runpy.run_module("sglang.launch_server", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
