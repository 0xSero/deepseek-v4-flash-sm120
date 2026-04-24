"""Build script for deepseek_v4_kernel.cuda (SM_120a / Blackwell workstation).

The generated extension exposes the same `sparse_decode_fwd` signature as
`flash_mla.cuda` so that it can be plugged in via `patch_flash_mla.install()`.
"""

from pathlib import Path
import os
import subprocess
from datetime import datetime

from setuptools import find_packages, setup

from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
    CUDA_HOME,
    IS_WINDOWS,
)


def _flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in ("1", "true", "yes", "y")


def _arch_flags():
    assert CUDA_HOME is not None, "CUDA_HOME not set"
    nvcc_version = subprocess.check_output(
        [os.path.join(CUDA_HOME, "bin", "nvcc"), "--version"],
        stderr=subprocess.STDOUT,
    ).decode("utf-8")
    release = nvcc_version.split("release ")[1].split(",")[0].strip()
    major, minor = [int(x) for x in release.split(".")[:2]]
    print(f"[deepseek_v4_kernel] NVCC {major}.{minor}")
    # SM_120a is added in CUDA 12.8; we default to sm_120 for >=12.8.
    flags = []
    if (major, minor) >= (12, 8):
        flags += ["-gencode", "arch=compute_120a,code=sm_120a"]
    else:
        # Fall back to sm_90a so the code still compiles on older toolchains
        # (though it will not run on Blackwell workstation without 12.8+).
        flags += ["-gencode", "arch=compute_90a,code=sm_90a"]
    # Allow user override.
    override = os.getenv("DSV4_KERNEL_ARCH")
    if override:
        flags = []
        for token in override.split(";"):
            token = token.strip()
            if not token:
                continue
            flags += ["-gencode", token]
    return flags


def _nvcc_threads():
    return ["--threads", os.getenv("NVCC_THREADS", "16")]


this_dir = Path(__file__).resolve().parent
csrc = Path("csrc")
csrc_abs = this_dir / "csrc"

cxx_args = (
    ["/O2", "/std:c++20", "/DNDEBUG", "/W0"]
    if IS_WINDOWS
    else ["-O3", "-std=c++20", "-DNDEBUG", "-Wno-deprecated-declarations"]
)

nvcc_args = [
    "-O3",
    "-std=c++20",
    "-DNDEBUG",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--use_fast_math",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-lineinfo",
]
if _flag("DSV4_KERNEL_DEBUG"):
    nvcc_args += ["-G"]

ext_modules = [
    CUDAExtension(
        name="deepseek_v4_kernel.cuda",
        sources=[
            str(csrc / "api" / "api.cpp"),
            str(csrc / "api" / "sparse_decode.cpp"),
            str(csrc / "sm120" / "decode" / "sparse_decode_instantiation.cu"),
        ],
        extra_compile_args={
            "cxx": cxx_args,
            "nvcc": nvcc_args + _arch_flags() + _nvcc_threads(),
        },
        # Headers live under csrc/.  If CUTLASS_DIR is set, its `include/`
        # is added before our shim so the full CUTLASS headers win.
        include_dirs=[
            str(csrc_abs),
            *(
                [os.path.join(os.environ["CUTLASS_DIR"], "include")]
                if os.environ.get("CUTLASS_DIR")
                else []
            ),
        ],
    )
]


try:
    rev = "+" + subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"]
    ).decode("ascii").rstrip()
except Exception:
    rev = "+" + datetime.now().strftime("%Y%m%d%H%M%S")


setup(
    name="deepseek_v4_kernel",
    version="0.1.0" + rev,
    description="SM_120 sparse decode kernels for DeepSeek-V4-Flash",
    packages=find_packages(include=["deepseek_v4_kernel", "deepseek_v4_kernel.*"]),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
    install_requires=["torch>=2.5"],
)
