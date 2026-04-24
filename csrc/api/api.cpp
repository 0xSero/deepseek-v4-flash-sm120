// pybind11 entry point for the SM_120 DeepSeek-V4 sparse decode kernel.
// Exposes the same module name (`deepseek_v4_kernel.cuda`) and symbol
// (`sparse_decode_fwd`) that `flash_mla.cuda` publishes so that we can
// monkey-patch it at runtime.

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "api/sparse_decode.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "DeepSeek-V4-Flash sparse decode kernels (SM_120).";

    m.def("sparse_decode_fwd", &dsv4_kernel::sparse_decode_fwd,
          pybind11::arg("q"),
          pybind11::arg("kv"),
          pybind11::arg("indices"),
          pybind11::arg("topk_length"),
          pybind11::arg("attn_sink"),
          pybind11::arg("tile_scheduler_metadata"),
          pybind11::arg("num_splits"),
          pybind11::arg("extra_kv"),
          pybind11::arg("extra_indices"),
          pybind11::arg("extra_topk_length"),
          pybind11::arg("d_v"),
          pybind11::arg("sm_scale"),
          "Sparse MLA decode forward (SM_120 / Blackwell workstation).");
}
