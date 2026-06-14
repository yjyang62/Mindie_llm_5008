#include <torch/library.h>

#include <iostream>

#include "ops_common.h"

namespace mie_ops {
using namespace at_npu::native;

at::Tensor npu_apply_top_k_top_p_custom_npu(const at::Tensor& logits, const c10::optional<at::Tensor>& p,
                                            const c10::optional<at::Tensor>& k, const at::Tensor& out) {
    EXEC_NPU_CMD_V1(aclnnApplyTopKTopPCustom, logits, p, k, out);
    return out;
}

at::Tensor npu_apply_top_k_top_p_custom_meta(const at::Tensor& logits, const c10::optional<at::Tensor>& p,
                                             const c10::optional<at::Tensor>& k, const at::Tensor& out) {
    return out;
}

}  // namespace mie_ops

TORCH_LIBRARY_IMPL(mie_ops, PrivateUse1, m) {
    m.impl("apply_top_k_top_p_custom", &mie_ops::npu_apply_top_k_top_p_custom_npu);
}

TORCH_LIBRARY_IMPL(mie_ops, Meta, m) {
    m.impl("apply_top_k_top_p_custom", &mie_ops::npu_apply_top_k_top_p_custom_meta);
}
