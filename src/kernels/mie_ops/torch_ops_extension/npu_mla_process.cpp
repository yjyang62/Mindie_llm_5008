
#include "ops_common.h"
namespace mie_ops {
using npu_preparation = at_npu::native::OpPreparation;

std::unordered_map<c10::string_view, int64_t> cache_mode_map = {{"krope_ctkv", 1}, {"int8_nzcache", 2}, {"nzcache", 3}};

std::unordered_map<c10::string_view, int64_t> quant_mode_map = {
    {"per_tensor_quant_asymm", 0},
    {"per_token_quant_symm", 1},
};

template <typename MapType>
inline int64_t get_op_mode(const MapType &mode_map, c10::optional<c10::string_view> mode_opt,
                           c10::string_view default_mode, const char *mode_name)
{
    c10::string_view mode_str = mode_opt.value_or(default_mode);
    auto it = mode_map.find(mode_str);
    TORCH_CHECK(it != mode_map.end(), "Unsupported ", mode_name, " value: '", mode_str, "'");
    return it->second;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
npu_mla_process_npu(const at::Tensor &input, const at::Tensor &gamma0, const at::Tensor &beta0, const at::Tensor &wdqkv,
                    const at::Tensor &descale0, const at::Tensor &gamma1, const at::Tensor &beta1,
                    const at::Tensor &wuq, const at::Tensor &descale1, const at::Tensor &gamma2, const at::Tensor &cos,
                    const at::Tensor &sin, const at::Tensor &wuk, const at::Tensor &kv_cache,
                    const at::Tensor &kv_cache_rope, const at::Tensor &slotmapping,
                    const c10::optional<at::Tensor> &quant_scale0, const c10::optional<at::Tensor> &quant_offset0,
                    const c10::optional<at::Tensor> &bias0, const c10::optional<at::Tensor> &quant_scale1,
                    const c10::optional<at::Tensor> &quant_offset1, const c10::optional<at::Tensor> &bias1,
                    const c10::optional<at::Tensor> &ctkv_scale, const c10::optional<at::Tensor> &q_nope_scale,
                    c10::optional<c10::string_view> cache_mode_opt, c10::optional<c10::string_view> quant_mode_opt)
{
    // construct the output tensor
    const c10::OptionalDeviceGuard device_guard(device_of(input));
    int64_t wdq_dim = 0;
    int64_t q_rope_dim = 0;
    int64_t k_rope_dim = 0;
    double epsilon = 1e-5;
    int64_t q_rotary_coeff = 2;
    int64_t k_rotary_coeff = 2;
    bool transpose_wdq = true;
    bool transpose_wuq = true;
    bool transpose_wuk = true;
    int64_t token_num = input.size(0);
    int64_t head_num = wuk.size(0);
    at::Tensor q_out0 = at::empty({token_num, head_num, 512}, kv_cache.options());
    at::Tensor kv_cache_out0;
    at::Tensor q_out1 = at::empty({token_num, head_num, 64}, input.options());
    at::Tensor kv_cache_out1;
    auto cache_mode = get_op_mode(cache_mode_map, cache_mode_opt, "krope_ctkv", "cache_mode");
    auto quant_mode = get_op_mode(quant_mode_map, quant_mode_opt, "per_token_quant_symm", "quant_mode");
    bool do_rms_norm = true;
    int64_t wdkv_split_count = 1;
    EXEC_NPU_CMD_v0(aclnnMlaPreprocess, input, gamma0, beta0, quant_scale0, quant_offset0, wdqkv, descale0, bias0,
                    gamma1, beta1, quant_scale1, quant_offset1, wuq, descale1, bias1, gamma2, cos, sin, wuk, kv_cache,
                    kv_cache_rope, slotmapping, ctkv_scale, q_nope_scale, wdq_dim, q_rope_dim, k_rope_dim, epsilon,
                    q_rotary_coeff, k_rotary_coeff, transpose_wdq, transpose_wuq, transpose_wuk, cache_mode, quant_mode,
                    do_rms_norm, wdkv_split_count, q_out0, kv_cache, q_out1, kv_cache_rope);
    return std::make_tuple(q_out0, kv_cache, q_out1, kv_cache_rope);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
npu_mla_process_meta(const at::Tensor &input, const at::Tensor &gamma0, const at::Tensor &beta0,
                     const at::Tensor &wdqkv, const at::Tensor &descale0, const at::Tensor &gamma1,
                     const at::Tensor &beta1, const at::Tensor &wuq, const at::Tensor &descale1,
                     const at::Tensor &gamma2, const at::Tensor &cos, const at::Tensor &sin, const at::Tensor &wuk,
                     const at::Tensor &kv_cache, const at::Tensor &kv_cache_rope, const at::Tensor &slotmapping,
                     const c10::optional<at::Tensor> &quant_scale0, const c10::optional<at::Tensor> &quant_offset0,
                     const c10::optional<at::Tensor> &bias0, const c10::optional<at::Tensor> &quant_scale1,
                     const c10::optional<at::Tensor> &quant_offset1, const c10::optional<at::Tensor> &bias1,
                     const c10::optional<at::Tensor> &ctkv_scale, const c10::optional<at::Tensor> &q_nope_scale,
                     c10::optional<c10::string_view> cache_mode_opt, c10::optional<c10::string_view> quant_mode_opt)
{
    // construct the output tensor
    const c10::OptionalDeviceGuard device_guard(device_of(input));
    int64_t wdq_dim = 0;
    int64_t q_rope_dim = 0;
    int64_t k_rope_dim = 0;
    double epsilon = 1e-5;
    int64_t q_rotary_coeff = 2;
    int64_t k_rotary_coeff = 2;
    bool transpose_wdq = true;
    bool transpose_wuq = true;
    bool transpose_wuk = true;
    int64_t token_num = input.size(0);
    int64_t head_num = wuk.size(0);
    at::Tensor q_out0 = at::empty({token_num, head_num, 512}, kv_cache.options());
    at::Tensor kv_cache_out0;
    at::Tensor q_out1 = at::empty({token_num, head_num, 64}, input.options());
    at::Tensor kv_cache_out1;

    return std::make_tuple(q_out0, kv_cache, q_out1, kv_cache_rope);
}
} // namespace mie_ops

// step4, 为NPU设备注册前向实现
TORCH_LIBRARY_IMPL(mie_ops, PrivateUse1, m) { m.impl("npu_mla_process", &mie_ops::npu_mla_process_npu); }

// step5, 为META设备注册前向实现
TORCH_LIBRARY_IMPL(mie_ops, Meta, m) { m.impl("npu_mla_process", &mie_ops::npu_mla_process_meta); }