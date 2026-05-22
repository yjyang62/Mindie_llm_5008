# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import torch
import torch_npu

from ..quantize.pack_type import PackType, LinearType, ALL_PACK_LIST, TransposeType
from ..quantize.quant_type import QuantType, LinearTypeV2
from ..log import logger, print_log


def get_module(obj, name, delimiter='.'):
    names = name.split(delimiter)
    for name in names:
        obj = getattr(obj, name)
    return obj


class NormWrapper:
    def __init__(self, norm_name):
        self.norm_name = norm_name


class AttnWrapper(NormWrapper):
    def __init__(self, norm_name, wrapper_name, pack_name=None, sep_names=None, o_name=None):
        super().__init__(norm_name)
        self.wrapper_name = wrapper_name
        self.pack_name = pack_name if pack_name else None
        self.q_name = f'{sep_names[0]}' if sep_names and len(sep_names) == 3 else None
        self.k_name = f'{sep_names[1]}' if sep_names and len(sep_names) == 3 else None
        self.v_name = f'{sep_names[2]}' if sep_names and len(sep_names) == 3 else None
        self.o_name = o_name


class MlpWrapper(NormWrapper):
    def __init__(self, norm_name, wrapper_name, pack_name=None, sep_names=None, down_name=None):
        super().__init__(norm_name)
        self.wrapper_name = wrapper_name
        self.pack_name = f'{pack_name}' if pack_name else None
        self.gate_name = f'{sep_names[0]}' if sep_names and len(sep_names) == 2 else None
        self.up_name = f'{sep_names[1]}' if sep_names and len(sep_names) == 2 else None
        self.down_name = down_name


class WeightWrapper:
    def __init__(self, soc_info, tp_rank, attn_wrapper, mlp_wrapper, **kwargs):
        self.soc_info = soc_info
        self.tp_rank = tp_rank
        self.attn_wrapper = attn_wrapper
        self.mlp_wrapper = mlp_wrapper
        self.placeholder = torch.zeros(1, dtype=torch.float16, device="npu")
        self.weights = []
        self.linear_type = []
        self.linear_transpose_types = []
        self.pack_quant_type = []
        self.layer_linear_type = []
        self.layer_linear_transpose_types = []
        self.linear_descs = []
        self.layer_linear_descs = []
        self.is_anti_outlier = []
        self.layer_is_anti_outlier = []
        self.enable_rope_quant_kvcache = kwargs.get("enable_rope_quant_kvcache", False)
        self.enable_swiglu_quant = kwargs.get('enable_swiglu_quant', False)

    def weight_format_cast(self, tensor, enable_nz=False, nd_weight=False):
        need_nz = enable_nz or self.soc_info.need_nz or self.soc_info.matmul_nd_nz
        if need_nz and not nd_weight:
            torch.npu.config.allow_internal_format = True
            torch_npu.npu_format_cast_(tensor, 29)
            print_log(self.tp_rank, logger.debug, f"Tensor trans to {torch_npu.get_npu_format(tensor)}")
        return tensor

    def register_embedding(self, embed_tokens):
        self.weights.append(embed_tokens.weight.data)

    def register_linear_bias(self, linear, enable_nz=False):
        linear.weight.data = self.weight_format_cast(
            linear.weight.data, enable_nz=enable_nz, nd_weight=linear.nd_weight)
        self.weights.append(linear.weight.data)
        if hasattr(linear, 'bias') and (getattr(linear, 'bias') is not None):
            self.weights.append(linear.bias.data)
        else:
            self.weights.append(self.placeholder)

    def register_norm(self, norm):
        self.weights.append(norm.weight.data)
        if hasattr(norm, 'bias') and (getattr(norm, 'bias') is not None):
            self.weights.append(norm.bias.data)
        else:
            self.weights.append(self.placeholder)

    def register_norm_wrapper(self, norm, norm_anti):
        self.register_norm(norm)
        self.register_norm(norm_anti)

    def register_layer_norm(self, layer, wrapper, pack_type):
        if pack_type in [
            PackType.ALL_W8A8_ANTI, PackType.MIX_W8A8_ANTI,
            PackType.ALL_W8A8SC_ANTI, PackType.MIX_W8A8SC_ANTI,
            PackType.ALL_W8A16_ANTI, PackType.MIX_W8A16_ANTI,
            PackType.ALL_W4A16_ANTI, PackType.MIX_W4A16_ANTI,
            PackType.ALL_W8A8_DYNAMIC_ANTI, PackType.MIX_W8A8_DYNAMIC_ANTI
        ]:
            norm = get_module(layer, f'{wrapper.norm_name}.ori')
            norm_anti = get_module(layer, f'{wrapper.norm_name}.anti')
            self.register_norm_wrapper(norm, norm_anti)
            self.layer_is_anti_outlier.append(True)
        else:
            norm = get_module(layer, wrapper.norm_name)
            self.register_norm(norm)
            self.layer_is_anti_outlier.append(False)
            self.weights.extend([self.placeholder] * 2)

    def register_linear_wrapper(self, linear, quantize_type, is_down=False, is_lcoc=False):
        trans_flag = linear.trans_flag
        if linear.weight.dtype in [torch.float16, torch.bfloat16]:
            self.register_linear_bias(linear, enable_nz=False)
            self.weights.extend([self.placeholder] * 4)
            self.layer_linear_type.append(LinearType.FP)
        elif quantize_type in [QuantType.W4A16, QuantType.W8A16]:
            self.register_linear_bias(linear, enable_nz=False)
            self.weights.append(self.placeholder)
            self.weights.append(linear.weight_offset.data)
            self.weights.append(linear.weight_scale.data)
            self.weights.append(self.placeholder)
            self.layer_linear_type.append(LinearType.INT)
        elif quantize_type in [QuantType.W8A8_DYNAMIC]:
            self.register_linear_bias(linear, enable_nz=False)
            self.weights.append(self.placeholder)
            self.register_linear_offset_scale(is_down, linear.weight_offset, linear.weight_scale, is_lcoc=is_lcoc)
            self.weights.append(self.placeholder)
            self.layer_linear_type.append(LinearType.INT)
        elif quantize_type in [QuantType.W4A8_DYNAMIC]:
            self.register_linear_bias(linear, enable_nz=False)
            self.weights.append(self.placeholder)
            self.weights.append(self.placeholder)
            self.weights.append(linear.weight_scale.data)
            self.weights.append(self.placeholder)
            self.layer_linear_type.append(LinearType.INT)
        elif quantize_type == QuantType.W16A16SC:
            linear.weight = linear.weight.view(torch.float16)
            linear.quant_bias = linear.quant_bias.to(torch.float32)
            self.weights.append(linear.weight.data)
            self.weights.append(linear.quant_bias.data)
            self.weights.extend([self.placeholder] * 3)
            self.weights.append(linear.index.data)
            self.layer_linear_type.append(LinearType.FP)
        else:
            linear.weight.data = self.weight_format_cast(linear.weight.data)
            self.weights.append(linear.weight.data)
            self.weights.append(linear.quant_bias.data)
            self.weights.append(linear.deq_scale.data)
            self.register_linear_offset_scale(is_down, linear.input_offset, linear.input_scale, is_lcoc=is_lcoc)
            if quantize_type == QuantType.W8A8SC:
                self.weights.append(linear.index.data)
            else:
                self.weights.append(self.placeholder)
            self.layer_linear_type.append(LinearType.INT)
        self.layer_linear_descs.append(linear.linear_desc)
        self.layer_linear_transpose_types.append(trans_flag)

    def register_linear_offset_scale(self, is_down, offset, scale, is_lcoc=False):
        """
        mlp的down模块权重在swiglu开启下需设置为float32; 其他场景或module保持原权重数据类型不变
        """
        if is_lcoc:
            dequant_scale = scale.to(torch.float32).view(torch.int32).to(torch.int64)
            self.weights.append(offset.data.to(torch.float32))
            self.weights.append(dequant_scale)
        elif is_down and self.enable_swiglu_quant:
            self.weights.append(offset.data.to(torch.float32))
            self.weights.append(scale.data.to(torch.float32))
        else:
            self.weights.append(offset.data)
            self.weights.append(scale.data)

    def register_layer_linear_pack(self, layer, wrapper, quantize_type, linear_type='attn'):
        wrapper_module = get_module(layer, wrapper.wrapper_name)
        pack_type = wrapper_module.pack_type
        self.register_layer_norm(layer, wrapper, pack_type)
        linear = get_module(wrapper_module, wrapper.pack_name).linear
        self.register_linear_wrapper(linear, quantize_type)
        if linear_type == 'attn':
            self.weights.extend([self.placeholder] * 12)
            self.layer_linear_descs.extend([LinearTypeV2.INVALID, LinearTypeV2.INVALID])
            self.layer_linear_type.extend([LinearType.INVALID, LinearType.INVALID])
            self.layer_linear_transpose_types.extend([LinearType.INVALID, LinearType.INVALID])
        elif linear_type == 'mlp':
            self.weights.extend([self.placeholder] * 6)
            self.layer_linear_descs.extend([LinearTypeV2.INVALID])
            self.layer_linear_type.extend([LinearType.INVALID])
            self.layer_linear_transpose_types.extend([LinearType.INVALID])
        else:
            raise AssertionError(f'{linear_type} not yet implemented in register_layer_linear_pack')

    def register_layer_linear_sep(self, layer, wrapper, quantize_type, linear_type='attn'):
        wrapper_module = get_module(layer, wrapper.wrapper_name)
        pack_type = wrapper_module.pack_type
        self.register_layer_norm(layer, wrapper, pack_type)
        if linear_type == 'attn':
            q_linear = get_module(wrapper_module, wrapper.q_name).linear
            k_linear = get_module(wrapper_module, wrapper.k_name).linear
            v_linear = get_module(wrapper_module, wrapper.v_name).linear
            self.register_linear_wrapper(q_linear, quantize_type)
            self.register_linear_wrapper(k_linear, quantize_type)
            self.register_linear_wrapper(v_linear, quantize_type)
        elif linear_type == 'mlp':
            gate_linear = get_module(wrapper_module, wrapper.gate_name).linear
            up_linear = get_module(wrapper_module, wrapper.up_name).linear
            self.register_linear_wrapper(gate_linear, quantize_type)
            self.register_linear_wrapper(up_linear, quantize_type)
        else:
            raise AssertionError(f'{linear_type} not yet implemented in register_layer_linear_sep')

    def register_layer_attn(self, layer, wrapper, quantize_type):
        wrapper_module = get_module(layer, wrapper.wrapper_name)
        pack_type = wrapper_module.pack_type
        if pack_type in ALL_PACK_LIST:
            self.register_layer_linear_pack(layer, wrapper, quantize_type, 'attn')
        else:
            self.register_layer_linear_sep(layer, wrapper, quantize_type, 'attn')
        o_linear = get_module(wrapper_module, wrapper.o_name).linear
        self.register_linear_wrapper(o_linear, quantize_type)

    def register_layer_mlp(self, layer, wrapper, quantize_type):
        wrapper_module = get_module(layer, wrapper.wrapper_name)
        pack_type = wrapper_module.pack_type
        if pack_type in ALL_PACK_LIST:
            self.register_layer_linear_pack(layer, wrapper, quantize_type, 'mlp')
        else:
            self.register_layer_linear_sep(layer, wrapper, quantize_type, 'mlp')
        down_linear = get_module(wrapper_module, wrapper.down_name).linear
        self.register_linear_wrapper(down_linear, quantize_type, is_down=True)

    def register_layer_kvquant(self, layer):
        attn_attrs = ['self_attn', 'attn', 'self_attention', 'attention']
        attn_attr = None
        for name in attn_attrs:
            if hasattr(layer, name):
                attn_attr = getattr(layer, name)
                break
        if attn_attr is None:
            raise AssertionError('run register_layer_kvquant failed')
        if self.enable_rope_quant_kvcache:
            self.weights.append(attn_attr.kv_cache_quant.k_quant_scale.data.to(torch.float32))
            self.weights.append(attn_attr.kv_cache_quant.k_dequant_scale.data)
            self.weights.append(attn_attr.kv_cache_quant.v_quant_scale.data.to(torch.float32))
            self.weights.append(attn_attr.kv_cache_quant.v_dequant_scale.data)
            self.weights.append(attn_attr.kv_cache_quant.k_quant_offset.data.to(torch.float32))
            self.weights.append(attn_attr.kv_cache_quant.k_dequant_offset.data)
            self.weights.append(attn_attr.kv_cache_quant.v_quant_offset.data.to(torch.float32))
            self.weights.append(attn_attr.kv_cache_quant.v_dequant_offset.data)
        else:
            self.weights.append(attn_attr.kv_cache_quant.k_quant_scale.data)
            self.weights.append(attn_attr.kv_cache_quant.k_dequant_scale.data)
            self.weights.append(attn_attr.kv_cache_quant.v_quant_scale.data)
            self.weights.append(attn_attr.kv_cache_quant.v_dequant_scale.data)
            self.weights.append(attn_attr.kv_cache_quant.k_quant_offset.data)
            self.weights.append(attn_attr.kv_cache_quant.k_dequant_offset.data)
            self.weights.append(attn_attr.kv_cache_quant.v_quant_offset.data)
            self.weights.append(attn_attr.kv_cache_quant.v_dequant_offset.data)
    
    def append_weight_if_exists(self, weight):
        if weight is not None:
            self.weights.append(weight.data)

    def register_layer_qkvquant(self, layer):
        attn_attrs = ['self_attn', 'attn', 'self_attention']
        attn_attr = None
        for name in attn_attrs:
            if hasattr(layer, name):
                attn_attr = getattr(layer, name)
                break
        if attn_attr is None:
            raise AssertionError('run register_layer_qkvquant failed')
        self.append_weight_if_exists(attn_attr.fa3.q_scale)
        self.append_weight_if_exists(attn_attr.fa3.k_scale)
        self.append_weight_if_exists(attn_attr.fa3.v_scale)
        self.append_weight_if_exists(attn_attr.fa3.qk_scale)
        self.append_weight_if_exists(attn_attr.fa3.q_offset)
        self.append_weight_if_exists(attn_attr.fa3.kv_offset)
        self.append_weight_if_exists(attn_attr.fa3.fa3_v_scale)
        self.append_weight_if_exists(attn_attr.fa3.fa3_offset)

    def register_layer_reducequant(self, layer):
        reduce_attrs = ['self_attn', 'mlp']
        for name in reduce_attrs:
            reduce_attr = getattr(layer, name)
            self.weights.append(reduce_attr.reduce_quant.reduce_quant_scale.data)
            self.weights.append(torch.zeros_like(reduce_attr.reduce_quant.reduce_quant_scale.data, dtype=torch.int8))
            self.weights.append(reduce_attr.reduce_quant.gather_quant_scale.data)
            self.weights.append(self.placeholder)

    def register_model_norm(self, norm):
        # self.register_norm(norm) #后面统一用这个, model参数需要+1
        self.weights.append(norm.weight.data)
        if hasattr(norm, 'bias') and (getattr(norm, 'bias') is not None):
            self.weights.append(norm.bias.data)

    def register_model_lmhead(self, lm_head):
        lm_head.linear.weight.data = self.weight_format_cast(lm_head.linear.weight.data)
        self.weights.append(lm_head.linear.weight.data)

    def register_model_lmhead_quant(self, lm_head):
        self.weights.append(lm_head.linear.weight.data)
        self.weights.append(lm_head.linear.quant_bias.data)
        self.weights.append(lm_head.linear.deq_scale.data)
        self.weights.append(lm_head.linear.input_offset.data)
        self.weights.append(lm_head.linear.input_scale.data)
        self.weights.append(self.placeholder)
        self.layer_linear_type.append(LinearType.INT)

    def register_layer(self, layer, quantize_type):
        self.layer_linear_type.clear()
        self.layer_linear_descs.clear()
        self.layer_linear_transpose_types.clear()
        self.layer_is_anti_outlier.clear()
        self.register_layer_attn(layer, self.attn_wrapper, quantize_type)
        self.register_layer_mlp(layer, self.mlp_wrapper, quantize_type)
        self.linear_type.append(self.layer_linear_type.copy())
        self.linear_descs.append(self.layer_linear_descs.copy())
        self.linear_transpose_types.append(self.layer_linear_transpose_types.copy())
        self.is_anti_outlier.append(self.layer_is_anti_outlier.copy())

        attn_pack_type = get_module(layer, self.attn_wrapper.wrapper_name).pack_type
        mlp_pack_type = get_module(layer, self.mlp_wrapper.wrapper_name).pack_type
        self.pack_quant_type.append([attn_pack_type, mlp_pack_type])

    def register_layer_addrmsnormquant(self, layer, attn_wrapper, mlp_wrapper, quantize):
        self.register_layer_addrmsnormquant_by_wrapper(layer, attn_wrapper, quantize)
        self.register_layer_addrmsnormquant_by_wrapper(layer, mlp_wrapper, quantize)

    def register_layer_addrmsnormquant_by_wrapper(self, layer, wrapper, quantize):
        """
        addrmsnormquant层涉及的attn、mlp层注册
        """
        wrapper_module = get_module(layer, wrapper.wrapper_name)
        linear = get_module(wrapper_module, wrapper.pack_name).linear
        # 不回退场景 且 [W8A8、W8A8_PDMIX场景](A2/A3)、[W8A8S场景](310P)
        if (linear.weight.dtype not in [torch.float16, torch.bfloat16] and
                quantize in [QuantType.W8A8, QuantType.W8A8_PDMIX, QuantType.W8A8S]):
            # input_offset根据input_scale类型判断,linear register的原input_offset已被修改过,无法使用原input_offset判断
            input_scale_fill_dtype = torch.float32 if linear.input_scale.dtype == torch.float16 else torch.bfloat16
            input_offset_fill_dtype = torch.int32 if linear.input_scale.dtype == torch.float16 else torch.bfloat16
            shape = linear.weight.shape[-1] if linear.trans_flag == TransposeType.TRANSPOSE else linear.weight.shape[-2]
            self.weights.append(linear.input_scale.data.repeat(shape).to(input_scale_fill_dtype))
            self.weights.append(linear.input_offset.data.repeat(shape).to(input_offset_fill_dtype))
        # 回退/部分回退场景; [FLOAT、W4A16、W8A16、W8A8_DYNAMIC场景](A2/A3)、[W8A8SC场景](310P)
        else:
            self.weights.append(self.placeholder)
            self.weights.append(self.placeholder)