# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from atb_llm.utils.quantize.pack_type import LinearType
from atb_llm.utils.quantize.quant_type import QuantType
from atb_llm.utils.data.weight_wrapper import get_module
from atb_llm.utils.data.moe_weight_wrapper import MoeWeightWrapper
import torch


class Deepseekv2WeightWrapper(MoeWeightWrapper):
    def __init__(self,
                 soc_info,
                 tp_rank,
                 mla_wrapper,
                 moe_mlp_wrapper,
                 num_experts,
                 enable_lcoc_all2all=False,
                 moe_is_nzcasted=False
                 ):
        super().__init__(
            soc_info,
            tp_rank,
            mla_wrapper,
            moe_mlp_wrapper,
            num_experts,
            enable_lcoc_all2all=enable_lcoc_all2all,
            moe_is_nzcasted=moe_is_nzcasted
        )
        self.ein_weight_idx = []
        self.enable_lcoc_tp = False

    def register_moe_layer(self, 
                           layer, 
                           quantize_type, 
                           dense_layer=False, 
                           expert_roster=None, 
                           attn_quantize_type=None,
                           **kwargs):
        self.enable_lcoc_tp = kwargs.get("enable_lcoc_tp", False)
        super().register_moe_layer(layer, quantize_type, dense_layer, expert_roster, attn_quantize_type, **kwargs)

    def register_layer_attn(self, layer, wrapper, quantize_type):
        wrapper_module = get_module(layer, wrapper.wrapper_name)
        pack_type = wrapper_module.pack_type
        self.register_layer_norm(layer, wrapper, pack_type)

        if hasattr(wrapper_module, "q_a_proj"):
            if not self.enable_lcoc_tp:
                wrapper_module.q_a_proj.linear.weight.data = torch.cat(
                    (wrapper_module.kv_a_proj_with_mqa.linear.weight.data, wrapper_module.q_a_proj.linear.weight.data), 
                    dim=0).contiguous()
                wrapper_module.kv_a_proj_with_mqa.linear.weight.data = self.placeholder
                torch.npu.config.allow_internal_format = True
                wrapper_module.q_a_proj.linear.weight.data = self.weight_format_cast(
                    wrapper_module.q_a_proj.linear.weight.data, enable_nz=self._enable_mlapo_nz(quantize_type),
                    nd_weight=wrapper_module.q_a_proj.linear.nd_weight)
                if quantize_type == QuantType.W8A8:
                    wrapper_module.q_a_proj.linear.quant_bias.data = torch.cat(
                        (wrapper_module.kv_a_proj_with_mqa.linear.quant_bias.data,
                        wrapper_module.q_a_proj.linear.quant_bias.data), dim=0).contiguous()
                    wrapper_module.q_a_proj.linear.deq_scale.data = torch.cat(
                        (wrapper_module.kv_a_proj_with_mqa.linear.deq_scale.data,
                        wrapper_module.q_a_proj.linear.deq_scale.data), dim=0).contiguous()
                elif quantize_type == QuantType.W8A16 or quantize_type == QuantType.W8A8_DYNAMIC:
                    wrapper_module.q_a_proj.linear.weight_scale.data = torch.cat(
                        (wrapper_module.kv_a_proj_with_mqa.linear.weight_scale.data,
                        wrapper_module.q_a_proj.linear.weight_scale.data), dim=0).contiguous()
                    wrapper_module.q_a_proj.linear.weight_offset.data = torch.cat(
                        (wrapper_module.kv_a_proj_with_mqa.linear.weight_offset.data,
                        wrapper_module.q_a_proj.linear.weight_offset.data), dim=0).contiguous()
            self.register_linear_wrapper(wrapper_module.q_a_proj.linear, quantize_type)
            self.register_norm(wrapper_module.q_a_layernorm)
            # deepseekv2 no bias
            torch.npu.config.allow_internal_format = True
            if not self.enable_lcoc_tp:
                wrapper_module.q_b_proj.linear.weight.data = self.weight_format_cast(
                    wrapper_module.q_b_proj.linear.weight.data, enable_nz=self._enable_mlapo_nz(quantize_type),
                    nd_weight=wrapper_module.q_b_proj.linear.nd_weight)
            self.register_linear_wrapper(wrapper_module.q_b_proj.linear, quantize_type)
        else:
            # deepseekv2 no bias
            self.register_linear_wrapper(wrapper_module.q_proj.linear, quantize_type)
            # if not qlora，add 8 placeholders.
            self.weights.extend([self.placeholder] * 8)
            self.layer_linear_type.append(LinearType.FP)
            self.layer_linear_transpose_types.append(LinearType.INVALID)

        self.register_linear_wrapper(wrapper_module.kv_a_proj_with_mqa.linear, quantize_type)
        self.register_norm(wrapper_module.kv_a_layernorm)

        self.register_linear_wrapper(wrapper_module.k_b_proj.linear, quantize_type)
        self.ein_weight_idx.append(len(self.weights))
        self.register_linear_wrapper(wrapper_module.v_b_proj.linear, quantize_type)
        self.register_linear_wrapper(wrapper_module.o_proj.linear, quantize_type)

    def _enable_mlapo_nz(self, quantize_type):
        if quantize_type == QuantType.W8A8_DYNAMIC:
            return False
        return True