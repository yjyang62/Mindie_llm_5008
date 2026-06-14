# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import ast
import torch
import torch_npu

from atb_llm.utils.quantize.pack_type import LinearType, ALL_PACK_LIST, TransposeType
from atb_llm.utils.quantize.quant_type import QuantType
from atb_llm.utils.data.weight_wrapper import NormWrapper, WeightWrapper

SHARED = "shared"


def get_moe_module(obj, name, delimiter="."):
    names = name.split(delimiter)
    for name in names:
        if name.isdigit():
            obj_idx = ast.literal_eval(name)
            obj = obj[obj_idx]
        else:
            obj = getattr(obj, name)
    return obj


class MoeMlpWrapper(NormWrapper):
    def __init__(self, norm_name, wrapper_name, router_name=None, pack_name=None,
                 sep_names=None, down_name=None, shared_experts=False):
        super().__init__(norm_name)
        self.router_name = router_name
        self.wrapper_name = wrapper_name
        self.pack_name = pack_name if pack_name else None
        self.gate_name = sep_names[0] if sep_names and len(sep_names) == 2 else None
        self.up_name = sep_names[1] if sep_names and len(sep_names) == 2 else None
        self.down_name = down_name
        self.shared_experts = shared_experts
        self.soc_info = None
        self.attn_linear_transpose_types = []
        self.attn_linear_types = []
        self.moe_linear_transpose_types = []


class MoeWeightWrapper(WeightWrapper):
    def __init__(
            self,
            soc_info,
            tp_rank,
            attn_wrapper,
            moe_mlp_wrapper,
            num_experts,
            enable_lcoc_all2all=False,
            moe_is_nzcasted=False
        ):
        super().__init__(soc_info, tp_rank, attn_wrapper, None)
        self.moe_mlp_wrapper = moe_mlp_wrapper
        self.moe_mlp_wrapper.soc_info = soc_info
        self.moe_mlp_wrapper.soc_info.matmul_nd_nz = False
        self.num_experts = num_experts
        self.shared_experts = self.moe_mlp_wrapper.shared_experts
        self.attn_linear_types = []
        self.mlp_linear_types = []
        self.moe_linear_types = []
        self.attn_linear_transpose_types = []
        self.mlp_linear_transpose_types = []
        self.moe_linear_transpose_types = []
        self.gmm_quant_nd_nz = {"gate_up": False, "down": False}
        self.moe_pack_type = None
        self.enable_lcoc_all2all = enable_lcoc_all2all,
        self.enable_lcoc_all2all = enable_lcoc_all2all
        self.moe_is_nzcasted = moe_is_nzcasted
        # [[第1个MoE层权重的起始idx, 第1个MoE层权重的结束idx], [第2个MoE层权重的起始idx, 第2个MoE层权重的结束idx], ...]
        self.buffer_replace_weights_ids = []

    def register_linear_wrapper(self, linear, quantize_type, is_down=False, is_lcoc=False):
        if linear.weight.dtype == torch.float32:
            self.register_linear_bias(linear, enable_nz=False)
            self.weights.extend([self.placeholder] * 4)
            self.layer_linear_type.append(LinearType.FP)
            self.layer_linear_transpose_types.append(linear.trans_flag)
        else:
            super().register_linear_wrapper(linear, quantize_type, is_down=is_down, is_lcoc=is_lcoc)

    def register_linear_list_without_bias(self, linear_list, quantize_type,
                                          hidden_dim, trans_flag, is_down_weight=False):
        tensor_stacked = None
        is_w4 = quantize_type == QuantType.W4A16
        down_weight_align = is_down_weight and linear_list[0].weight.data.shape[1] == hidden_dim
        other_weight_align = not is_down_weight and linear_list[0].weight.data.shape[0] == hidden_dim
        if is_w4 or down_weight_align or other_weight_align:
            tensor_stacked = torch.stack([linear.weight.data for linear in linear_list], dim=0)
        else:
            tensor_stacked = torch.stack([linear.weight.data.transpose(0, 1) for linear in linear_list], dim=0)
            if trans_flag == TransposeType.NOT_TRANSPOSE:
                trans_flag = TransposeType.TRANSPOSE
            elif trans_flag == TransposeType.TRANSPOSE:
                trans_flag = TransposeType.NOT_TRANSPOSE
        if self.gmm_quant_nd_nz["gate_up"]:
            self.weights.append(self.weight_format_cast(tensor_stacked, enable_nz=True))
        else:
            self.weights.append(self.weight_format_cast(tensor_stacked))
        return trans_flag 

    def register_linear_list_bias(self, linear_list, quantize_type, hidden_dim, trans_flag, is_down_weight=False):
        trans_flag = self.register_linear_list_without_bias(linear_list, quantize_type,
                                                            hidden_dim, trans_flag, is_down_weight=is_down_weight)
        if hasattr(linear_list[0], 'bias') and (getattr(linear_list[0], 'bias') is not None):
            bias_stacked = torch.stack([linear.bias.data for linear in linear_list], dim=0)
            if self.gmm_quant_nd_nz["gate_up"]:
                self.weights.append(self.weight_format_cast(bias_stacked, enable_nz=True))
            else:
                self.weights.append(self.weight_format_cast(bias_stacked))
        else:
            self.weights.append(self.placeholder)
        return trans_flag

    def besides_float_and_antiquant(self, linear_list, quantize_type, hidden_dim, trans_flag, is_down_weight):
        trans_flag = self.register_linear_list_without_bias(linear_list, quantize_type,
                                                            hidden_dim, trans_flag, is_down_weight=is_down_weight)
        quant_bias_list = []
        deq_scale_list = []
        input_offset_list = []
        input_scale_list = []
        for linear in linear_list:
            quant_bias_list.append(super().weight_format_cast(linear.quant_bias.data))
            deq_scale_list.append(super().weight_format_cast(linear.deq_scale.data))
            input_offset_list.append(super().weight_format_cast(linear.input_offset.data))
            input_scale_list.append(super().weight_format_cast(linear.input_scale.data))
        self.weights.append(torch.stack(quant_bias_list, dim=0))
        self.weights.append(torch.stack(deq_scale_list, dim=0))
        self.weights.append(torch.stack(input_offset_list, dim=0))
        self.weights.append(torch.stack(input_scale_list, dim=0))
        del quant_bias_list
        del deq_scale_list
        del input_offset_list
        del input_scale_list

        if quantize_type == QuantType.W8A8SC:
            for linear in linear_list:
                self.weights.append(super().weight_format_cast(linear.index.data))
        else:
            self.weights.append(self.placeholder)
        self.layer_linear_type.append(LinearType.INT)
        return trans_flag

    def register_linear_list_wrapper(self, linear_list, quantize_type, hidden_dim, is_down_weight=False):
        trans_flag = linear_list[0].trans_flag
        if linear_list[0].weight.dtype in [torch.float16, torch.bfloat16]:
            trans_flag = self.register_linear_list_bias(linear_list, quantize_type,
                                                        hidden_dim, trans_flag, is_down_weight=is_down_weight)
            self.weights.extend([self.placeholder] * 4)
            self.layer_linear_type.append(LinearType.FP)
        elif quantize_type in [QuantType.W4A16, QuantType.W8A16, QuantType.W8A8_DYNAMIC]:
            trans_flag = self.register_linear_list_bias(linear_list, quantize_type,
                                                        hidden_dim, trans_flag, is_down_weight=is_down_weight)
            self.weights.append(self.placeholder)
            offset_list, scale_list = [], []
            for linear in linear_list:
                offset_list.append(linear.weight_offset.data)
                if quantize_type == QuantType.W8A8_DYNAMIC:
                    scale_dtype = \
                        linear.weight_scale.dtype if linear.weight_scale.dtype == torch.bfloat16 else torch.float32
                    scale_list.append(linear.weight_scale.data.type(scale_dtype))
                else:
                    scale_list.append(linear.weight_scale.data)
            self.weights.append(torch.stack(offset_list, dim=0))
            self.weights.append(torch.stack(scale_list, dim=0))
            del offset_list, scale_list
            self.weights.append(self.placeholder)
            self.layer_linear_type.append(LinearType.INT)
        else:
            trans_flag = self.besides_float_and_antiquant(linear_list, quantize_type, hidden_dim,
                                                          trans_flag, is_down_weight=is_down_weight)
        self.layer_linear_transpose_types.append(trans_flag)

    def register_layer_linear_pack(self, layer, wrapper, quantize_type, linear_type='attn'):
        wrapper_module = get_moe_module(layer, wrapper.wrapper_name)
        pack_type = wrapper_module.pack_type
        if linear_type == "shared_mlp":
            self.register_layer_norm(layer, wrapper, pack_type)
            linear = get_moe_module(wrapper_module, f"shared_experts.{wrapper.pack_name}").linear
            self.register_linear_wrapper(linear, quantize_type)
        elif linear_type == "dense_mlp":
            self.register_layer_norm(layer, wrapper, pack_type)
            linear = get_moe_module(wrapper_module, wrapper.pack_name).linear
            self.register_linear_wrapper(linear, quantize_type)
        else:
            super().register_layer_norm(layer, wrapper, pack_type)
            linear = get_moe_module(wrapper_module, wrapper.pack_name).linear
            super().register_linear_wrapper(linear, quantize_type)
        if linear_type == 'attn':
            self.weights.extend([self.placeholder] * 12)
            self.layer_linear_type.extend([LinearType.INVALID, LinearType.INVALID])
            self.layer_linear_transpose_types.extend([LinearType.INVALID, LinearType.INVALID])
        elif linear_type == 'mlp':
            self.layer_linear_type.extend([LinearType.INVALID])
            self.layer_linear_transpose_types.extend([LinearType.INVALID])
        elif linear_type == "shared_mlp" or linear_type == "dense_mlp":
            self.layer_linear_type.extend([LinearType.INVALID])
            self.layer_linear_transpose_types.extend([LinearType.INVALID])
        else:
            raise AssertionError(f'{linear_type} not yet implemented in register_layer_linear_pack')

    def register_layer_linear_sep(self, layer, wrapper, quantize_type, linear_type='attn'):
        wrapper_module = get_moe_module(layer, wrapper.wrapper_name)
        pack_type = wrapper_module.pack_type
        super().register_layer_norm(layer, wrapper, pack_type)
        if linear_type == 'attn':
            q_linear = get_moe_module(wrapper_module, wrapper.q_name).linear
            k_linear = get_moe_module(wrapper_module, wrapper.k_name).linear
            v_linear = get_moe_module(wrapper_module, wrapper.v_name).linear
            super().register_linear_wrapper(q_linear, quantize_type)
            super().register_linear_wrapper(k_linear, quantize_type)
            super().register_linear_wrapper(v_linear, quantize_type)
        elif linear_type == 'mlp':
            gate_linear = get_moe_module(wrapper_module, wrapper.gate_name).linear
            up_linear = get_moe_module(wrapper_module, wrapper.up_name).linear
            super().register_linear_wrapper(gate_linear, quantize_type)
            super().register_linear_wrapper(up_linear, quantize_type)
        else:
            raise AssertionError(f'{linear_type} not yet implemented in register_layer_linear_sep')

    def register_layer_moe_linear_pack(self, layer, wrapper, quantize_type, linear_type, expert_roster):
        wrapper_module = get_moe_module(layer, wrapper.wrapper_name)
        pack_type = wrapper_module.pack_type
        if not self.shared_experts:
            self.register_layer_norm(layer, wrapper, pack_type)
        if linear_type == 'moe_mlp':
            router = get_moe_module(wrapper_module, wrapper.router_name)
            self.register_linear_wrapper(router, quantize_type, is_down=False, is_lcoc=self.enable_lcoc_all2all)
            if self.moe_is_nzcasted:
                self.buffer_replace_weights_ids[-1].append(len(self.weights))
            linear_list = []
            for i in expert_roster:
                pack_name = f"{wrapper.pack_name}.{i}"
                linear_list.append(get_moe_module(wrapper_module, pack_name).linear)
            self.register_linear_list_wrapper(linear_list, quantize_type, hidden_dim=wrapper_module.hidden_dim)
            self.layer_linear_type.extend([LinearType.INVALID])
            self.layer_linear_transpose_types.extend([LinearType.INVALID])
        else:
            raise AssertionError(f'{linear_type} not yet implemented in register_layer_linear_pack')

    def register_layer_moe_gate_up_stacked(self, layer, wrapper, quantize_type, enable_dangling=False):
        wrapper_module = get_moe_module(layer, wrapper.wrapper_name)
        pack_type = wrapper_module.pack_type
        if not (self.shared_experts or enable_dangling):
            self.register_layer_norm(layer, wrapper, pack_type)

        router = get_moe_module(wrapper_module, wrapper.router_name)
        self.register_linear_wrapper(router, quantize_type)
        if self.moe_is_nzcasted:
            self.buffer_replace_weights_ids[-1].append(len(self.weights))
        linear = get_moe_module(wrapper_module, wrapper.pack_name).linear
        if quantize_type == QuantType.W8A8_DYNAMIC and not self.moe_mlp_wrapper.soc_info.need_nz:
            if self.moe_is_nzcasted:
                if linear.trans_flag == TransposeType.NOT_TRANSPOSE:
                    linear.trans_flag = TransposeType.TRANSPOSE
                elif linear.trans_flag == TransposeType.TRANSPOSE:
                    linear.trans_flag = TransposeType.NOT_TRANSPOSE
            else:
                if linear.trans_flag == TransposeType.TRANSPOSE:
                    linear.weight.data = linear.weight.data.transpose(1, 2).contiguous()
                    linear.trans_flag = TransposeType.NOT_TRANSPOSE
        if self.gmm_quant_nd_nz["gate_up"]:
            self.moe_mlp_wrapper.soc_info.matmul_nd_nz = True
        self.register_linear_wrapper(linear, quantize_type, is_lcoc=self.enable_lcoc_all2all)
        self.moe_mlp_wrapper.soc_info.matmul_nd_nz = False
        self.layer_linear_type.extend([LinearType.INVALID])
        self.layer_linear_transpose_types.extend([LinearType.INVALID])

    def register_layer_moe_linear_sep(self, layer, wrapper, quantize_type, linear_type, expert_roster):
        wrapper_module = get_moe_module(layer, wrapper.wrapper_name)
        pack_type = wrapper_module.pack_type
        self.register_layer_norm(layer, wrapper, pack_type)
        if linear_type == 'moe_mlp':
            router = get_moe_module(wrapper_module, wrapper.router_name)
            self.register_linear_wrapper(router, quantize_type)
            if self.moe_is_nzcasted:
                self.buffer_replace_weights_ids[-1].append(len(self.weights))
            gate_linear_list = []
            up_linear_list = []
            for i in expert_roster:
                gate_name = f"{wrapper.gate_name}.{i}"
                up_name = f"{wrapper.up_name}.{i}"
                gate_linear_list.append(get_moe_module(wrapper_module, gate_name).linear)
                up_linear_list.append(get_moe_module(wrapper_module, up_name).linear)
            self.register_linear_list_wrapper(gate_linear_list, quantize_type, hidden_dim=wrapper_module.hidden_dim)
            self.register_linear_list_wrapper(up_linear_list, quantize_type, hidden_dim=wrapper_module.hidden_dim)
        else:
            raise AssertionError(f'{linear_type} not yet implemented in register_layer_linear_pack')

    def register_layer_moe_mlp(self, layer, wrapper, quantize_type, linear_type='mlp'):
        wrapper_module = get_moe_module(layer, wrapper.wrapper_name)
        pack_type = wrapper_module.pack_type
        if pack_type in ALL_PACK_LIST:
            self.register_layer_linear_pack(layer, wrapper, quantize_type, linear_type)
        else:
            self.register_layer_linear_sep(layer, wrapper, quantize_type, linear_type)
        if linear_type == "shared_mlp":
            down_linear = get_moe_module(wrapper_module, f"shared_experts.{wrapper.down_name}").linear
        else:
            down_linear = get_moe_module(wrapper_module, wrapper.down_name).linear
        self.register_linear_wrapper(down_linear, quantize_type)

    def register_layer_moe_mlp_experts(self, layer, wrapper, quantize_type, expert_roster, enable_dangling=False):
        if self.moe_is_nzcasted:
            self.buffer_replace_weights_ids.append([])
        wrapper_module = get_moe_module(layer, wrapper.wrapper_name)
        pack_type = wrapper_module.pack_type
        if pack_type in ALL_PACK_LIST:
            gate_up_layer = get_moe_module(wrapper_module, wrapper.pack_name)
            if isinstance(gate_up_layer, torch.nn.ModuleList):
                self.register_layer_moe_linear_pack(layer, wrapper, quantize_type, 'moe_mlp', expert_roster)
            else:
                self.register_layer_moe_gate_up_stacked(layer, wrapper, quantize_type, enable_dangling)
        else:
            self.register_layer_moe_linear_sep(layer, wrapper, quantize_type, 'moe_mlp', expert_roster)

        down_layer = get_moe_module(wrapper_module, wrapper.down_name)
        if isinstance(down_layer, torch.nn.ModuleList):
            down_linear_list = []
            for i in expert_roster:
                down_name = f"{wrapper.down_name}.{i}"
                down_linear_list.append(get_moe_module(wrapper_module, down_name).linear)
            self.register_linear_list_wrapper(down_linear_list, quantize_type,
                                            hidden_dim=wrapper_module.hidden_dim, is_down_weight=True)
        else:
            if self.gmm_quant_nd_nz["down"]:
                self.moe_mlp_wrapper.soc_info.matmul_nd_nz = True
            self.register_linear_wrapper(down_layer.linear, quantize_type, is_lcoc=self.enable_lcoc_all2all)
            self.moe_mlp_wrapper.soc_info.matmul_nd_nz = False
        if self.moe_is_nzcasted:
            self.buffer_replace_weights_ids[-1].append(len(self.weights))

    def pad_linear_types(self, list_type, target_length):
        if list_type == "attention":
            self.attn_linear_types.append(self.layer_linear_type.copy())
            self.attn_linear_transpose_types.append(self.layer_linear_transpose_types.copy())
            for _ in range(target_length - len(self.attn_linear_types[-1])):
                self.attn_linear_types[-1].append(LinearType.INVALID)
                self.attn_linear_transpose_types[-1].append(-1)
        elif list_type == "shared":
            self.mlp_linear_types.append(self.layer_linear_type.copy())
            self.mlp_linear_transpose_types.append(self.layer_linear_transpose_types.copy())
            for _ in range(target_length - len(self.mlp_linear_types[-1])):
                self.mlp_linear_types[-1].append(LinearType.INVALID)
                self.mlp_linear_transpose_types[-1].append(-1)
        elif list_type == "moe":
            self.moe_linear_types.append(self.layer_linear_type.copy())
            self.moe_linear_transpose_types.append(self.layer_linear_transpose_types.copy())
            for _ in range(target_length - len(self.moe_linear_types[-1])):
                self.moe_linear_types[-1].append(LinearType.INVALID)
                self.moe_linear_transpose_types[-1].append(-1)
        self.layer_linear_type.clear()
        self.layer_linear_transpose_types.clear()

    def set_gmm_nd_nz(self, quantize_type, enable_atlas_gmm_fused):
        if self.moe_is_nzcasted:
            return
        if (quantize_type == QuantType.W8A8_DYNAMIC or quantize_type == QuantType.W4A8_DYNAMIC):
            self.gmm_quant_nd_nz["gate_up"] = True
        if (quantize_type == QuantType.W4A8_DYNAMIC or enable_atlas_gmm_fused):
            self.gmm_quant_nd_nz["down"] = True

    def register_moe_layer(self, layer, quantize_type, dense_layer=False, expert_roster=None, attn_quantize_type=None,
                            **kwargs):
        if attn_quantize_type is None:
            attn_quantize_type = quantize_type
        moe_quantize_type = kwargs.get('moe_quantize_type', quantize_type)
        ep_rank = kwargs.get('ep_rank', 0)
        num_dangling_shared_experts = kwargs.get('num_dangling_shared_experts', 0)
        mix_shared_routing = kwargs.get('mix_shared_routing', 0)
        enable_atlas_gmm_fused = kwargs.get('enable_atlas_gmm_fused', 0)
        if_shared_expert = False
        enable_dangling = num_dangling_shared_experts > 0
        if not expert_roster:
            expert_roster = [i for i in range(self.num_experts)]
        self.register_layer_attn(layer, self.attn_wrapper, attn_quantize_type)
        if "qk_norm" in kwargs.keys():
            if kwargs["qk_norm"]:
                self.register_model_norm(layer.self_attn.q_norm)  # q_norm
                self.register_model_norm(layer.self_attn.k_norm)  # k_norm
            else:
                self.weights.extend([self.placeholder] * 2)
        self.pad_linear_types(list_type="attention", target_length=6)
        if dense_layer:
            self.set_gmm_nd_nz(quantize_type, enable_atlas_gmm_fused)
            self.register_layer_moe_mlp(layer, self.moe_mlp_wrapper, quantize_type, "dense_mlp")
            self.weights.append(self.placeholder)
            self.weights.append(self.placeholder)
            self.weights.extend([self.placeholder] * 20) # place holder for gmm quant & router quant
            self.weights.extend([self.placeholder] * 2)
            self.pad_linear_types(list_type=SHARED, target_length=4)
            self.pad_linear_types(list_type="moe", target_length=4)
        else:
            self.set_gmm_nd_nz(moe_quantize_type, enable_atlas_gmm_fused)
            if num_dangling_shared_experts == 0 and not mix_shared_routing:
                if self.shared_experts:
                    if_shared_expert = True
                    self.register_layer_moe_mlp(layer, self.moe_mlp_wrapper, quantize_type, "shared_mlp")
                    self.weights.append(self.placeholder)
                    self.weights.extend([self.placeholder] * 5)
                    self.pad_linear_types(list_type=SHARED, target_length=4)
                self.register_layer_moe_mlp_experts(layer, self.moe_mlp_wrapper, moe_quantize_type,
                                                    expert_roster, enable_dangling)
            else:
                if ep_rank < num_dangling_shared_experts:
                    wrapper_module = get_moe_module(layer, self.moe_mlp_wrapper.wrapper_name)
                    pack_type = wrapper_module.pack_type
                    shared_expert_weight_start_idx = len(self.weights) + 4
                    self.register_layer_moe_mlp(layer, self.moe_mlp_wrapper, quantize_type, "shared_mlp")
                    shared_expert_weight_end_idx = len(self.weights)
                    self.weights.extend([self.placeholder] * 6) # shared experts gate
                    router = get_moe_module(wrapper_module, self.moe_mlp_wrapper.router_name)
                    self.register_linear_wrapper(router, quantize_type)
                    self.weights.extend(self.weights[shared_expert_weight_start_idx:shared_expert_weight_end_idx])
                    
                    # shared experts gate 和 routed experts gate
                    expert_weight_start_idx = shared_expert_weight_end_idx + 12
                    # 共享专家GMM1权重转置
                    self.weights[expert_weight_start_idx] = \
                        self.weights[expert_weight_start_idx][None, :].permute(0, 2, 1).contiguous()
                    self.weights[expert_weight_start_idx] = \
                        torch_npu.npu_format_cast(self.weights[expert_weight_start_idx], 29) # 转回nz
                    self.weights[expert_weight_start_idx + 6] = \
                        self.weights[expert_weight_start_idx + 6][None, :].contiguous() # 共享专家GMM2权重转置
                    self.weights[expert_weight_start_idx + 6] = \
                        torch_npu.npu_format_cast(self.weights[expert_weight_start_idx + 6], 29)  # 转nz
                    if self.weights[expert_weight_start_idx + 3].shape[0] != 1:
                        ### 非MTP layer 量化系数增维
                        self.weights[expert_weight_start_idx + 3] = \
                            self.weights[expert_weight_start_idx + 3][None, :, None]
                        self.weights[expert_weight_start_idx + 4] = \
                            self.weights[expert_weight_start_idx + 4][None, :, None]
                        self.weights[expert_weight_start_idx + 9] = \
                            self.weights[expert_weight_start_idx + 9][None, :, None]
                        self.weights[expert_weight_start_idx + 10] = \
                            self.weights[expert_weight_start_idx + 10][None, :, None]
                        self.layer_linear_type = [LinearType.FP, LinearType.INT, LinearType.INVALID, LinearType.INT]
                    else:
                        ### MTP layer无量化
                        self.layer_linear_type = [LinearType.FP, LinearType.FP, LinearType.INVALID, LinearType.FP]
                        
                    self.layer_linear_transpose_types = [
                        TransposeType.TRANSPOSE,
                        TransposeType.NOT_TRANSPOSE,
                        LinearType.INVALID,
                        TransposeType.TRANSPOSE
                    ]
                else:
                    wrapper_module = get_moe_module(layer, self.moe_mlp_wrapper.wrapper_name)
                    pack_type = wrapper_module.pack_type
                    self.register_layer_norm(layer, self.moe_mlp_wrapper, pack_type)
                    self.weights.extend([self.placeholder] * 18)
                    self.register_layer_moe_mlp_experts(layer, self.moe_mlp_wrapper, moe_quantize_type,
                                                        expert_roster, enable_dangling)
            self.pad_linear_types(list_type="moe", target_length=4)
            if not if_shared_expert:
                self.pad_linear_types(list_type=SHARED, target_length=4)

        attn_pack_type = get_moe_module(layer, self.attn_wrapper.wrapper_name).pack_type
        moe_mlp_pack_type = get_moe_module(layer, self.moe_mlp_wrapper.wrapper_name).pack_type
        self.pack_quant_type.append([attn_pack_type, moe_mlp_pack_type])
        if not dense_layer and self.moe_pack_type is None:
            self.moe_pack_type = getattr(get_moe_module(layer, self.moe_mlp_wrapper.wrapper_name),
                                        "moe_pack_type", None)

    def register_router(self, layer, quantize_type, name="gate"):
        wrapper_module = get_moe_module(layer, self.moe_mlp_wrapper.wrapper_name)
        router = get_moe_module(wrapper_module, name)
        self.register_linear_wrapper_copy(router, quantize_type)

    def register_shared_expert_dp2tp(self, layer, quantize_type, name="shared_experts_tp"):
        wrapper_module = get_moe_module(layer, self.moe_mlp_wrapper.wrapper_name)
        linear = get_moe_module(wrapper_module, f"{name}.{self.moe_mlp_wrapper.pack_name}").linear
        self.register_linear_wrapper_copy(linear, quantize_type)
        down_linear = get_moe_module(wrapper_module, f"{name}.{self.moe_mlp_wrapper.down_name}").linear
        self.register_linear_wrapper_copy(down_linear, quantize_type)
        self.weights.append(self.placeholder)
        self.weights.extend([self.placeholder] * 5)

    def register_linear_wrapper_copy(self, linear, quantize_type):
        if linear.weight.dtype in [torch.float16, torch.bfloat16]:
            self.register_linear_bias(linear, enable_nz=False)
            self.weights.extend([self.placeholder] * 4)
        elif quantize_type in [QuantType.W4A16, QuantType.W8A16, QuantType.W8A8_DYNAMIC, QuantType.W4A8_DYNAMIC]:
            self.register_linear_bias(linear, enable_nz=False)
            self.weights.append(self.placeholder)
            self.weights.append(linear.weight_offset.data)
            self.weights.append(linear.weight_scale.data)
            self.weights.append(self.placeholder)
        else:
            linear.weight.data = self.weight_format_cast(linear.weight.data)
            self.weights.append(linear.weight.data)
            self.weights.append(linear.quant_bias.data)
            self.weights.append(linear.deq_scale.data)
            self.weights.append(linear.input_offset.data)
            self.weights.append(linear.input_scale.data)
            if quantize_type == QuantType.W8A8SC:
                self.weights.append(linear.index.data)
            else:
                self.weights.append(self.placeholder)