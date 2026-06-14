# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import unittest
from unittest.mock import MagicMock, patch
from unittest import TestCase
import torch

from atb_llm.utils.data.moe_weight_wrapper import MoeMlpWrapper, MoeWeightWrapper, WeightWrapper
from atb_llm.utils.initial import NPUSocInfo
from atb_llm.utils.data.weight_wrapper import AttnWrapper
from atb_llm.utils.quantize.quant_type import QuantType
from atb_llm.utils.quantize.pack_type import PackType, TransposeType


class TestMoeMlpWrapper(TestCase):
    def test_moemlpwrapper(self):
        MoeMlpWrapper("norm", "wrapper")


class TestMoeWeightWrapper(TestCase):
    def setUp(self):
        mock_linear, mock_layer, mock_layer_stacked = MagicMock(), MagicMock(), MagicMock()
        mock_linear.linear.weight.data = torch.empty(100, 100, dtype=torch.float16)
        mock_linear.linear.weight.dtype = torch.float16
        mock_linear.linear.bias = None
        mock_linear.linear.quant_bias.data = torch.empty(100, 100, dtype=torch.float16)
        mock_linear.linear.deq_scale.data = torch.empty(100, 100, dtype=torch.float16)
        mock_linear.linear.input_offset.data = torch.empty(100, 100, dtype=torch.float16)
        mock_linear.linear.input_scale.data = torch.empty(100, 100, dtype=torch.float16)
        mock_linear.linear.weight_offset.data = torch.empty(100, 100, dtype=torch.float16)
        mock_linear.linear.weight_scale.data = torch.empty(100, 100, dtype=torch.float16)
        mock_linear.linear.index.data = 1
        mock_layer.attn.pack_type = PackType.ALL_FP
        mock_layer_stacked.attn.pack_type = PackType.ALL_FP
        mock_layer.attn.qkv = mock_linear
        mock_layer_stacked.attn.qkv = mock_linear
        mock_layer_stacked.mlp.pack_type = PackType.ALL_FP
        mock_layer.mlp.gate_up_proj = [mock_linear for _ in range(64)]
        mock_layer_stacked.mlp.gate_up_proj = mock_linear
        mock_layer.mlp.gate_proj = [mock_linear for _ in range(64)]
        mock_layer_stacked.mlp.gate_proj = mock_linear
        mock_layer.mlp.up_proj = [mock_linear for _ in range(64)]
        mock_layer_stacked.mlp.up_proj = mock_linear
        mock_layer.mlp.down_proj = [mock_linear for _ in range(64)]
        mock_layer_stacked.mlp.down_proj = mock_linear
        self.mock_layer = mock_layer
        self.mock_layer_stacked = mock_layer_stacked
        self.linear_list = [mock_linear.linear for _ in range(64)]
        soc_info = NPUSocInfo()
        self.attn_wrapper = AttnWrapper("norm", "attn", "qkv")
        self.moe_mlp_wrapper = MoeMlpWrapper(
            norm_name='post_attention_layernorm',
            router_name='gate',
            wrapper_name='mlp',
            pack_name='gate_up_proj',
            sep_names=['gate_proj', 'up_proj'],
            down_name='down_proj',
            shared_experts=True
        )
        self.weight_wrapper = MoeWeightWrapper(soc_info, 0, self.attn_wrapper, self.moe_mlp_wrapper, 64)
        self.mock_format_cast = patch("torch_npu.npu_format_cast").start()



    @patch.object(MoeWeightWrapper, "register_layer_attn")
    def test_register_moe_layer_fp16(self, mock_layer_attn):
        def mock_increase(self, linear, quantize_type, is_down=False, is_lcoc=False):
            self.weights.extend([torch.zeros(4, 4)] * 6)
        with patch.object(WeightWrapper, "register_linear_wrapper", mock_increase):
            self.weight_wrapper.weights = [torch.zeros(4, 4, 4)] * 16
            self.weight_wrapper.register_moe_layer(self.mock_layer_stacked,
                                                   PackType.ALL_FP, dense_layer=True, qk_norm=True)
            self.weight_wrapper.register_moe_layer(self.mock_layer_stacked,
                                                   PackType.ALL_FP, dense_layer=False, qk_norm=True)
            self.weight_wrapper.register_moe_layer(self.mock_layer_stacked,
                                                   PackType.ALL_W8A8_DYNAMIC, dense_layer=False,
                                                   qk_norm=True, num_dangling_shared_experts=1)
            self.weight_wrapper.register_moe_layer(self.mock_layer_stacked,
                                                   PackType.ALL_W8A8_DYNAMIC, dense_layer=False,
                                                   qk_norm=True, num_dangling_shared_experts=1, ep_rank=2)
        

    def test_besides_float_and_antiquant_w4a16(self):
        self.weight_wrapper.besides_float_and_antiquant(self.linear_list, QuantType.W4A16,
                                                        8192, TransposeType.TRANSPOSE, False)

    def test_register_linear_list_wrapper(self):
        self.weight_wrapper.register_linear_list_wrapper(self.linear_list, QuantType.FLOAT, 8192, False)
        self.linear_list[0].weight.dtype = torch.int8
        self.weight_wrapper.register_linear_list_wrapper(self.linear_list, QuantType.W8A8_DYNAMIC, 8192, False)
                
    def test_register_layer_linear_pack_attn(self):
        self.weight_wrapper.register_layer_linear_pack(self.mock_layer, self.attn_wrapper, PackType.ALL_FP, "attn")
                
    def test_register_layer_linear_sep(self):
        self.attn_wrapper = AttnWrapper("norm", "attn", sep_names=['q', 'k', 'v'])
        self.weight_wrapper.register_layer_linear_sep(self.mock_layer, self.attn_wrapper, PackType.ALL_FP, "attn")
        self.mock_layer.mlp.gate_proj = self.mock_layer
        self.mock_layer.mlp.up_proj = self.mock_layer
        self.weight_wrapper.register_layer_linear_sep(self.mock_layer, self.moe_mlp_wrapper, PackType.ALL_FP, "mlp")

    def test_register_linear_wrapper(self):
        self.linear_list[0].weight.dtype = torch.float32
        self.weight_wrapper.register_linear_wrapper(self.linear_list[0], QuantType.FLOAT)
                
    def test_register_layer_moe_linear_sep(self):
        self.weight_wrapper.register_layer_moe_linear_sep(self.mock_layer, self.moe_mlp_wrapper,
                                                          PackType.ALL_FP, 'moe_mlp', [i for i in range(64)])
                                                                                                                                                                    
    def test_register_linear_wrapper_copy_fp16(self):
        mock_linear = self.linear_list[0]
        self.weight_wrapper.register_linear_wrapper_copy(mock_linear, QuantType.FLOAT)
                
    def test_register_linear_wrapper_copy_w4a16(self):
        mock_linear = self.linear_list[0]
        mock_linear.weight.dtype = torch.int8
        self.weight_wrapper.register_linear_wrapper_copy(mock_linear, QuantType.W4A16)
        
    def test_register_linear_wrapper_copy_w8a8sc(self):
        mock_linear = self.linear_list[0]
        mock_linear.weight.dtype = torch.int8
        self.weight_wrapper.register_linear_wrapper_copy(mock_linear, QuantType.W8A8SC)

    def test_register_layer_moe_linear_pack(self):
        mock_linear, mock_layer, mock_layer_stacked = MagicMock(), MagicMock(), MagicMock()
        mock_layer.attn.pack_type = PackType.ALL_FP
        mock_layer_stacked.attn.pack_type = PackType.ALL_FP
        mock_layer.attn.qkv = mock_linear
        mock_layer_stacked.attn.qkv = mock_linear
        mock_layer.mlp.gate_up_proj = [mock_linear for _ in range(64)]
        mock_layer_stacked.mlp.gate_up_proj = mock_linear
        mock_layer.mlp.gate_proj = [mock_linear for _ in range(64)]
        mock_layer_stacked.mlp.gate_proj = mock_linear
        mock_layer.mlp.up_proj = [mock_linear for _ in range(64)]
        mock_layer_stacked.mlp.up_proj = mock_linear
        mock_layer.mlp.down_proj = [mock_linear for _ in range(64)]
        mock_layer_stacked.mlp.down_proj = mock_linear
        self.weight_wrapper.register_layer_moe_linear_pack(
            self.mock_layer, 
            self.moe_mlp_wrapper, 
            QuantType.W8A8SC, 
            'moe_mlp', 
            [0]
            )

    def test_register_shared_expert_dp2tp(self):
        mock_linear, mock_layer, mock_layer_stacked = MagicMock(), MagicMock(), MagicMock()
        mock_layer.attn.pack_type = PackType.ALL_FP
        mock_layer_stacked.attn.pack_type = PackType.ALL_FP
        mock_layer.attn.qkv = mock_linear
        mock_layer_stacked.attn.qkv = mock_linear
        mock_layer.mlp.gate_up_proj = [mock_linear for _ in range(64)]
        mock_layer_stacked.mlp.gate_up_proj = mock_linear
        mock_layer.mlp.gate_proj = [mock_linear for _ in range(64)]
        mock_layer_stacked.mlp.gate_proj = mock_linear
        mock_layer.mlp.up_proj = [mock_linear for _ in range(64)]
        mock_layer_stacked.mlp.up_proj = mock_linear
        mock_layer.mlp.down_proj = [mock_linear for _ in range(64)]
        mock_layer_stacked.mlp.down_proj = mock_linear
        self.mock_layer = mock_layer
        self.weight_wrapper.register_shared_expert_dp2tp(
            self.mock_layer, 
            QuantType.W8A8SC,
            "shared_experts_tp"
        )
    
    def test_register_layer_qkvquant(self):
        mock_linear, mock_layer, mock_layer_stacked = MagicMock(), MagicMock(), MagicMock()
        mock_layer.attn.pack_type = PackType.ALL_FP
        mock_layer_stacked.attn.pack_type = PackType.ALL_FP
        mock_layer.attn.qkv = mock_linear
        mock_layer_stacked.attn.qkv = mock_linear
        mock_layer.mlp.gate_up_proj = [mock_linear for _ in range(64)]
        mock_layer_stacked.mlp.gate_up_proj = mock_linear
        mock_layer.mlp.gate_proj = [mock_linear for _ in range(64)]
        mock_layer_stacked.mlp.gate_proj = mock_linear
        mock_layer.mlp.up_proj = [mock_linear for _ in range(64)]
        mock_layer_stacked.mlp.up_proj = mock_linear
        mock_layer.mlp.down_proj = [mock_linear for _ in range(64)]
        mock_layer_stacked.mlp.down_proj = mock_linear
        self.mock_layer = mock_layer
        self.weight_wrapper.register_layer_qkvquant(self.mock_layer)
        
if __name__ == '__main__':
    unittest.main()