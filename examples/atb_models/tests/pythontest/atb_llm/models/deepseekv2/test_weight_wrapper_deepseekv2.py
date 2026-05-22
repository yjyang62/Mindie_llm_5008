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
from unittest.mock import patch, MagicMock
import torch

from atb_llm.models.deepseekv2.weight_wrapper_deepseekv2 import Deepseekv2WeightWrapper
from atb_llm.utils.data.weight_wrapper import AttnWrapper
from atb_llm.utils.data.moe_weight_wrapper import MoeMlpWrapper
from atb_llm.utils.initial import NPUSocInfo


class TestDeepseekv2WeightWrapper(unittest.TestCase):
    @patch("atb_llm.utils.data.weight_wrapper.WeightWrapper.register_norm")
    @patch("atb_llm.utils.data.weight_wrapper.WeightWrapper.register_linear_wrapper")
    def test_register_layer_attn(self, func1, func2):
        layer = MagicMock()
        spec_list = ["kv_a_proj_with_mqa", "kv_a_layernorm", "k_b_proj", "v_b_proj", "o_proj", "pack_type"]
        layer.self_attn = MagicMock(spec=["q_proj"] + spec_list)
        attn_wrapper = AttnWrapper(norm_name='input_layernorm', wrapper_name='self_attn')
        moe_wrapper = MoeMlpWrapper(norm_name='input_layernorm', wrapper_name='self_attn')
        ds2_wrapper = Deepseekv2WeightWrapper(NPUSocInfo(), 1, attn_wrapper, moe_wrapper, 64)
        ds2_wrapper.register_layer_attn(layer, attn_wrapper, 1)
        layer.self_attn = MagicMock(spec=["q_a_proj", "q_a_layernorm", "q_b_proj"] + spec_list)
        layer.self_attn.q_a_proj.linear.weight.data = torch.empty(576, 7168, dtype=torch.float16)
        layer.self_attn.kv_a_proj_with_mqa.linear.weight.data = torch.empty(1536, 7168, dtype=torch.float16)
        ds2_wrapper.register_layer_attn(layer, attn_wrapper, 1)


if __name__ == '__main__':
    unittest.main()