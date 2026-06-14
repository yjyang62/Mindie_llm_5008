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
import logging

from ddt import ddt, data
import torch
import torch_npu

from atb_llm.nn.network_manager import get_default_net
from atb_llm.nn.modules import RmsNorm
from atb_llm.nn.tensor import Tensor
from atb_llm.nn.quantized.quantization import quantize_per_channel
from tests.pythontest.atb_llm.utils import NO_BF16_SUPPORT_SOC_VERSION

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger()


@ddt
class TestRmsNormQuantPertensorPass(unittest.TestCase):
    def setUp(self):
        self.x = Tensor("x")
        self.bias = Tensor("bias")
        self.scale = Tensor("scale")
        self.offset = Tensor("offset")
        self.eps = 1e-6
    
    def forward(self, x, bias, scale, offset):
        rmsNorm = RmsNorm("norm", self.eps)
        rmsNorm_out = rmsNorm(x)
        add_out = rmsNorm_out + bias
        norm_quant_out = quantize_per_channel(add_out, scale, offset)
        return norm_quant_out

    # 创建normquant融合算子engine
    def get_norm_quant_fusion_engine(self):
        norm_quant_out = self.forward(self.x, self.bias, self.scale, self.offset)
        get_default_net().mark_output(norm_quant_out, "norm_quant_out")
        norm_quant_fusion_engine = get_default_net().build_engine()
        logger.info(norm_quant_fusion_engine)
        return norm_quant_fusion_engine

    # 创建norm和quant普通算子engine
    def get_norm_quant_engine(self):
        norm_quant_out = self.forward(self.x, self.bias, self.scale, self.offset)
        get_default_net().mark_output(norm_quant_out, "norm_quant_out")
        norm_quant_engine = get_default_net().build_engine(del_fpass_keys=["All"])
        logger.info(norm_quant_engine)
        return norm_quant_engine

    @data(torch.float16, torch.bfloat16)
    def test_norm_quant(self, dtype):
        if torch_npu._C._npu_get_soc_version() in NO_BF16_SUPPORT_SOC_VERSION and dtype == torch.bfloat16:
            self.skipTest("Atlas 300I DUO doesn't support bfloat16.")
        hidden_size = 32  # 设置hidden_size大小
        batch_size = 4  # 设置batch_size大小
        seq_len = 8  # 设置seq_len长度
        shape = (batch_size * seq_len, hidden_size)
        
        scale = torch.randn(1, dtype=dtype).npu()
        offset = torch.zeros(1, dtype=torch.int8).npu()        
        
        norm_quant_engine = self.get_norm_quant_engine()
        norm_quant_fusion_engine = self.get_norm_quant_fusion_engine()

        self.assertIn("quantType", str(norm_quant_fusion_engine))
        
        x = torch.randn(shape, dtype=dtype).npu()

        bias = torch.randn(hidden_size, dtype=dtype).npu()
        
        norm_quant_yOut = torch.empty(shape, dtype=torch.int8).npu()

        norm_quant_fusion_yOut = torch.empty(shape, dtype=torch.int8).npu()

        weights = {}
        weights["norm.weight"] = torch.ones(hidden_size, dtype=dtype).npu()
        
        inputs = {"x": x, "bias": bias, "scale": scale, "offset": offset}
        
        norm_quant_outputs = {"norm_quant_out": norm_quant_yOut}        
        norm_quant_fusion_outputs = {"norm_quant_out": norm_quant_fusion_yOut}
        
        norm_quant_engine.set_weights(weights)
        norm_quant_engine.forward(inputs, norm_quant_outputs)        
        
        norm_quant_fusion_engine.set_weights(weights)
        norm_quant_fusion_engine.forward(inputs, norm_quant_fusion_outputs)
        
        self.assertTrue(torch.allclose(norm_quant_outputs["norm_quant_out"], norm_quant_fusion_outputs["norm_quant_out"], rtol=1, atol=1))


if __name__ == "__main__":
    unittest.main()