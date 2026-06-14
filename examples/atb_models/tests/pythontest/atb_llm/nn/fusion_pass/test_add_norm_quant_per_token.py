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
import torch
import torch_npu
from atb_llm.nn.network_manager import get_default_net
from atb_llm.nn.modules import RmsNorm
from atb_llm.nn.tensor import Tensor
from atb_llm.nn.quantized.quantization import quantize_per_token
from ddt import ddt, data

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger()


@ddt
class TestAddRmsNormQuantPerTokenPass(unittest.TestCase):
    def setUp(self):
        self.x = Tensor("x")
        self.residual = Tensor("residual")
        self.scale = Tensor("scale")
        self.offset = Tensor("offset")
        self.eps = 1e-6
    
    def forward(self, x, residual):
        rmsNorm = RmsNorm("norm", self.eps)
        add_out = x + residual
        rmsNorm_out = rmsNorm(add_out)
        add_norm_quant_out, scale = quantize_per_token(rmsNorm_out)
        return add_norm_quant_out, add_out, scale

    # 创建addnorm融合算子engine
    def get_addnorm_quant_fusion_engine(self):
        add_norm_quant_out, add_out, scale = self.forward(self.x, self.residual)
        get_default_net().mark_output(add_norm_quant_out, "add_norm_quant_out")
        get_default_net().mark_output(add_out, "add_out")
        get_default_net().mark_output(scale, "scale")
        addnorm_fusion_engine = get_default_net().build_engine()
        logger.info(addnorm_fusion_engine)
        return addnorm_fusion_engine

    # 创建add和norm普通算子engine
    def get_addnorm_quant_engine(self):
        add_norm_quant_out, add_out, scale = self.forward(self.x, self.residual)
        get_default_net().mark_output(add_norm_quant_out, "add_norm_quant_out")
        get_default_net().mark_output(add_out, "add_out")
        get_default_net().mark_output(scale, "scale")
        addnorm_engine = get_default_net().build_engine(del_fpass_keys=["All"])
        logger.info(addnorm_engine)
        return addnorm_engine

    @data(torch.float16, torch.bfloat16)
    def test_add_rms_norm_quant(self, dtype):
        if torch_npu._C._npu_get_soc_version() in NO_BF16_SUPPORT_SOC_VERSION and dtype == torch.bfloat16:
            self.skipTest("Atlas 300I DUO doesn't support bfloat16.")
        hidden_size = 32  # 设置hidden_size大小
        batch_size = 4  # 设置batch_size大小
        seq_len = 8  # 设置seq_len长度
        shape = (batch_size * seq_len, hidden_size)
        
        addnorm_engine = self.get_addnorm_quant_engine()
        addnormquant_fusion_engine = self.get_addnorm_quant_fusion_engine()

        self.assertIn("AddRmsNormDynamicQuant", str(addnormquant_fusion_engine))

        x = torch.normal(mean=0, std=0.001, size=shape, dtype=dtype).npu()
        residual = torch.normal(mean=0, std=0.001, size=shape, dtype=dtype).npu()
        scale = torch.empty(hidden_size, dtype=torch.float).npu() 
        
        addnormquant_addOut = torch.empty(shape, dtype=dtype).npu()        
        addnormquant_yOut = torch.empty(shape, dtype=torch.int8).npu()

        addnormquant_fusion_addOut = torch.empty(shape, dtype=dtype).npu()
        addnormquant_fusion_yOut = torch.empty(shape, dtype=torch.int8).npu()

        weights = {}
        weights["norm.weight"] = torch.ones(hidden_size, dtype=dtype).npu()
        
        inputs = {"x": x, "residual": residual}
        inputs_fusion = {"x": x, "residual": residual}

        addnormquant_outputs = {"add_out": addnormquant_addOut, "add_norm_quant_out": addnormquant_yOut, "scale": scale}        
        addnormquant_fusion_outputs = {"add_out": addnormquant_fusion_addOut, "add_norm_quant_out": addnormquant_fusion_yOut, "scale": scale}

        addnorm_engine.set_weights(weights)
        addnorm_engine.forward(inputs, addnormquant_outputs)

        addnormquant_fusion_engine.set_weights(weights)
        addnormquant_fusion_engine.forward(inputs_fusion, addnormquant_fusion_outputs)
        
        self.assertTrue(torch.allclose(addnormquant_outputs["add_out"], addnormquant_fusion_outputs["add_out"], rtol=1e-03, atol=1e-03))
        self.assertTrue(torch.allclose(addnormquant_outputs["add_norm_quant_out"], addnormquant_fusion_outputs["add_norm_quant_out"], rtol=1, atol=1))


if __name__ == "__main__":
    unittest.main()