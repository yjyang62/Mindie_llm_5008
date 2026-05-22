#!/usr/bin/env python
# coding=utf-8
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
from atb_llm.nn.functional import activation, ActType
from atb_llm.nn.quantized import quantize_per_token
from atb_llm.nn.tensor import Tensor
from ddt import ddt, data
from tests.pythontest.atb_llm.utils import NO_BF16_SUPPORT_SOC_VERSION

torch_npu.npu.set_device(0)

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger()


@ddt
class TestSwigluWeightNoPackQuantPerTokenPass(unittest.TestCase):
    def setUp(self):
        self.gate_linear_out = Tensor("gate_linear_out")
        self.up_linear_out = Tensor("up_linear_out")

    def forward(self,):
        gate_activation_out = activation(self.gate_linear_out, ActType.SWISH)
        mul_out = gate_activation_out * self.up_linear_out
        quant_out, scale_out = quantize_per_token(mul_out)
        return quant_out, scale_out

    # 创建SwigluQuant融合算子engine
    def get_swiglu_fusion_engine(self,):
        quant_out, scale_out = self.forward()
        get_default_net().mark_output(scale_out, "scale_out")
        get_default_net().mark_output(quant_out, "quant_out")
        swiglu_fusion_engine = get_default_net().build_engine(del_fpass_keys=["SwigluWeightNoPackPass"])
        logger.info(swiglu_fusion_engine)
        return swiglu_fusion_engine

    # 创建swiglu和quant普通算子engine
    def get_swiglu_engine(self,):
        quant_out, scale_out = self.forward()
        get_default_net().mark_output(scale_out, "scale_out")
        get_default_net().mark_output(quant_out, "quant_out")
        swiglu_engine = get_default_net().build_engine(del_fpass_keys=["ALL"])
        logger.info(swiglu_engine)
        return swiglu_engine

    @data(torch.float16, torch.bfloat16)
    def test_swiglu_quant(self, dtype):
        if torch_npu._C._npu_get_soc_version() in NO_BF16_SUPPORT_SOC_VERSION and dtype == torch.bfloat16:
            self.skipTest("Atlas 300I DUO doesn't support bfloat16.")
        hidden_size = 128  # 设置hidden_size大小
        batch_size = 4  # 设置batch_size大小
        seq_len = 64  # 设置seq_len长度
        shape = (batch_size * seq_len, hidden_size)
        scaleOut_shape = (batch_size * seq_len, 1)


        swiglu_fusion_engine = self.get_swiglu_fusion_engine()
        swiglu_engine = self.get_swiglu_engine()

        gate_linear_out = torch.randn(shape, dtype=dtype).npu()
        up_linear_out = torch.randn(shape, dtype=dtype).npu()

        scale_fusion_out = torch.empty(scaleOut_shape, dtype=torch.float32).npu()
        quant_fusion_yOut = torch.empty(shape, dtype=torch.int8).npu()

        scale_out = torch.empty(scaleOut_shape, dtype=torch.float32).npu()
        quant_yOut = torch.empty(shape, dtype=torch.int8).npu()

        inputs = {"gate_linear_out": gate_linear_out, "up_linear_out":up_linear_out}
        swiglu_fusion_outputs = {"scale_out": scale_fusion_out, "quant_out": quant_fusion_yOut}
        swiglu_outputs = {"scale_out": scale_out, "quant_out": quant_yOut}

        swiglu_fusion_engine.forward(inputs, swiglu_fusion_outputs)
        swiglu_engine.forward(inputs, swiglu_outputs)

        self.assertTrue(
            torch.allclose(swiglu_fusion_outputs["quant_out"], swiglu_outputs["quant_out"], rtol=1, atol=1))
        self.assertEqual(
            swiglu_fusion_engine.engine.sub_ops[1].op_name, "DequantSwigluQuant"
        )


if __name__ == "__main__":
    unittest.main()
