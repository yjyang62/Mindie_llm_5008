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
from atb_llm import nn
from atb_llm.nn.network_manager import get_default_net
from atb_llm.nn.functional.activation import ActType
from atb_llm.nn.tensor import Tensor
from ddt import ddt, data
from tests.pythontest.atb_llm.utils import NO_BF16_SUPPORT_SOC_VERSION

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger()


@ddt
class TestSwigluWeightNoPackPass(unittest.TestCase):
    def setUp(self):
        self.gate_linear_out = Tensor("gate_linear_out")
        self.up_linear_out = Tensor("up_linear_out")

    def forward(self):
        gate_activation_out = nn.functional.activation(self.gate_linear_out, ActType.SWISH)
        out = gate_activation_out * self.up_linear_out
        return out

    # 创建Swiglu融合算子engine
    def get_swiglu_weight_no_pack_fusion_engine(self, dtype):
        out = self.forward()
        get_default_net().mark_output(out, "out")
        swiglu_weight_no_pack_fusion_engine = get_default_net().build_engine()
        logger.info(swiglu_weight_no_pack_fusion_engine)
        return swiglu_weight_no_pack_fusion_engine

    # 创建Swish + mul普通算子engine
    def get_swiglu_weight_no_pack_engine(self, dtype):
        out = self.forward()
        get_default_net().mark_output(out, "out")
        swiglu_weight_no_pack_engine = get_default_net().build_engine(del_fpass_keys=["SwigluWeightNoPackPass"])
        logger.info(swiglu_weight_no_pack_engine)
        return swiglu_weight_no_pack_engine

    @data(torch.float16, torch.bfloat16)
    def test_swiglu_weight_no_pack(self, dtype):
        soc_version = torch_npu._C._npu_get_soc_version()
        if soc_version in NO_BF16_SUPPORT_SOC_VERSION:
            self.skipTest("Atlas 300I DUO doesn't support swiglu_weight.")
        hidden_size = 128  # 设置hidden_size大小
        batch_size = 4  # 设置batch_size大小
        seq_len = 64  # 设置seq_len长度
        shape = (batch_size * seq_len, hidden_size)

        fusion_engine = self.get_swiglu_weight_no_pack_fusion_engine(dtype)
        engine = self.get_swiglu_weight_no_pack_engine(dtype)

        gate_linear_out = torch.randn(shape, dtype=dtype).npu()
        up_linear_out = torch.randn(shape, dtype=dtype).npu()

        fusion_out = torch.empty(shape, dtype=dtype).npu()
        out = torch.empty(shape, dtype=dtype).npu()

        inputs = {"gate_linear_out": gate_linear_out, "up_linear_out": up_linear_out}
        fusion_outputs = {"out": fusion_out}
        outputs = {"out": out}

        fusion_engine.forward(inputs, fusion_outputs)
        engine.forward(inputs, outputs)

        self.assertTrue(torch.allclose(fusion_outputs["out"], outputs["out"], rtol=1e-02, atol=1e-02))
        self.assertIn("ACTIVATION_SWIGLU_FORWARD", str(fusion_engine))


if __name__ == "__main__":
    unittest.main()

