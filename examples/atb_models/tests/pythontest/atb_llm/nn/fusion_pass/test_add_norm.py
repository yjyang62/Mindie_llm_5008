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
from tests.pythontest.atb_llm.utils import NO_BF16_SUPPORT_SOC_VERSION

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger()


@ddt
class TestAddRmsNormPass(unittest.TestCase):
    def setUp(self):
        self.x = Tensor("x")
        self.residual = Tensor("residual")
        self.eps = 1e-6
    
    def forward(self, x, residual):
        rmsNorm = RmsNorm("norm", self.eps)
        add_out = x + residual
        rmsNorm_out = rmsNorm(add_out)
        return rmsNorm_out, add_out

    # 创建addnorm融合算子engine
    def get_addnorm_fusion_engine(self):
        rmsNorm_out, add_out = self.forward(self.x, self.residual)
        get_default_net().mark_output(rmsNorm_out, "rmsNorm_out")
        get_default_net().mark_output(add_out, "add_out")
        addnorm_fusion_engine = get_default_net().build_engine()
        logger.info(addnorm_fusion_engine)
        return addnorm_fusion_engine

    # 创建add和norm普通算子engine
    def get_addnorm_engine(self):
        rmsNorm_out, add_out = self.forward(self.x, self.residual)
        get_default_net().mark_output(rmsNorm_out, "rmsNorm_out")
        get_default_net().mark_output(add_out, "add_out")
        addnorm_engine = get_default_net().build_engine(del_fpass_keys=["AddRmsNormPass"])
        logger.info(addnorm_engine)
        return addnorm_engine

    @data(torch.float16, torch.bfloat16)
    def test_add_rms_norm(self, dtype):
        if torch_npu._C._npu_get_soc_version() in NO_BF16_SUPPORT_SOC_VERSION and dtype == torch.bfloat16:
            self.skipTest("Atlas 300I DUO doesn't support bfloat16.")
        hidden_size = 128  # 设置hidden_size大小
        batch_size = 4  # 设置batch_size大小
        seq_len = 64  # 设置seq_len长度
        shape = (batch_size * seq_len, hidden_size)

        addnorm_fusion_engine = self.get_addnorm_fusion_engine()
        addnorm_engine = self.get_addnorm_engine()

        self.assertIn("AddRmsNorm", str(addnorm_fusion_engine))
        
        x = torch.randn(shape, dtype=dtype).npu()
        residual = torch.randn(shape, dtype=dtype).npu()
        
        addnorm_fusion_yOut = torch.empty(shape, dtype=dtype).npu()
        addnorm_fusion_rstdOut = torch.empty(shape, dtype=dtype).npu()
        
        addnorm_yOut = torch.empty(shape, dtype=dtype).npu()
        addnorm_rstdOut = torch.empty(shape, dtype=dtype).npu()

        weights = {}
        weights["norm.weight"] = torch.ones(hidden_size, dtype=dtype).npu()
        
        inputs = {"x": x, "residual": residual}
        addnorm_fusion_outputs = {"add_out": addnorm_fusion_yOut, "rmsNorm_out": addnorm_fusion_rstdOut}
        addnorm_outputs = {"add_out": addnorm_yOut, "rmsNorm_out": addnorm_rstdOut}        
        
        addnorm_fusion_engine.set_weights(weights)
        addnorm_fusion_engine.forward(inputs, addnorm_fusion_outputs)
        
        addnorm_engine.set_weights(weights)
        addnorm_engine.forward(inputs, addnorm_outputs)
        
        self.assertTrue(torch.allclose(addnorm_fusion_outputs["add_out"], addnorm_outputs["add_out"], rtol=1e-03, atol=1e-03))


if __name__ == "__main__":
    unittest.main()
        