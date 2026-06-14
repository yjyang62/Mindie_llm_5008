# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from collections import OrderedDict
import unittest
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from atb_llm.utils.layers.linear.linear_utils import LinearUtils
from atb_llm.utils.env import ENV
from atb_llm.utils.quantize.pack_type import TransposeType


class TestLinearUtils(unittest.TestCase):
    def setUp(self):
        # 强制清除单例，确保每个测试从 clean state 开始
        LinearUtils.soc_info = None
        ENV.auto_transpose_enable = True
        
    def tearDown(self):
        LinearUtils.soc_info = None

    def test_init_and_soc_info(self):
        """
        修复: 使用准确的 patch 路径。
        必须 patch LinearUtils 内部引用 NPUSocInfo 的那个‘位置’。
        """
        target = 'atb_llm.utils.layers.linear.linear_utils.NPUSocInfo'
        with patch(target) as mock_soc_cls:
            instance = LinearUtils()
            # 验证构造函数是否触发了单例设置
            self.assertIsNotNone(LinearUtils.soc_info)
            mock_soc_cls.assert_called_once()
            self.assertEqual(instance.trans_flag, TransposeType.TRANSPOSE)

    def test_weight_format_cast(self):
        """修复: 解决 Tensor 比较歧义"""
        # 手动注入 mock soc_info 避免依赖外部 NPU 环境
        LinearUtils.soc_info = MagicMock()
        test_tensor = torch.randn(2, 2)

        # 分支1: 不需要 NZ
        LinearUtils.soc_info.need_nz = False
        res1 = LinearUtils.weight_format_cast(test_tensor)
        self.assertTrue(torch.equal(res1, test_tensor))

        # 分支2: 需要 NZ
        LinearUtils.soc_info.need_nz = True
        with patch('torch_npu.npu_format_cast_') as mock_npu_cast:
            res2 = LinearUtils.weight_format_cast(test_tensor)
            mock_npu_cast.assert_called_once()
            # 注意：此处只需验证返回的是同一个对象即可，或者用 torch.equal
            self.assertIs(res2, test_tensor)

    def test_check_transpose_logic(self):
        """测试 check_transpose 的多分支逻辑，并验证 logger 警告"""
        utils = LinearUtils()
        utils.soc_info = MagicMock()
        
        # 分支: matmul_nd_nz 开启且 ENV 禁止自动转置 -> 触发强制修正
        utils.soc_info.need_nz = False
        utils.soc_info.matmul_nd_nz = True
        ENV.auto_transpose_enable = False
        
        with patch('atb_llm.utils.layers.linear.linear_utils.logger') as mock_log:
            res = utils.check_transpose(torch.randn(128, 128))
            self.assertEqual(res, TransposeType.NOT_TRANSPOSE)
            self.assertTrue(ENV.auto_transpose_enable)
            mock_log.warning.assert_called()

        # 分支: 维度对齐 logic (N整除256, K不整除)
        utils.soc_info.matmul_nd_nz = False
        weight_n_aligned = torch.randn(512, 10) # Shape: [512, 10] -> [-2]是512, [-1]是10
        self.assertEqual(utils.check_transpose(weight_n_aligned), TransposeType.NOT_TRANSPOSE)

    def test_set_transpose_and_op(self):
        """测试 set_transpose 是否真正执行了转置"""
        # LinearUtils 需要 self.weight 属性，通常它被子类混入
        class MockLinear(LinearUtils, nn.Module):
            def __init__(self, w):
                nn.Module.__init__(self)
                LinearUtils.__init__(self)
                self.weight = nn.Parameter(w)

        init_w = torch.randn(2, 4)
        model = MockLinear(init_w.clone())
        
        # 模拟状态切换：从 TRANSPOSE 切换到 NOT_TRANSPOSE
        model.set_transpose(TransposeType.NOT_TRANSPOSE)
        self.assertEqual(model.trans_flag, TransposeType.NOT_TRANSPOSE)
        self.assertEqual(model.weight.shape, (4, 2)) # 确认执行了 .T
        
    def test_get_weights(self):
        """验证返回的是 OrderedDict 且前缀正确"""
        class MockLinear(LinearUtils, nn.Module):
            def __init__(self):
                nn.Module.__init__(self)
                LinearUtils.__init__(self)
                self.register_buffer('test_buf', torch.ones(1))

        LinearUtils.soc_info = MagicMock(need_nz=False)
        model = MockLinear()
        weights = model.get_weights("gpt.block.0")
        
        self.assertIn("gpt.block.0.test_buf", weights)
        self.assertIsInstance(weights, OrderedDict)

if __name__ == '__main__':
    unittest.main()