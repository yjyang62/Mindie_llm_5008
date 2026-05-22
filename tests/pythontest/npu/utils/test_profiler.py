# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import sys
import builtins
import unittest
from unittest.mock import Mock, patch
import torch

ACTION_NAME = "test"
FLAG = True
UNFLAG = False
REAL_IMPORT = builtins.__import__


def mock_import_module(name, *args, **kwargs):
    if name == "ms_service_profiler":
        raise ImportError("Mocked ImportERROR")
    else:
        return REAL_IMPORT(name, *args, **kwargs)


# 测试正常导入的情况
class TestProfilerFunctions(unittest.TestCase):
    def __init__(self, methodName="runTest"):
        super().__init__(methodName)
        self.torch_npu = None

    def clean(self):
        if "mindie_llm.utils.prof.profiler" in sys.modules:
            del sys.modules["mindie_llm.utils.prof.profiler"]
        if "ms_service_profiler" in sys.modules:
            del sys.modules["ms_service_profiler"]
        if "ms_service_profiler.mstx" in sys.modules:
            del sys.modules["ms_service_profiler.mstx"]
        if "mindie_llm.utils.log.logging" in sys.modules:
            del sys.modules["mindie_llm.utils.log.logging"]
        if "torch_npu" in sys.modules:
            del sys.modules["torch_npu"]
            del torch.npu

    def setUp(self):
        self.torch_npu = sys.modules.get("torch_npu", None)
        self.clean()
        torch.npu = Mock()
        sys.modules["torch_npu"] = Mock()

    def tearDown(self):
        self.clean()
        if self.torch_npu is not None:
            sys.modules["torch_npu"] = self.torch_npu
            self.torch_npu = None

    def test_span_start_with_error_import(self):
        with patch('builtins.__import__', side_effect=mock_import_module):
            from mindie_llm.utils.prof.profiler import span_start

        try:
            span_start(ACTION_NAME, FLAG)
        except Exception as e:
            self.fail(f"Function raised an exception when it shouldn't have: {e}")

    def test_span_end_with_error_import(self):
        with patch('builtins.__import__', side_effect=mock_import_module):
            from mindie_llm.utils.prof.profiler import span_start, span_end

        try:
            prof = span_start(ACTION_NAME)
            span_end(prof, FLAG)
        except Exception as e:
            self.fail(f"Function raised an exception when it shouldn't have: {e}")

    def test_span_req_with_error_import(self):
        with patch('builtins.__import__', side_effect=mock_import_module):
            from mindie_llm.utils.prof.profiler import span_req

        try:
            span_req(ACTION_NAME, ["rid_1234"])
        except Exception as e:
            self.fail(f"Function raised an exception when it shouldn't have: {e}")

    def test_profiler_with_error_import(self):
        with patch('builtins.__import__', side_effect=mock_import_module):
            from mindie_llm.utils.prof.profiler import Profiler

        try:
            prof = Profiler()
            prof.span_start()
            prof.span_end()
        except Exception as e:
            self.fail(f"Function raised an exception when it shouldn't have: {e}")

    def test_count_block_with_success_import(self):
        from mindie_llm.utils.prof.profiler import count_block
        from mindie_llm.utils.prof.profiler import np

        try:
            count_block(np.zeros((2, 3, 4), dtype=np.int32))
            count_block(np.zeros((10, 2), dtype=np.int32))
        except Exception as e:
            self.fail(f"Function raised an exception when it shouldn't have: {e}")

    def test_tensor_attr_with_error_import(self):
        """测试在导入失败的情况下 tensor_attr 是否可用"""
        with patch('builtins.__import__', side_effect=mock_import_module):
            from mindie_llm.utils.prof.profiler import tensor_attr

        # 创建测试张量
        test_tensor = torch.Tensor([1, 2, 3, 4, 5])

        try:
            # 测试应该正常执行而不报错
            _ = tensor_attr(test_tensor, statistics=True)
        except Exception as e:
            self.fail(f"tensor_attr raised an exception when it shouldn't have: {e}")

    def test_tensor_attr_statistics_mode(self):
        sys.modules["ms_service_profiler"] = Mock()
        sys.modules["ms_service_profiler.mstx"] = Mock()
        sys.modules["mindie_llm.utils.log.logging"] = Mock()
        sys.modules["mindie_llm.utils.tensor"] = Mock()
        """测试 tensor_attr 在 statistics=True 时的行为"""
        from mindie_llm.utils.prof.profiler import tensor_attr

        # 测试 1D 张量
        test_tensor_1d = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = tensor_attr(test_tensor_1d, statistics=True)

        # 验证返回的字典结构
        self.assertIsInstance(result, dict)
        self.assertIn("min", result)
        self.assertIn("max", result)
        self.assertIn("mean", result)
        self.assertIn("first_10", result)
        self.assertIn("shape", result)

        # 验证统计值
        self.assertEqual(result["min"], 1.0)
        self.assertEqual(result["max"], 5.0)
        self.assertEqual(result["mean"], 3.0)
        self.assertEqual(result["first_10"], [1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(result["shape"], [5])

    def test_tensor_attr_no_statistics_mode(self):
        sys.modules["ms_service_profiler"] = Mock()
        sys.modules["ms_service_profiler.mstx"] = Mock()
        sys.modules["mindie_llm.utils.log.logging"] = Mock()
        sys.modules["mindie_llm.utils.tensor"] = Mock()
        """测试 tensor_attr 在 statistics=False 时的行为"""
        from mindie_llm.utils.prof.profiler import tensor_attr

        # 测试 1D 张量
        test_tensor_1d = torch.Tensor([1, 2, 3])
        result = tensor_attr(test_tensor_1d, statistics=False)

        # 应该返回列表形式
        self.assertEqual(result, [1, 2, 3])

    def test_tensor_attr_2d_tensor(self):
        sys.modules["ms_service_profiler"] = Mock()
        sys.modules["ms_service_profiler.mstx"] = Mock()
        sys.modules["mindie_llm.utils.log.logging"] = Mock()
        sys.modules["mindie_llm.utils.tensor"] = Mock()
        """测试 2D 张量的处理"""
        from mindie_llm.utils.prof.profiler import tensor_attr

        test_tensor_2d = torch.Tensor([[1, 2, 3], [4, 5, 6]])
        result = tensor_attr(test_tensor_2d, statistics=True)

        self.assertEqual(result["min"], 1)
        self.assertEqual(result["max"], 6)
        self.assertEqual(result["mean"], 3.5)
        self.assertEqual(result["first_10"], [1, 2, 3, 4, 5, 6])  # 展平后的前6个元素
        self.assertEqual(result["shape"], [2, 3])

    def test_tensor_attr_large_tensor(self):
        sys.modules["ms_service_profiler"] = Mock()
        sys.modules["ms_service_profiler.mstx"] = Mock()
        sys.modules["mindie_llm.utils.log.logging"] = Mock()
        sys.modules["mindie_llm.utils.tensor"] = Mock()
        """测试大型张量（元素超过10个）"""
        from mindie_llm.utils.prof.profiler import tensor_attr

        test_tensor_large = torch.Tensor(list(range(20)))
        result = tensor_attr(test_tensor_large, statistics=True)

        # 只取前10个元素
        self.assertEqual(result["first_10"], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertEqual(result["shape"], [20])

    def test_tensor_attr_tuple_input(self):
        """测试输入为元组的情况"""
        with patch('builtins.__import__', side_effect=mock_import_module):
            from mindie_llm.utils.prof.profiler import tensor_attr

        test_tuple = (1, 2, 3)
        result = tensor_attr(test_tuple, statistics=True)

        # 应该直接返回元组，不进行处理
        self.assertEqual(result, (1, 2, 3))

    def test_tensor_attr_empty_tensor(self):
        sys.modules["ms_service_profiler"] = Mock()
        sys.modules["ms_service_profiler.mstx"] = Mock()
        sys.modules["mindie_llm.utils.log.logging"] = Mock()
        sys.modules["mindie_llm.utils.tensor"] = Mock()
        """测试空张量"""
        from mindie_llm.utils.prof.profiler import tensor_attr

        empty_tensor = torch.Tensor([])
        result = tensor_attr(empty_tensor, statistics=True)

        # 验证空张量的处理
        self.assertEqual(result, {})  # 空数组的min是nan

    def test_tensor_attr_complex_tensor(self):
        sys.modules["ms_service_profiler"] = Mock()
        sys.modules["ms_service_profiler.mstx"] = Mock()
        sys.modules["mindie_llm.utils.log.logging"] = Mock()
        sys.modules["mindie_llm.utils.tensor"] = Mock()
        """测试复杂张量（浮点数、负数等）"""
        from mindie_llm.utils.prof.profiler import tensor_attr

        complex_tensor = torch.Tensor([-1.5, 0.0, 3.14, -2.7, 10.0])
        result = tensor_attr(complex_tensor, statistics=True)

        self.assertAlmostEqual(result["min"], -2.7)
        self.assertAlmostEqual(result["max"], 10.0)
        self.assertAlmostEqual(result["mean"], 1.788, places=2)
        self.assertEqual(result["first_10"], torch.Tensor([-1.5, 0.0, 3.14, -2.7, 10.0]).tolist())

    def test_tensor_attr_with_import_success(self):
        """测试在成功导入的情况下 tensor_attr 的行为"""
        sys.modules["ms_service_profiler"] = Mock()
        sys.modules["ms_service_profiler.mstx"] = Mock()
        sys.modules["mindie_llm.utils.log.logging"] = Mock()
        sys.modules["mindie_llm.utils.tensor"] = Mock()

        from mindie_llm.utils.prof.profiler import tensor_attr

        test_tensor = torch.Tensor([[1, 2], [3, 4]])
        result = tensor_attr(test_tensor, statistics=True)

        # 验证功能正常
        self.assertIsInstance(result, dict)
        self.assertEqual(result["min"], 1)
        self.assertEqual(result["max"], 4)
        self.assertEqual(result["mean"], 2.5)

    def mock_profiler(self):
        ms_service_profiler = Mock()
        ms_service_profiler.Profiler = lambda _: self.mock_prof()

        mstx = Mock()
        mstx.service_profiler = Mock()
        mstx.service_profiler.is_enable = lambda _: True
        return ms_service_profiler, mstx

    def mock_prof(self):
        prof = Mock()
        prof.span_start = Mock()
        prof.span_end = Mock()
        prof._enable = True
        return prof

    def test_span_start_when_import_success_no_error(self):
        ms_service_profiler, mstx = self.mock_profiler()
        sys.modules["ms_service_profiler"] = ms_service_profiler
        sys.modules["ms_service_profiler.mstx"] = mstx
        sys.modules["mindie_llm.utils.log.logging"] = Mock()
        from mindie_llm.utils.prof.profiler import span_start

        try:
            span_start(ACTION_NAME, FLAG)
        except Exception as e:
            self.fail(f"Function raised an exception when it shouldn't have: {e}")

    def test_span_end_when_import_success_no_error(self):
        ms_service_profiler, mstx = self.mock_profiler()
        sys.modules["ms_service_profiler"] = ms_service_profiler
        sys.modules["ms_service_profiler.mstx"] = mstx
        sys.modules["mindie_llm.utils.log.logging"] = Mock()
        from mindie_llm.utils.prof.profiler import span_attr, span_end

        try:
            span_attr(self.mock_prof(), "key", "value")
            span_end(self.mock_prof(), FLAG)
        except Exception as e:
            self.fail(f"Function raised an exception when it shouldn't have: {e}")

    def test_span_req_when_import_success(self):
        sys.modules["ms_service_profiler"] = Mock()
        sys.modules["ms_service_profiler.mstx"] = Mock()
        sys.modules["mindie_llm.utils.log.logging"] = Mock()
        from mindie_llm.utils.prof.profiler import span_req

        try:
            span_req(0, [])
        except Exception as e:
            self.fail(f"Function raised an exception when it shouldn't have: {e}")

    def test_span_req_when_import_success_no_error(self):
        sys.modules["ms_service_profiler"] = Mock()
        sys.modules["ms_service_profiler.mstx"] = Mock()
        sys.modules["mindie_llm.utils.log.logging"] = Mock()
        from mindie_llm.utils.prof.profiler import span_req

        try:
            span_req(ACTION_NAME, [1])
        except Exception as e:
            self.fail(f"Function raised an exception when it shouldn't have: {e}")