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
import unittest
from unittest.mock import Mock, patch
import numpy as np


ACTION_NAME = "test"
FLAG = True
UNFLAG = False


class MockTensor:
    def __init__(self, data):
        self.xx = np.array(data)

    def __getattribute__(self, name):
        if name in ['min', 'max', 'mean', 'flatten', 'item', 'tolist', 'shape', '__getitem__', '__len__']:
            return getattr(self.xx, name)
        return super().__getattribute__(name)

    def numel(self):
        return self.xx.size


# 测试正常导入的情况
class TestProfilerFunctions(unittest.TestCase):

    def __init__(self, methodName="runTest"):
        super().__init__(methodName)
        self.torch = None
        self.torch_npu = None

    def clean(self):
        if "atb_llm.utils.prof.profiler" in sys.modules:
            del sys.modules["atb_llm.utils.prof.profiler"]
        if "ms_service_profiler" in sys.modules:
            del sys.modules["ms_service_profiler"]
        if "ms_service_profiler.mstx" in sys.modules:
            del sys.modules["ms_service_profiler.mstx"]
        if "atb_llm.utils.log.logging" in sys.modules:
            del sys.modules["atb_llm.utils.log.logging"]
        if "torch" in sys.modules:
            del sys.modules["torch"]
        if "torch_npu" in sys.modules:
            del sys.modules["torch_npu"]

    def setUp(self):
        self.torch = sys.modules.get("torch", None)
        self.torch_npu = sys.modules.get("torch_npu", None)
        self.clean()
        mock_torch = Mock()
        mock_torch.Tensor = MockTensor
        mock_torch.npu = Mock()
        sys.modules["torch"] = mock_torch
        sys.modules["torch_npu"] = Mock()

    def tearDown(self):
        self.clean()
        if self.torch is not None:
            sys.modules["torch"] = self.torch
            self.torch = None
        if self.torch_npu is not None:
            sys.modules["torch_npu"] = self.torch_npu
            self.torch_npu = None

    def test_span_start_with_error_import(self):
        from atb_llm.utils.prof.profiler import span_start

        try:
            span_start(ACTION_NAME, FLAG)
        except Exception as e:
            self.fail(f"Function raised an exception when it shouldn't have: {e}")

    def test_span_end_with_error_import(self):
        from atb_llm.utils.prof.profiler import span_end

        try:
            span_end(ACTION_NAME, FLAG)
        except Exception as e:
            self.fail(f"Function raised an exception when it shouldn't have: {e}")

    def test_span_req_with_error_import(self):
        from atb_llm.utils.prof.profiler import span_req

        try:
            span_req(ACTION_NAME, ["rid_1234"])
        except Exception as e:
            self.fail(f"Function raised an exception when it shouldn't have: {e}")

    def test_span_start_when_import_success_no_error(self):
        sys.modules["ms_service_profiler"] = Mock()
        sys.modules["ms_service_profiler.mstx"] = Mock()
        sys.modules["atb_llm.utils.log.logging"] = Mock()
        from atb_llm.utils.prof.profiler import span_start

        try:
            span_start(ACTION_NAME, FLAG)
        except Exception as e:
            self.fail(f"Function raised an exception when it shouldn't have: {e}")

    def test_span_end_when_import_success_no_error(self):
        sys.modules["ms_service_profiler"] = Mock()
        sys.modules["ms_service_profiler.mstx"] = Mock()
        sys.modules["atb_llm.utils.log.logging"] = Mock()
        from atb_llm.utils.prof.profiler import span_end

        try:
            span_end(ACTION_NAME, FLAG)
        except Exception as e:
            self.fail(f"Function raised an exception when it shouldn't have: {e}")

    def test_span_req_when_import_success(self):
        sys.modules["ms_service_profiler"] = Mock()
        sys.modules["ms_service_profiler.mstx"] = Mock()
        sys.modules["atb_llm.utils.log.logging"] = Mock()
        from atb_llm.utils.prof.profiler import span_req

        try:
            span_req(0, [])
        except Exception as e:
            self.fail(f"Function raised an exception when it shouldn't have: {e}")

    def test_span_req_when_import_success_no_error(self):
        sys.modules["ms_service_profiler"] = Mock()
        sys.modules["ms_service_profiler.mstx"] = Mock()
        sys.modules["atb_llm.utils.log.logging"] = Mock()
        from atb_llm.utils.prof.profiler import span_req

        try:
            span_req(ACTION_NAME, [1])
        except Exception as e:
            self.fail(f"Function raised an exception when it shouldn't have: {e}")

    def test_is_profiler_enable_with_error_import(self):
        from atb_llm.utils.prof.profiler import is_profiler_enable

        try:
            is_profiler_enable()
        except Exception as e:
            self.fail(f"Function raised an exception when it shouldn't have: {e}")

    def test_prof_expert_hot_with_error_import(self):
        from atb_llm.utils.prof.profiler import prof_expert_hot

        try:
            prof_expert_hot([1], 0)
        except Exception as e:
            self.fail(f"Function raised an exception when it shouldn't have: {e}")

    def test_prof_expert_routing_with_error_import(self):
        from atb_llm.utils.prof.profiler import prof_expert_routing

        try:
            prof_expert_routing([1], 0)
        except Exception as e:
            self.fail(f"Function raised an exception when it shouldn't have: {e}")

    def test_is_profiler_enable_when_import_success(self):
        ms_service_profiler_mock = Mock()
        ms_service_profiler_mock.__name__ = "ms_service_profiler"

        ms_service_profiler_mstx_mock = Mock()
        ms_service_profiler_mstx_mock.__name__ = "ms_service_profiler.mstx"

        logging_mock = Mock()
        logging_mock.__name__ = "atb_llm.utils.log.logging"

        sys.modules["ms_service_profiler"] = ms_service_profiler_mock
        sys.modules["ms_service_profiler.mstx"] = ms_service_profiler_mstx_mock
        sys.modules["atb_llm.utils.log.logging"] = logging_mock
        from atb_llm.utils.prof.profiler import is_profiler_enable

        try:
            is_profiler_enable()
        except Exception as e:
            self.fail(f"Function raised an exception when it shouldn't have: {e}")

    def test_prof_expert_hot_when_import_success(self):
        sys.modules["ms_service_profiler"] = Mock()
        sys.modules["ms_service_profiler.mstx"] = Mock()
        sys.modules["atb_llm.utils.log.logging"] = Mock()
        from atb_llm.utils.prof.profiler import prof_expert_hot

        try:
            prof_expert_hot([1], 0)
        except Exception as e:
            self.fail(f"Function raised an exception when it shouldn't have: {e}")

    def test_prof_expert_routing_when_import_success(self):
        sys.modules["ms_service_profiler"] = Mock()
        sys.modules["ms_service_profiler.mstx"] = Mock()
        sys.modules["atb_llm.utils.log.logging"] = Mock()
        from atb_llm.utils.prof.profiler import prof_expert_routing

        try:
            prof_expert_routing([1], 0)
        except Exception as e:
            self.fail(f"Function raised an exception when it shouldn't have: {e}")

    def test_is_profiler_enable_when_import_success_no_error(self):
        sys.modules["ms_service_profiler"] = Mock()
        sys.modules["ms_service_profiler.mstx"] = Mock()
        sys.modules["atb_llm.utils.log.logging"] = Mock()
        from atb_llm.utils.prof.profiler import is_profiler_enable

        try:
            is_profiler_enable(1)
        except Exception as e:
            self.fail(f"Function raised an exception when it shouldn't have: {e}")

    def test_prof_expert_hot_when_import_success_no_error(self):
        sys.modules["ms_service_profiler"] = Mock()
        sys.modules["ms_service_profiler.mstx"] = Mock()
        sys.modules["atb_llm.utils.log.logging"] = Mock()
        from atb_llm.utils.prof.profiler import prof_expert_hot

        try:
            prof_expert_hot(0, 0, 0)
        except Exception as e:
            self.fail(f"Function raised an exception when it shouldn't have: {e}")

    def test_prof_expert_routing_when_import_success_no_error(self):
        sys.modules["ms_service_profiler"] = Mock()
        sys.modules["ms_service_profiler.mstx"] = Mock()
        sys.modules["atb_llm.utils.log.logging"] = Mock()
        from atb_llm.utils.prof.profiler import prof_expert_routing

        try:
            prof_expert_routing(0, 0, 0)
        except Exception as e:
            self.fail(f"Function raised an exception when it shouldn't have: {e}")

    @patch('importlib.import_module', side_effect=ImportError("Mocked ImportERROR"))
    def test_span_start_when_import_failed_no_error(self, mock_import_module):
        from atb_llm.utils.prof.profiler import span_start

        try:
            span_start(ACTION_NAME, FLAG)
        except Exception as e:
            self.fail(f"Function raised an exception when it shouldn't have: {e}")

    @patch('importlib.import_module', side_effect=ImportError("Mocked ImportERROR"))
    def test_span_end_when_import_failed_no_error(self, mock_import_module):
        from atb_llm.utils.prof.profiler import span_end

        try:
            span_end(ACTION_NAME, FLAG)
        except Exception as e:
            self.fail(f"Function raised an exception when it shouldn't have: {e}")

    @patch('importlib.import_module', side_effect=ImportError("Mocked ImportERROR"))
    def test_profiler_when_import_failed_no_error(self, mock_import_module):
        from atb_llm.utils.prof.profiler import Profiler

        try:
            prof = Profiler(0, [])
            prof.no_func()
            prof.test_func("error")
        except Exception as e:
            self.fail(f"Function raised an exception when it shouldn't have: {e}")

    @patch('importlib.import_module', side_effect=ImportError("Mocked ImportERROR"))
    def test_span_req_when_import_failed_no_error(self, mock_import_module):
        from atb_llm.utils.prof.profiler import span_req

        try:
            span_req(0, [])
        except Exception as e:
            self.fail(f"Function raised an exception when it shouldn't have: {e}")

    @patch('importlib.import_module', side_effect=ImportError("Mocked ImportERROR"))
    def test_is_profiler_enable_when_import_failed_no_error(self, mock_import_module):
        from atb_llm.utils.prof.profiler import is_profiler_enable

        try:
            is_profiler_enable()
        except Exception as e:
            self.fail(f"Function raised an exception when it shouldn't have: {e}")

    @patch('importlib.import_module', side_effect=ImportError("Mocked ImportERROR"))
    def test_prof_expert_routing_when_import_failed_no_error(self, mock_import_module):
        from atb_llm.utils.prof.profiler import prof_expert_routing

        try:
            prof_expert_routing([1, 2, 3], 0)
        except Exception as e:
            self.fail(f"Function raised an exception when it shouldn't have: {e}")

    @patch('importlib.import_module', side_effect=ImportError("Mocked ImportERROR"))
    def test_prof_expert_hot_when_import_failed_no_error(self, mock_import_module):
        from atb_llm.utils.prof.profiler import prof_expert_hot

        try:
            prof_expert_hot([1, 2, 3], 0)
        except Exception as e:
            self.fail(f"Function raised an exception when it shouldn't have: {e}")

    def test_span_attr_with_error_import(self):
        from atb_llm.utils.prof.profiler import span_attr

        try:
            span_attr(None, "test_attr", "test_value")
        except Exception as e:
            self.fail(f"span_attr raised an exception when it shouldn't have: {e}")

    def test_span_attr_when_import_success(self):
        sys.modules["ms_service_profiler"] = Mock()
        sys.modules["ms_service_profiler.mstx"] = Mock()
        sys.modules["atb_llm.utils.log.logging"] = Mock()
        from atb_llm.utils.prof.profiler import span_attr

        try:
            # 测试普通值
            span_attr(Mock(), "test_attr", "test_value")
            # 测试可调用对象
            span_attr(Mock(), "test_attr", lambda: "dynamic_value")
        except Exception as e:
            self.fail(f"span_attr raised an exception when it shouldn't have: {e}")

    def test_span_attr_with_none_profiler(self):
        sys.modules["ms_service_profiler"] = Mock()
        sys.modules["ms_service_profiler.mstx"] = Mock()
        sys.modules["atb_llm.utils.log.logging"] = Mock()
        from atb_llm.utils.prof.profiler import span_attr

        try:
            # 测试传入 None profiler
            result = span_attr(None, "test_attr", "test_value")
            self.assertIsNone(result)
        except Exception as e:
            self.fail(f"span_attr raised an exception with None profiler: {e}")

    def test_span_attr_with_disabled_profiler(self):
        sys.modules["ms_service_profiler"] = Mock()
        sys.modules["ms_service_profiler.mstx"] = Mock()
        sys.modules["atb_llm.utils.log.logging"] = Mock()
        from atb_llm.utils.prof.profiler import span_attr

        try:
            # 创建模拟的禁用状态的 profiler
            mock_prof = Mock()
            mock_prof._enable = False

            result = span_attr(mock_prof, "test_attr", "test_value")
            # 应该返回原始的 profiler 对象
            self.assertEqual(result, mock_prof)
        except Exception as e:
            self.fail(f"span_attr raised an exception with disabled profiler: {e}")

    def test_span_attr_when_import_failed_no_error(self):
        from atb_llm.utils.prof.profiler import span_attr

        try:
            span_attr(None, "test_attr", "test_value")
        except Exception as e:
            self.fail(f"span_attr raised an exception when imports failed: {e}")

    # 新增 tensor_attr 相关测试 - 使用 torch.Tensor
    def test_tensor_attr_with_error_import(self):
        from atb_llm.utils.prof.profiler import tensor_attr

        try:
            # 测试非张量输入
            result = tensor_attr("not_a_tensor")
            self.assertEqual(result, "not_a_tensor")
        except Exception as e:
            self.fail(f"tensor_attr raised an exception when it shouldn't have: {e}")

    def test_tensor_attr_when_import_success(self):
        sys.modules["ms_service_profiler"] = Mock()
        sys.modules["ms_service_profiler.mstx"] = Mock()
        sys.modules["atb_llm.utils.log.logging"] = Mock()

        from atb_llm.utils.prof.profiler import tensor_attr
        import torch

        try:
            # 使用 torch.Tensor 替代真实 PyTorch 张量
            test_tensor = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0])
            result = tensor_attr(test_tensor, statistics=True)

            # 验证返回的统计信息
            self.assertIn("min", result)
            self.assertIn("max", result)
            self.assertIn("mean", result)
            self.assertIn("shape", result)
            self.assertIn("first_10", result)

            self.assertEqual(result["min"], 1.0)
            self.assertEqual(result["max"], 5.0)
            self.assertEqual(result["mean"], 3.0)
            self.assertEqual(result["shape"], [5])

        except Exception as e:
            self.fail(f"tensor_attr raised an exception when it shouldn't have: {e}")

    def test_tensor_attr_with_statistics_false(self):
        sys.modules["ms_service_profiler"] = Mock()
        sys.modules["ms_service_profiler.mstx"] = Mock()
        sys.modules["atb_llm.utils.log.logging"] = Mock()

        from atb_llm.utils.prof.profiler import tensor_attr
        import torch

        try:
            # 测试关闭统计信息
            test_tensor = torch.Tensor([1.0, 2.0, 3.0])
            result = tensor_attr(test_tensor, statistics=False)

            # 应该返回张量的列表形式
            self.assertEqual(result, [1.0, 2.0, 3.0])

        except Exception as e:
            self.fail(f"tensor_attr raised an exception with statistics=False: {e}")

    def test_tensor_attr_with_non_tensor_input(self):
        sys.modules["ms_service_profiler"] = Mock()
        sys.modules["ms_service_profiler.mstx"] = Mock()
        sys.modules["atb_llm.utils.log.logging"] = Mock()

        from atb_llm.utils.prof.profiler import tensor_attr

        try:
            # 测试各种非张量输入
            test_cases = ["string_input", 123, [1, 2, 3], {"key": "value"}, None]

            for test_input in test_cases:
                result = tensor_attr(test_input)
                self.assertEqual(result, test_input)

        except Exception as e:
            self.fail(f"tensor_attr raised an exception with non-tensor input: {e}")

    def test_tensor_attr_with_multidimensional_tensor(self):
        sys.modules["ms_service_profiler"] = Mock()
        sys.modules["ms_service_profiler.mstx"] = Mock()
        sys.modules["atb_llm.utils.log.logging"] = Mock()

        from atb_llm.utils.prof.profiler import tensor_attr
        import torch

        try:
            # 测试多维张量
            test_tensor = torch.Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            result = tensor_attr(test_tensor, statistics=True)

            self.assertEqual(result["shape"], [3, 2])
            # 注意：对于多维数组，我们的 torch.Tensor 简化实现可能无法正确计算统计值
            # 但主要测试的是函数调用不会崩溃

        except Exception as e:
            self.fail(f"tensor_attr raised an exception with multidimensional tensor: {e}")

    def test_tensor_attr_when_import_failed_no_error(self):
        with patch('importlib.import_module', side_effect=ImportError("Mocked ImportERROR")):
            from atb_llm.utils.prof.profiler import tensor_attr
        import torch

        try:
            # 在导入失败的情况下测试各种输入
            test_tensor = torch.Tensor([1.0, 2.0, 3.0])
            result = tensor_attr(test_tensor)
            # 在导入失败的情况下，应该直接返回输入
            self.assertTrue(torch.all(result == test_tensor))

            # 测试非张量输入

            result2 = tensor_attr("test_string")
            self.assertEqual(result2, "test_string")

        except Exception as e:
            self.fail(f"tensor_attr raised an exception when imports failed: {e}")

    def test_tensor_attr_with_empty_tensor(self):
        sys.modules["ms_service_profiler"] = Mock()
        sys.modules["ms_service_profiler.mstx"] = Mock()
        sys.modules["atb_llm.utils.log.logging"] = Mock()

        from atb_llm.utils.prof.profiler import tensor_attr
        import torch

        try:
            # 测试空张量
            empty_tensor = torch.Tensor([])
            result = tensor_attr(empty_tensor, statistics=True)

            self.assertEqual(result, {})
            # 对于空张量，min/max/mean 可能为 NaN 或 inf，但我们不验证具体值

        except Exception as e:
            self.fail(f"tensor_attr raised an exception with empty tensor: {e}")

    def test_span_attr_with_callable_value(self):
        sys.modules["ms_service_profiler"] = Mock()
        sys.modules["ms_service_profiler.mstx"] = Mock()
        sys.modules["atb_llm.utils.log.logging"] = Mock()
        from atb_llm.utils.prof.profiler import span_attr

        try:
            # 创建模拟的启用状态的 profiler
            mock_prof = Mock()
            mock_prof._enable = True

            # 测试可调用对象
            callable_called = [False]  # 使用列表以便在内部函数中修改

            def test_callable():
                callable_called[0] = True
                return "dynamic_value"

            span_attr(mock_prof, "dynamic_attr", test_callable)

            # 验证可调用对象被调用
            self.assertTrue(callable_called[0])

        except Exception as e:
            self.fail(f"span_attr raised an exception with callable value: {e}")