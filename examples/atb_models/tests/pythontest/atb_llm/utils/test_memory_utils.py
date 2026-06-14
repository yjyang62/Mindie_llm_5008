# Copyright Huawei Technologies Co., Ltd. 2025. All rights reserved.
import unittest
from unittest.mock import patch

from atb_llm.utils.memory_utils import check_npu_mem


class TestMemoryUtils(unittest.TestCase):
    @patch("torch.npu.mem_get_info", return_value=(20, 30))
    def test_check_npu_mem_enough(self, mock_mem):
        check_npu_mem(0, 15, 1)

    @patch("torch.npu.mem_get_info", return_value=(40, 50))
    def test_check_npu_mem_not_enough(self, mock_mem):
        with self.assertRaises(RuntimeError):
            check_npu_mem(0, 45, 1)