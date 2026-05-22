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
from unittest.mock import MagicMock, patch

import torch
from ddt import ddt, data

from atb_llm.layers import QuantTypeV3
from atb_llm.nn.parameter import Parameter
from atb_llm.utils.loader.weight_loader import get_linear_quant_type, replicated_loader, \
    is_all_zero, check_weight_exists
from atb_llm.utils.loader.file_loader import BaseFileLoader
from atb_llm.utils.loader.safetensor_file_loader import SafetensorFileLoader


KEY = "key"


@ddt
class TestWeightLoader(unittest.TestCase):
    @data((None, torch.float16, QuantTypeV3.FLOAT16),
          (None, torch.bfloat16, QuantTypeV3.BFLOAT16),
          ({KEY: "FLOAT"}, torch.float16, QuantTypeV3.FLOAT16),
          ({KEY: "W8A8"}, torch.float16, QuantTypeV3.W8A8),
          ({"invalid_key": "W8A8"}, torch.float16, QuantTypeV3.INVALID)
    )
    @patch('atb_llm.utils.loader.weight_loader.os')
    @patch('atb_llm.utils.loader.weight_loader.json')
    @patch("atb_llm.utils.loader.weight_loader.file_utils")
    def test_get_linear_quant_type(self, test_data, mock_os, mock_json, _1):
        return_value, dtype, expected_quant_type = test_data
        mock_os.path.exists.return_value = True
        mock_json.load.return_value = return_value
        self.assertEqual(get_linear_quant_type("fake_path", dtype, KEY), expected_quant_type)

    def test_replicated_loader(self):
        param = Parameter(prefix="fake_prefix", suffix="fake_suffix")
        file_loader = BaseFileLoader("fake_path")
        file_loader.get_tensor = MagicMock(return_value=torch.tensor([1, 2, 3]))
        tensor = replicated_loader(param, file_loader, [KEY])
        self.assertTrue(torch.equal(tensor, torch.tensor([1, 2, 3])))

        tensor = replicated_loader(param, file_loader, [KEY, KEY])
        self.assertTrue(torch.equal(tensor, torch.tensor([1, 2, 3, 1, 2, 3])))

        with self.assertRaises(ValueError):
            replicated_loader(param, file_loader, [KEY, KEY], is_uniform=True)

        file_loader.get_tensor = MagicMock(return_value=torch.tensor([[1, 2, 3]]))
        tensor = replicated_loader(param, file_loader, [KEY, KEY], is_uniform=True)
        self.assertTrue(torch.equal(tensor, torch.tensor([[1, 2, 3]])))

    def test_is_all_zero(self):
        file_loader = BaseFileLoader("fake_path")
        file_loader.get_tensor = MagicMock(return_value=torch.tensor([1, 2, 3]))
        out = is_all_zero(file_loader, "fake_tensor")
        self.assertFalse(out)

        file_loader.get_tensor = MagicMock(side_effect=ValueError(''))
        out = is_all_zero(file_loader, "fake_tensor")
        self.assertFalse(out)

    @patch('atb_llm.utils.loader.safetensor_file_loader.get_weight_filenames')
    def test_check_weight_exists(self, _):
        file_loader = SafetensorFileLoader("fake_path", torch.device(0))
        file_loader.get_filename = MagicMock(return_value=("fake_file_name", "fake_tensor"))
        out = check_weight_exists(file_loader, "fake_tensor")
        self.assertTrue(out)

        file_loader.get_filename = MagicMock(side_effect=ValueError(''))
        out = check_weight_exists(file_loader, "fake_tensor")
        self.assertFalse(out)


if __name__ == '__main__':
    unittest.main()