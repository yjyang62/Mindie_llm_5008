# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
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

from ddt import ddt, data, unpack

from atb_llm.utils.quantize.quant_type import QuantType
from atb_llm.utils.quantize.pack_type import PackType, get_pack_type, calc_w4a8_linear_pack_type, \
    calc_linear_pack_type, calc_w8a8sc_linear_pack_type, calc_w8a8_linear_pack_type, calc_w8a16_linear_pack_type, \
    calc_w16a16sc_linear_pack_type

FLOAT_UPPER = "FLOAT"
FLOAT_WEIGHT = "float"
NOT_EXIST_WEIGHT = "not_exist"
W8A8_WEIGHT = "w8a8"
W8A8S_WEIGHT = "w8a8s"
W8A8SC_WEIGHT = "w8a8sc"
W8A8_DYNAMIC_WEIGHT = "w8a8_dynamic"
W8A16_WEIGHT = "w8a16"
W4A16_WEIGHT = "w4a16"
W4A8_DYNAMIC_WEIGHT = "w4a8_dynamic"
W16A16SC_WEIGHT = "w16a16sc"
NORM_WEIGHT = "norm"


@ddt
class TestPackType(unittest.TestCase):
    def setUp(self):
        self.weights = MagicMock()
        self.weights.quant_desc = {
            "float.weight": FLOAT_UPPER,
            "w4a16.weight": "W4A16",
            "w8a8.weight": "W8A8",
            "w8a16.weight": "W8A16",
            "w8a8s.weight": "W8A8S",
            "w8a8sc.weight": "W8A8SC",
            "w8a8_dynamic.weight": "W8A8_DYNAMIC",
            "norm.anti.weight": FLOAT_UPPER,
            "norm.module.weight": FLOAT_UPPER,
            "packed_w8a8sc.weight": "W8A8SC",
            "w4a8_dynamic.weight": "W4A8_DYNAMIC",
            "w16a16sc.weight": "W16A16SC",
        }

    @data(
        ([FLOAT_WEIGHT, FLOAT_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.ALL_FP),
        ([FLOAT_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.ALL_FP),
    )
    @unpack
    def test_float(self, linear_names, norm_name, pack_name, expected_pack_type):
        self.weights.quantize = QuantType.FLOAT
        res = get_pack_type(self.weights, linear_names, norm_name)
        self.assertEqual(res, expected_pack_type)
        res = get_pack_type(self.weights, linear_names, norm_name, pack_name)
        self.assertEqual(res, expected_pack_type)

    @data(
        ([W8A8_WEIGHT, W8A8_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.ALL_W8A8),
        ([W8A8_WEIGHT, FLOAT_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.MIX_W8A8),
        ([W8A8_WEIGHT, W8A8_WEIGHT], NORM_WEIGHT, NOT_EXIST_WEIGHT, PackType.ALL_W8A8_ANTI),
        ([W8A8_WEIGHT, FLOAT_WEIGHT], NORM_WEIGHT, NOT_EXIST_WEIGHT, PackType.MIX_W8A8_ANTI),
        ([FLOAT_WEIGHT, FLOAT_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.ALL_FP),
        ([W8A8_WEIGHT], NOT_EXIST_WEIGHT, W8A8_WEIGHT, PackType.ALL_W8A8),
        ([W8A8_WEIGHT], NORM_WEIGHT, W8A8_WEIGHT, PackType.ALL_W8A8_ANTI),
        ([FLOAT_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.ALL_FP),
    )
    @unpack
    def test_w8a8(self, linear_names, norm_name, pack_name, expected_pack_type):
        self.weights.quantize = QuantType.W8A8
        res = get_pack_type(self.weights, linear_names, norm_name)
        self.assertEqual(res, expected_pack_type)
        res = get_pack_type(self.weights, linear_names, norm_name, pack_name)
        self.assertEqual(res, expected_pack_type)

    @data(
        ([W8A8S_WEIGHT, W8A8S_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.ALL_W8A8),
        ([W8A8S_WEIGHT, FLOAT_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.MIX_W8A8),
        ([W8A8S_WEIGHT, W8A8S_WEIGHT], NORM_WEIGHT, NOT_EXIST_WEIGHT, PackType.ALL_W8A8_ANTI),
        ([W8A8S_WEIGHT, FLOAT_WEIGHT], NORM_WEIGHT, NOT_EXIST_WEIGHT, PackType.MIX_W8A8_ANTI),
        ([FLOAT_WEIGHT, FLOAT_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.ALL_FP),
        ([W8A8S_WEIGHT], NOT_EXIST_WEIGHT, W8A8S_WEIGHT, PackType.ALL_W8A8),
        ([W8A8S_WEIGHT], NORM_WEIGHT, W8A8S_WEIGHT, PackType.ALL_W8A8_ANTI),
        ([FLOAT_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.ALL_FP),
    )
    @unpack
    def test_w8a8s(self, linear_names, norm_name, pack_name, expected_pack_type):
        self.weights.quantize = QuantType.W8A8S
        res = get_pack_type(self.weights, linear_names, norm_name)
        self.assertEqual(res, expected_pack_type)
        res = get_pack_type(self.weights, linear_names, norm_name, pack_name)
        self.assertEqual(res, expected_pack_type)

    @data(
        ([W8A8SC_WEIGHT, FLOAT_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.MIX_W8A8SC),
        ([W8A8SC_WEIGHT, FLOAT_WEIGHT], NORM_WEIGHT, NOT_EXIST_WEIGHT, PackType.MIX_W8A8SC_ANTI),
        ([W8A8SC_WEIGHT], NOT_EXIST_WEIGHT, W8A8SC_WEIGHT, PackType.ALL_W8A8SC),
        ([W8A8SC_WEIGHT], NORM_WEIGHT, W8A8SC_WEIGHT, PackType.ALL_W8A8SC_ANTI),
        ([FLOAT_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.ALL_FP),
    )
    @unpack
    def test_w8a8sc(self, linear_names, norm_name, pack_name, expected_pack_type):
        self.weights.quantize = QuantType.W8A8SC
        res = get_pack_type(self.weights, linear_names, norm_name)
        self.assertEqual(res, expected_pack_type)
        res = get_pack_type(self.weights, linear_names, norm_name, pack_name)
        self.assertEqual(res, expected_pack_type)

    @data(
        ([W8A8_DYNAMIC_WEIGHT, W8A8_DYNAMIC_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.ALL_W8A8_DYNAMIC),
        ([W8A8_DYNAMIC_WEIGHT, FLOAT_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.MIX_W8A8_DYNAMIC),
        ([W8A8_DYNAMIC_WEIGHT, W8A8_DYNAMIC_WEIGHT], NORM_WEIGHT, NOT_EXIST_WEIGHT, PackType.ALL_W8A8_DYNAMIC_ANTI),
        ([W8A8_DYNAMIC_WEIGHT, FLOAT_WEIGHT], NORM_WEIGHT, NOT_EXIST_WEIGHT, PackType.MIX_W8A8_DYNAMIC_ANTI),
        ([FLOAT_WEIGHT, FLOAT_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.ALL_FP),
        ([W8A8_DYNAMIC_WEIGHT], NOT_EXIST_WEIGHT, W8A8_DYNAMIC_WEIGHT, PackType.ALL_W8A8_DYNAMIC),
        ([W8A8_DYNAMIC_WEIGHT], NORM_WEIGHT, W8A8_DYNAMIC_WEIGHT, PackType.ALL_W8A8_DYNAMIC_ANTI),
        ([FLOAT_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.ALL_FP),
    )
    @unpack
    def test_w8a8_dynamic(self, linear_names, norm_name, pack_name, expected_pack_type):
        self.weights.quantize = QuantType.W8A8_DYNAMIC
        res = get_pack_type(self.weights, linear_names, norm_name)
        self.assertEqual(res, expected_pack_type)
        res = get_pack_type(self.weights, linear_names, norm_name, pack_name)
        self.assertEqual(res, expected_pack_type)

    @data(
        ([W8A16_WEIGHT, W8A16_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.ALL_W8A16),
        ([W8A16_WEIGHT, FLOAT_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.MIX_W8A16),
        ([W8A16_WEIGHT, W8A16_WEIGHT], NORM_WEIGHT, NOT_EXIST_WEIGHT, PackType.ALL_W8A16_ANTI),
        ([W8A16_WEIGHT, FLOAT_WEIGHT], NORM_WEIGHT, NOT_EXIST_WEIGHT, PackType.MIX_W8A16_ANTI),
        ([FLOAT_WEIGHT, FLOAT_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.ALL_FP),
        ([W8A16_WEIGHT], NOT_EXIST_WEIGHT, W8A16_WEIGHT, PackType.ALL_W8A16),
        ([W8A16_WEIGHT], NORM_WEIGHT, W8A16_WEIGHT, PackType.ALL_W8A16_ANTI),
        ([FLOAT_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.ALL_FP),
    )
    @unpack
    def test_w8a16(self, linear_names, norm_name, pack_name, expected_pack_type):
        self.weights.quantize = QuantType.W8A16
        res = get_pack_type(self.weights, linear_names, norm_name)
        self.assertEqual(res, expected_pack_type)
        res = get_pack_type(self.weights, linear_names, norm_name, pack_name)
        self.assertEqual(res, expected_pack_type)

    @data(
        ([W4A16_WEIGHT, W4A16_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.ALL_W4A16),
        ([W4A16_WEIGHT, FLOAT_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.MIX_W4A16),
        ([W4A16_WEIGHT, W4A16_WEIGHT], NORM_WEIGHT, NOT_EXIST_WEIGHT, PackType.ALL_W4A16_ANTI),
        ([W4A16_WEIGHT, FLOAT_WEIGHT], NORM_WEIGHT, NOT_EXIST_WEIGHT, PackType.MIX_W4A16_ANTI),
        ([FLOAT_WEIGHT, FLOAT_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.ALL_FP),
        ([W4A16_WEIGHT], NOT_EXIST_WEIGHT, W4A16_WEIGHT, PackType.ALL_W4A16),
        ([W4A16_WEIGHT], NORM_WEIGHT, W4A16_WEIGHT, PackType.ALL_W4A16_ANTI),
        ([FLOAT_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.ALL_FP),
    )
    @unpack
    def test_w4a16(self, linear_names, norm_name, pack_name, expected_pack_type):
        self.weights.quantize = QuantType.W4A16
        res = get_pack_type(self.weights, linear_names, norm_name)
        self.assertEqual(res, expected_pack_type)
        res = get_pack_type(self.weights, linear_names, norm_name, pack_name)
        self.assertEqual(res, expected_pack_type)

    @data(
        ([W4A8_DYNAMIC_WEIGHT, W4A8_DYNAMIC_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.ALL_W4A8),
        ([W4A8_DYNAMIC_WEIGHT, FLOAT_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.MIX_W4A8),
        ([FLOAT_WEIGHT, FLOAT_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.ALL_FP),
        ([W4A8_DYNAMIC_WEIGHT], NOT_EXIST_WEIGHT, W4A8_DYNAMIC_WEIGHT, PackType.ALL_W4A8),
        ([FLOAT_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.ALL_FP),
    )
    @unpack
    def test_calc_w4a8_linear_pack_type(self, linear_names, norm_name, pack_name, expected_pack_type):
        self.weights.quantize = QuantType.W4A8_DYNAMIC
        res = calc_w4a8_linear_pack_type(self.weights, linear_names, norm_name)
        self.assertEqual(res, expected_pack_type)
        res = calc_w4a8_linear_pack_type(self.weights, linear_names, norm_name, pack_name)
        self.assertEqual(res, expected_pack_type)

    @data(
        QuantType.W8A8, QuantType.W8A8S, QuantType.W8A8_PDMIX, QuantType.W4A16, QuantType.W8A16, QuantType.W8A8SC,
        QuantType.W8A8_DYNAMIC, QuantType.W4A8_DYNAMIC, QuantType.W16A16SC, QuantType.FLOAT,
    )
    @patch("atb_llm.utils.quantize.pack_type.calc_w16a16sc_linear_pack_type")
    @patch("atb_llm.utils.quantize.pack_type.calc_w8a8_linear_pack_type")
    @patch("atb_llm.utils.quantize.pack_type.calc_w4a16_linear_pack_type")
    @patch("atb_llm.utils.quantize.pack_type.calc_w8a16_linear_pack_type")
    @patch("atb_llm.utils.quantize.pack_type.calc_w8a8sc_linear_pack_type")
    @patch("atb_llm.utils.quantize.pack_type.calc_w8a8_dynamic_linear_pack_type")
    @patch("atb_llm.utils.quantize.pack_type.calc_w4a8_linear_pack_type")
    def test_calc_linear_pack_type(self, weight_quantize_type, mock_w4a8, mock_w8a8_dynamic,
        mock_w8a8sc, mock_w8a16, mock_w4a16, mock_w8a8, mock_w16a16sc):
        self.weights.quantize = weight_quantize_type
        out = calc_linear_pack_type(self.weights, "", "")
        if weight_quantize_type in [QuantType.W8A8, QuantType.W8A8S, QuantType.W8A8_PDMIX]:
            mock_w8a8.assert_called_once()
        if weight_quantize_type == QuantType.W4A16:
            mock_w4a16.assert_called_once()
        if weight_quantize_type == QuantType.W8A16:
            mock_w8a16.assert_called_once()
        if weight_quantize_type == QuantType.W8A8SC:
            mock_w8a8sc.assert_called_once()
        if weight_quantize_type == QuantType.W8A8_DYNAMIC:
            mock_w8a8_dynamic.assert_called_once()
        if weight_quantize_type == QuantType.W4A8_DYNAMIC:
            mock_w4a8.assert_called_once()
        if weight_quantize_type == QuantType.W16A16SC:
            mock_w16a16sc.assert_called_once()
        if weight_quantize_type == QuantType.FLOAT:
            self.assertEqual(out, PackType.ALL_FP)

    @data(
        ([W8A8SC_WEIGHT, W8A8SC_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.MIX_W8A8SC),
        ([W8A8SC_WEIGHT, FLOAT_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.MIX_W8A8SC),
        ([FLOAT_WEIGHT, FLOAT_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.ALL_FP),
        ([W8A8SC_WEIGHT], NOT_EXIST_WEIGHT, W8A8SC_WEIGHT, PackType.ALL_W8A8SC),
        ([FLOAT_WEIGHT], NOT_EXIST_WEIGHT, FLOAT_WEIGHT, PackType.ALL_FP),
        ([W8A8SC_WEIGHT, W8A8SC_WEIGHT], NORM_WEIGHT, NOT_EXIST_WEIGHT, PackType.MIX_W8A8SC_ANTI),
        ([W8A8SC_WEIGHT, FLOAT_WEIGHT], NORM_WEIGHT, NOT_EXIST_WEIGHT, PackType.MIX_W8A8SC_ANTI),
        ([W8A8SC_WEIGHT], NORM_WEIGHT, W8A8SC_WEIGHT, PackType.ALL_W8A8SC_ANTI),
    )
    @unpack
    def test_calc_w8a8sc_linear_pack_type(self, linear_names, norm_name, pack_name, expected_pack_type):
        self.weights.quantize = QuantType.W8A8SC
        if norm_name != NOT_EXIST_WEIGHT:
            self.weights.quant_desc[f'{norm_name}.anti.weight'] = "FLOAT"
        res = calc_w8a8sc_linear_pack_type(self.weights, linear_names, norm_name)
        self.assertEqual(res, expected_pack_type)
        res = calc_w8a8sc_linear_pack_type(self.weights, linear_names, norm_name, pack_name)
        self.assertEqual(res, expected_pack_type)

    @data(
        ([W8A8_WEIGHT, W8A8_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.ALL_W8A8),
        ([W8A8_WEIGHT, FLOAT_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.MIX_W8A8),
        ([FLOAT_WEIGHT, FLOAT_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.ALL_FP),
        ([W8A8_WEIGHT], NOT_EXIST_WEIGHT, W8A8_WEIGHT, PackType.ALL_W8A8),
        ([FLOAT_WEIGHT], NOT_EXIST_WEIGHT, FLOAT_WEIGHT, PackType.ALL_FP),
        ([W8A8_WEIGHT, W8A8_WEIGHT], NORM_WEIGHT, NOT_EXIST_WEIGHT, PackType.ALL_W8A8_ANTI),
        ([W8A8_WEIGHT, FLOAT_WEIGHT], NORM_WEIGHT, NOT_EXIST_WEIGHT, PackType.MIX_W8A8_ANTI),
    )
    @unpack
    def test_calc_w8a8_linear_pack_type(self, linear_names, norm_name, pack_name, expected_pack_type):
        self.weights.quantize = QuantType.W8A8
        if norm_name != NOT_EXIST_WEIGHT:
            self.weights.quant_desc[f'{norm_name}.module.weight'] = "FLOAT"
        res = calc_w8a8_linear_pack_type(self.weights, linear_names, norm_name)
        self.assertEqual(res, expected_pack_type)
        res = calc_w8a8_linear_pack_type(self.weights, linear_names, norm_name, pack_name)
        self.assertEqual(res, expected_pack_type)

    @data(
        ([W8A16_WEIGHT, W8A16_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.ALL_W8A16),
        ([W8A16_WEIGHT, FLOAT_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.MIX_W8A16),
        ([FLOAT_WEIGHT, FLOAT_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.ALL_FP),
        ([W8A16_WEIGHT], NOT_EXIST_WEIGHT, W8A16_WEIGHT, PackType.ALL_W8A16),
        ([FLOAT_WEIGHT], NOT_EXIST_WEIGHT, FLOAT_WEIGHT, PackType.ALL_FP),
        ([W8A16_WEIGHT, W8A16_WEIGHT], NORM_WEIGHT, NOT_EXIST_WEIGHT, PackType.ALL_W8A16_ANTI),
        ([W8A16_WEIGHT, FLOAT_WEIGHT], NORM_WEIGHT, NOT_EXIST_WEIGHT, PackType.MIX_W8A16_ANTI),
    )
    @unpack
    def test_calc_w8a16_linear_pack_type(self, linear_names, norm_name, pack_name, expected_pack_type):
        self.weights.quantize = QuantType.W8A16
        if norm_name != NOT_EXIST_WEIGHT:
            self.weights.quant_desc[f'{norm_name}.module.weight'] = "FLOAT"
        res = calc_w8a16_linear_pack_type(self.weights, linear_names, norm_name)
        self.assertEqual(res, expected_pack_type)
        res = calc_w8a16_linear_pack_type(self.weights, linear_names, norm_name, pack_name)
        self.assertEqual(res, expected_pack_type)
    
    
    @data(
        ([W16A16SC_WEIGHT], NOT_EXIST_WEIGHT, W16A16SC_WEIGHT, PackType.ALL_W16A16SC),
        ([FLOAT_WEIGHT], NOT_EXIST_WEIGHT, NOT_EXIST_WEIGHT, PackType.ALL_FP),
    )
    @unpack
    def test_calc_w16a16sc_linear_pack_type(self, linear_names, norm_name, pack_name, expected_pack_type):
        self.weights.quantize = QuantType.W16A16SC
        res = calc_w16a16sc_linear_pack_type(self.weights, linear_names, norm_name, pack_name)
        self.assertEqual(res, expected_pack_type)


if __name__ == "__main__":
    unittest.main()