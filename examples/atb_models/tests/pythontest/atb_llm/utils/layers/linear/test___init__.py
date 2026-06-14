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
from unittest.mock import patch, MagicMock
from ddt import ddt, data, unpack

import torch

from atb_llm.utils.layers import TensorReplicatedLinear, TensorParallelRowLinear, TensorParallelColumnLinear, get_linear, TensorHead, TensorParallelHead
from atb_llm.utils.quantize.quant_type import QuantType
from atb_llm.utils.dist import FakeGroup
from atb_llm.utils.layers.attention.process_mla_linear import preprocess_linear_for_rope
from atb_llm.utils.log import logger


combinations = [
    (a, tuple(b)) 
    for a in [QuantType.W4A16, QuantType.W8A16, QuantType.W8A8_DYNAMIC]
    for b in [[torch.ones(512, 256, dtype=torch.int8)] * 3, [torch.ones(256, 256, dtype=torch.int8)] * 3]
]
combinations_w4a8 = [
    (a, tuple(b)) 
    for a in [QuantType.W4A8_DYNAMIC]
    for b in [[torch.ones(256, 1768, dtype=torch.int8), torch.ones(512, 1, dtype=torch.int8), torch.ones(512, 28, dtype=torch.int8), torch.ones(512, 1, dtype=torch.int8)]]
]
combinations.extend(combinations_w4a8)
combinations.append((None, torch.ones(256, 256, dtype=torch.float16)))


@ddt
class TestTensorParallelColumnLinear(unittest.TestCase):
    @data(*combinations)
    @unpack
    def test_load_moe(self, quantize, weight):
        config, weights = MagicMock(), MagicMock()
        config.quantize = quantize
        config.is_nzcasted = False
        weights.get_multi_weights_col.return_value = weight
        prefixes = [[f"moe.{i}.gate_up_proj"] for i in range(64)]
        if quantize in [QuantType.W8A16, QuantType.W8A8_DYNAMIC]:
            with patch('atb_llm.utils.layers.linear.linear_utils.LinearUtils.check_transpose', return_value=1) as _:
                module = TensorParallelColumnLinear.load_moe(config, prefixes, weights, None)
                golden_weight = torch.stack([weight[0] for _ in range(64)], dim=0)
                self.assertTrue(torch.allclose(golden_weight, module.linear.weight))
        elif quantize in [QuantType.W4A8_DYNAMIC]:
            module = TensorParallelColumnLinear.load_moe(config, prefixes, weights, None)
            golden_weight = torch.stack([weight[0] for _ in range(64)], dim=0)
            golden_weight = golden_weight.transpose(1,2)
            self.assertTrue(torch.allclose(golden_weight, module.linear.weight))
        elif quantize is None:
            module = TensorParallelColumnLinear.load_moe(config, prefixes, weights, None)
            golden_weight = torch.stack([weight for _ in range(64)])
            self.assertTrue(torch.allclose(golden_weight, module.linear.weight))

    @data(True, False)
    def test_load_qkv(self, bias):
        config, weights = MagicMock(), MagicMock()
        config.quantize = None
        weight = torch.ones(256, 256, dtype=torch.float16)
        weights.get_weights_col_packed_qkv.return_value = weight
        weights.get_tensor_col_packed_qkv.return_value = weight
        module = TensorParallelColumnLinear.load_qkv(config, "qkv", weights, bias, weight, 64)
        self.assertTrue(torch.allclose(weight, module.linear.weight))
        if bias:
            self.assertTrue(torch.allclose(weight, module.linear.bias))

    @data(True, False)
    def test_load_gate_up(self, bias):
        config, weights = MagicMock(), MagicMock()
        config.quantize = None
        weight = torch.ones(256, 256, dtype=torch.float16)
        weights.get_weights_col_packed_mlp.return_value = weight
        weights.get_tensor_col_packed_mlp.return_value = weight
        module = TensorParallelColumnLinear.load_gate_up(config, "mlp", weights, bias)
        self.assertTrue(torch.allclose(weight, module.linear.weight))
        if bias:
            self.assertTrue(torch.allclose(weight, module.linear.bias))   

    @data(True, False)
    @patch('atb_llm.utils.layers.linear.linear_utils.LinearUtils.check_transpose', return_value=1)
    def test_load_multi(self, bias, mock_check_transpose):
        config, weights = MagicMock(), MagicMock()
        config.quantize = None
        weight = torch.ones(3072, 1536, dtype=torch.float16)
        weight = preprocess_linear_for_rope(weight, config, "projq")
        weights.get_multi_weights_col.return_value = weight
        if bias:
            bias_tensor = torch.ones(2, 3, dtype=torch.float16)
            weights.get_sharded.return_value = bias_tensor
 
        module = TensorParallelColumnLinear.load_multi(config, "q_b_proj", weights, bias, dim=0, proj_name="projq")
        self.assertTrue(torch.allclose(weight, module.linear.weight))
        if bias:
            self.assertTrue(torch.allclose(torch.ones(16, 3, dtype=torch.float16), module.linear.bias))


@ddt
class TestTensorParallelRowLinear(unittest.TestCase):
    @data(*combinations)
    @unpack
    def test_load_moe(self, quantize, weight):
        config, weights = MagicMock(), MagicMock()
        config.quantize = quantize
        config.is_nzcasted = False
        weights.get_multi_weights_row.return_value = weight
        prefixes = [f"moe.{i}.down_proj" for i in range(64)]
        if quantize in [QuantType.W8A16, QuantType.W8A8_DYNAMIC]:
            with patch('atb_llm.utils.layers.linear.linear_utils.LinearUtils.check_transpose', return_value=1) as _:
                module = TensorParallelRowLinear.load_moe(config, prefixes, FakeGroup(0, 8), weights, None)
                golden_weight = torch.stack([weight[0] for _ in range(64)], dim=0)
                self.assertTrue(torch.allclose(golden_weight, module.linear.weight))
        elif quantize in [QuantType.W4A8_DYNAMIC]:
            module = TensorParallelRowLinear.load_moe(config, prefixes, FakeGroup(0, 8), weights, None)
            golden_weight = torch.stack([weight[0] for _ in range(64)], dim=0)
            golden_weight = golden_weight.transpose(1,2)
            self.assertTrue(torch.allclose(golden_weight, module.linear.weight))
        elif quantize is None:
            module = TensorParallelRowLinear.load_moe(config, prefixes, FakeGroup(0, 8), weights, None)
            golden_weight = torch.stack([weight for _ in range(64)])
            self.assertTrue(torch.allclose(golden_weight, module.linear.weight))

    @data(True, False)
    def test_load(self, bias):
        config, weights = MagicMock(), MagicMock()
        config.quantize = None
        weight = torch.ones(256, 256, dtype=torch.float16)
        weights.get_multi_weights_row.return_value = weight
        if bias:
            bias_tensor = torch.ones(2, 3, dtype=torch.float16)
            weights.get_tensor.return_value = bias_tensor
        module = TensorParallelRowLinear.load(config, "down_proj", weights, bias, True)
        self.assertTrue(torch.allclose(weight, module.linear.weight))
        if bias:
            self.assertTrue(torch.allclose(bias_tensor, module.linear.bias))


@ddt
class TestTensorReplicatedLinear(unittest.TestCase):        
    @data(True, False)
    @patch('atb_llm.utils.layers.linear.linear_utils.LinearUtils.check_transpose', return_value=1)
    def test_load(self, bias, mock_check_transpose):
        config, weights = MagicMock(), MagicMock()
        config.quantize = None
        weight = torch.ones(256, 256, dtype=torch.float16)
        weight = preprocess_linear_for_rope(weight, config, "projk")
        weights.get_replicated_weights.return_value = weight
        if bias:
            bias_tensor = torch.ones(2, 3, dtype=torch.float16)
            weights.get_tensor.return_value = bias_tensor
        module = TensorReplicatedLinear.load(config, "kv_a_proj", weights, bias)
        self.assertTrue(torch.allclose(weight, module.linear.weight))
        if bias:
            self.assertTrue(torch.allclose(bias_tensor, module.linear.bias))


@ddt
class TestTensorHead(unittest.TestCase):
    @data(None, "gptq")
    def test_load(self, quantize):
        config, weights = MagicMock(), MagicMock()
        config.quantize = quantize
        weight = torch.ones(256, 256, dtype=torch.float16)
        weights.get_whole_tensor.return_value = weight
        module = TensorHead.load_weight(config, "kv_a_proj", weights)
        self.assertTrue(torch.allclose(weight, module.linear.weight))


@ddt
class TestTensorParallelHead(unittest.TestCase):
    @data(None, "gptq")
    def test_load_weight(self, quantize):
        config, weights = MagicMock(), MagicMock()
        config.quantize = quantize
        weight = torch.ones(256, 256, dtype=torch.float16)
        weights.get_tensor.return_value = weight
        module = TensorParallelHead.load_weight(config, "kv_a_proj", weights)
        self.assertTrue(torch.allclose(weight, module.linear.weight))
    
    @data(None, "gptq")
    def test_load(self, quantize):
        config, weights = MagicMock(), MagicMock()
        config.quantize = quantize
        weight = torch.ones(256, 256, dtype=torch.float16)
        weights.get_sharded.return_value = weight
        weights.process_group = FakeGroup(0, 8)
        module = TensorParallelHead.load(config, "kv_a_proj", weights)
        self.assertTrue(torch.allclose(weight, module.linear.weight))
    
    @data(None, "gptq")
    def test_load_process_group(self, quantize):
        try:
            config, weights = MagicMock(), MagicMock()
            config.quantize = quantize
            weight = torch.ones(256, 256, dtype=torch.float16)
            weights.get_sharded.return_value = weight
            weights.process_group = FakeGroup(0, 1)
            module = TensorParallelHead.load(config, "kv_a_proj", weights)
            self.assertTrue(torch.allclose(weight, module.linear.weight))
        except Exception as e:
            logger.error(e)


@ddt
class TestGetLinear(unittest.TestCase):
    
    @data(QuantType.W8A8, QuantType.W4A16, QuantType.W8A16, QuantType.W8A8SC, QuantType.W8A8_DYNAMIC, QuantType.W8A8_PDMIX, QuantType.W4A8_DYNAMIC,)
    def test_get_linear_quant(self, quantize):
        if quantize == QuantType.W8A8_DYNAMIC:
            weight = torch.ones(64, 128, 512, dtype=torch.int8)
            scale = torch.ones(64, 512, 1, dtype=torch.float16)
            offset = torch.ones(64, 512, 1, dtype=torch.float16)
            weights = (weight, scale, offset)
            with patch('atb_llm.utils.layers.linear.linear_utils.LinearUtils.check_transpose', return_value=1) as _:
                linear = get_linear(weights, None, quantize, False)
                self.assertEqual(linear.weight.shape, (64, 128, 512))
        elif quantize == QuantType.W8A8:
            weight = torch.ones(64, 128, 512, dtype=torch.int8)
            weight_scale = torch.ones(64, 512, 1, dtype=torch.float16)
            weight_offset = torch.ones(64, 512, 1, dtype=torch.float16)
            deq_scale = torch.ones(64, 512)
            quant_bias = torch.ones(64, 512)
            input_scale = torch.ones(64)
            input_offset = torch.ones(64)
            with patch('atb_llm.utils.layers.linear.linear_utils.LinearUtils.check_transpose', return_value=1) as _:
                weights = (weight, deq_scale, quant_bias, input_scale, input_offset)
                linear = get_linear(weights, None, quantize, False)
                self.assertEqual(linear.weight.shape, (64, 128, 512))
        elif quantize == QuantType.W8A8_PDMIX:
            weight = torch.ones(64, 128, 512, dtype=torch.int8)
            weight_scale = torch.ones(64, 512, 1, dtype=torch.float16)
            weight_offset = torch.ones(64, 512, 1, dtype=torch.float16)
            deq_scale = torch.ones(64, 512)
            quant_bias = torch.ones(64, 512)
            input_scale = torch.ones(64)
            input_offset = torch.ones(64)
            with patch('atb_llm.utils.layers.linear.linear_utils.LinearUtils.check_transpose', return_value=1) as _:
                weights = (weight, weight_scale, weight_offset, deq_scale, quant_bias, input_scale, input_offset)
                linear = get_linear(weights, None, quantize, False)
                self.assertEqual(linear.weight.shape, (64, 128, 512))
        elif quantize == QuantType.W8A8SC:
            weight = torch.ones(64, 128, 512, dtype=torch.int8)
            weight_scale = torch.ones(64, 512, 1, dtype=torch.float16)
            weight_offset = torch.ones(64, 512, 1, dtype=torch.float16)
            deq_scale = torch.ones(64, 512)
            quant_bias = torch.ones(64, 512)
            input_scale = torch.ones(64)
            input_offset = torch.ones(64)
            index = torch.ones(64)
            weights = (weight, deq_scale, quant_bias, input_scale, input_offset, index)
            linear = get_linear(weights, None, quantize, False)
            self.assertEqual(linear.weight.shape, (64, 128, 512))
        elif quantize == QuantType.W4A16:
            weight = torch.ones(64, 128, 512, dtype=torch.int8)
            weight_scale = torch.ones(64, 512, 1, dtype=torch.float16)
            weight_offset = torch.ones(64, 512, 1, dtype=torch.float16)
            weights = (weight, weight_scale, weight_offset)
            linear = get_linear(weights, None, quantize, False)
            self.assertEqual(linear.weight.shape, (64, 128, 256))
        elif quantize == QuantType.W8A16:
            weight = torch.ones(64, 128, 512, dtype=torch.int8)
            weight_scale = torch.ones(64, 512, 1, dtype=torch.float16)
            weight_offset = torch.ones(64, 512, 1, dtype=torch.float16)
            weights = (weight, weight_scale, weight_offset)
            linear = get_linear(weights, None, quantize, False)
            self.assertEqual(linear.weight.shape, (64, 128, 512))
        elif quantize == QuantType.W4A8_DYNAMIC:
            weight = torch.ones(64, 128, 256, dtype=torch.int8)
            scale = torch.ones(64, 256, 1, dtype=torch.float16)
            offset = torch.ones(64, 512, 1, dtype=torch.float16)
            weights = (weight, scale, offset)
            linear = get_linear(weights, None, quantize, False)
            self.assertEqual(linear.weight.shape, (64, 256, 128))
        elif quantize == QuantType.W16A16S:
            weight = torch.ones(256, 256, dtype=torch.float16)
            linear = get_linear(weight, None, quantize, False)
            self.assertTrue(torch.allclose(weight, linear.weight))
        elif quantize == QuantType.W16A16SC:
            weight = torch.ones(64, 128, 256, dtype=torch.int8)
            quant_bias = torch.ones(64, 512)
            index = torch.ones(64)
            weights = (weight, quant_bias, index)
            linear = get_linear(weights, None, quantize, False)
            self.assertTrue(torch.allclose(weight, linear.weight))
        else:
            weight = torch.ones(256, 256, dtype=torch.float16)
            linear = get_linear(weight, None, quantize, False)
            self.assertTrue(torch.allclose(weight, linear.weight))
        
    @data(QuantType.W8A8, QuantType.W4A16, QuantType.W8A16, QuantType.W8A8SC, QuantType.W8A8_DYNAMIC, QuantType.W8A8_PDMIX, QuantType.W4A8_DYNAMIC, "fp8")
    def test_get_linear_expection(self, quantize):
        if quantize == QuantType.W8A8_DYNAMIC:
            weight = torch.ones(64, 128, 512, dtype=torch.int8)
            scale = torch.ones(64, 256, 1, dtype=torch.float16)
            weights = (weight, scale)
            try:
                linear = get_linear(weights, None, quantize, False)
                self.assertEqual(linear.weight.shape, (64, 128, 512))
            except Exception as e:
                logger.error(e)
                
        elif quantize == QuantType.W8A8:
            weight = torch.ones(64, 128, 512, dtype=torch.int8)
            scale = torch.ones(64, 256, 1, dtype=torch.float16)
            weights = (weight, scale)
            try:
                linear = get_linear(weights, None, quantize, False)
                self.assertEqual(linear.weight.shape, (64, 128, 512))
            except Exception as e:
                logger.error(e)
            
        elif quantize == QuantType.W8A8_PDMIX:
            weight = torch.ones(64, 128, 512, dtype=torch.int8)
            scale = torch.ones(64, 256, 1, dtype=torch.float16)
            weights = (weight, scale)
            try:
                linear = get_linear(weight, None, quantize, False)
                self.assertEqual(linear.weights.shape, (64, 128, 512))
            except Exception as e:
                logger.error(e)
            
        elif quantize == QuantType.W8A8SC:
            weight = torch.ones(64, 128, 512, dtype=torch.int8)
            scale = torch.ones(64, 256, 1, dtype=torch.float16)
            weights = (weight, scale)
            try:
                linear = get_linear(weight, None, quantize, False)
                self.assertEqual(linear.weights.shape, (64, 128, 512))
            except Exception as e:
                logger.error(e)
            
        elif quantize == QuantType.W4A16:
            weight = torch.ones(64, 128, 512, dtype=torch.int8)
            scale = torch.ones(64, 256, 1, dtype=torch.float16)
            weights = (weight, scale)
            try:
                linear = get_linear(weights, None, quantize, False)
                self.assertEqual(linear.weight.shape, (64, 128, 256))
            except Exception as e:
                logger.error(e)
            
        elif quantize == QuantType.W8A16:
            weight = torch.ones(64, 128, 512, dtype=torch.int8)
            scale = torch.ones(64, 256, 1, dtype=torch.float16)
            weights = (weight, scale)
            try:
                linear = get_linear(weights, None, quantize, False)
                self.assertEqual(linear.weight.shape, (64, 128, 512))
            except Exception as e:
                logger.error(e)
            
        elif quantize == QuantType.W4A8_DYNAMIC:
            weight = torch.ones(64, 128, 256, dtype=torch.int8)
            scale = torch.ones(64, 256, 1, dtype=torch.float16)
            weights = (weight, scale)
            try:
                linear = get_linear(weights, None, quantize, False)
                self.assertEqual(linear.weight.shape, (64, 256, 128))
            except Exception as e:
                logger.error(e)
        
        elif quantize == QuantType.W16A16SC:
            weight = torch.ones(64, 128, 256, dtype=torch.int8)
            scale = torch.ones(64, 256, 1, dtype=torch.float16)
            weights = (weight, scale)
            try:
                linear = get_linear(weights, None, quantize, False)
                self.assertEqual(linear.weight.shape, (64, 256, 128))
            except Exception as e:
                logger.error(e)
        if quantize == "fp8":
            weight = torch.ones(64, 128, 256, dtype=torch.int8)
            scale = torch.ones(64, 256, 1, dtype=torch.float16)
            weights = (weight, scale)
            try:
                linear = get_linear(weights, None, quantize, False)
                self.assertEqual(linear.weight.shape, (64, 256, 128))
            except Exception as e:
                logger.error(e)
        else:
            weight = torch.ones(256, 256, dtype=torch.float16)
            linear = get_linear(weight, None, quantize, False)
            self.assertTrue(torch.allclose(weight, linear.weight))
        
    @data(QuantType.W8A8, QuantType.W4A16, QuantType.W8A16, QuantType.W8A8SC, QuantType.W8A8_DYNAMIC, QuantType.W8A8_PDMIX, QuantType.W4A8_DYNAMIC)
    def test_get_linear_float(self, quantize):
        weight = torch.ones(256, 256, dtype=torch.float16)
        linear = get_linear(weight, None, quantize, False)
        self.assertTrue(torch.allclose(weight, linear.weight))



if __name__ == '__main__':
    unittest.main()