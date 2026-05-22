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
from unittest.mock import MagicMock

import torch

from atb_llm.utils.eplb_expert_data_collect import EplbExpertDataCollect
from atb_llm.utils.env import ENV

WORLD_SIZE = 16


def mock_all_gather_execute(input_tensor_list, string_params):
    input_tensor = input_tensor_list[0]
    result = torch.stack([input_tensor for i in range(WORLD_SIZE)])
    return [result]


class TestEplbExpertDataCollect(unittest.TestCase):
    def setUp(self):
        mock_config = MagicMock()
        mock_config.configure_mock(first_k_dense_replace=3)
        mock_config.configure_mock(num_hidden_layers=10)

        mock_all_gather_operator = MagicMock()
        mock_all_gather_operator.configure_mock(execute=mock_all_gather_execute)

        self.mock_model = MagicMock()
        self.mock_model.configure_mock(config=mock_config)
        self.mock_model.configure_mock(warmup_is_end=True)
        self.mock_model.configure_mock(eplb_level=2)
        self.mock_model.configure_mock(acl_all_gather_operation=mock_all_gather_operator)
        self.mock_model.configure_mock(num_of_device_expert=7)
        self.mock_model.configure_mock(num_speculative_tokens=0)
        EplbExpertDataCollect().set_model_ref(self.mock_model)

    def tearDown(self):
        self.mock_model = None

    def test_split_eplb_expert_data(self):
        model_output_tensor = torch.randn(10, 10)
        single_layer_cumsum_tensor = torch.randint(4000, (self.mock_model.num_of_device_expert,), dtype=torch.int64)
        moe_layer_num = self.mock_model.config.num_hidden_layers - self.mock_model.config.first_k_dense_replace
        input_tensor_list = [model_output_tensor]
        input_tensor_list.extend([single_layer_cumsum_tensor for i in range(moe_layer_num)])
        output_tensor_list = EplbExpertDataCollect().split_eplb_expert_data(input_tensor_list)
        self.assertEqual(len(output_tensor_list), 1)
        assert output_tensor_list[0] is model_output_tensor

    def test_accumulation_expert_cumsum(self):
        single_layer_cumsum_tensor = torch.randint(4000, (self.mock_model.num_of_device_expert,), dtype=torch.int64)
        moe_layer_num = self.mock_model.config.num_hidden_layers - self.mock_model.config.first_k_dense_replace
        EplbExpertDataCollect().cumsum_list = [single_layer_cumsum_tensor for i in range(moe_layer_num)]
        EplbExpertDataCollect().accumulation_expert_cumsum(is_prefill=True)
        EplbExpertDataCollect().cumsum_list = [single_layer_cumsum_tensor for i in range(moe_layer_num)]
        EplbExpertDataCollect().accumulation_expert_cumsum(is_prefill=True)
        assert EplbExpertDataCollect().total_prefill_cumsum_per_expert.equal(2 *
                                       torch.vstack([single_layer_cumsum_tensor for i in range(moe_layer_num)]))

    def test_get_token_num_per_expert(self):
        layer_num = 7
        expert_num = 5
        collect_obj = EplbExpertDataCollect()
        collect_obj.total_prefill_cumsum_per_expert = torch.ones(layer_num, expert_num, dtype=torch.int64, device='npu')
        collect_obj.total_decode_cumsum_per_expert = torch.ones(layer_num, expert_num, dtype=torch.int64, device='npu')
        collect_obj.total_warmup_cumsum_per_expert = torch.ones(layer_num, expert_num, dtype=torch.int64, device='npu')

        expect_result = torch.cat([torch.ones(layer_num, 1, dtype=torch.int64, device='npu'),
                                   torch.zeros(layer_num, expert_num - 1, dtype=torch.int64, device='npu')], dim=1)
        assert collect_obj.get_prefill_token_num_per_expert().equal(expect_result)
        assert collect_obj.get_decode_token_num_per_expert().equal(expect_result)
        assert collect_obj.get_warmup_token_num_per_expert().equal(expect_result)

    def test_all_gather_token_num_per_expert(self):
        ENV.enable_expert_hotpot_gather = True
        layer_num = 7
        expert_num = 5
        collect_obj = EplbExpertDataCollect()
        collect_obj.total_prefill_cumsum_per_expert = torch.ones(layer_num, expert_num, dtype=torch.int64, device='npu')

        diff_result = torch.cat([torch.ones(layer_num, 1, dtype=torch.int64, device='npu'),
                                 torch.zeros(layer_num, expert_num - 1, dtype=torch.int64, device='npu')], dim=1)
        all_gather_result = torch.stack([diff_result for i in range(WORLD_SIZE)])
        expect_result = all_gather_result.transpose(0, 1)

        collect_obj.all_gather_token_num_per_expert(is_prefill=True).equal(expect_result)



