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
import json
import copy
from ddt import ddt, data, unpack
import torch
from atb_llm.utils.moe_utils import (parse_ep_balance_file, assign, random_generation,
                                    save_eplb_data, calculate_eplb_param)
from atb_llm.utils.env import ENV

JOSN_DICT = {
    "moe_layer_count": 1,
    "layer_list": [
        {
            "layer_id": 0,
            "device_count": 2,
            "device_list": [
                {
                    "device_id": 0,
                    "device_expert": [
                        1,
                        2,
                        3,
                        4
                    ]
                },
                {
                    "device_id": 1,
                    "device_expert": [
                        5,
                        6,
                        7,
                        8
                    ]
                }
                
            ]
        }
    ]
}


COMBINATIONS = [(64, 64, 0), (32, 32, 0), (56, 24, 4), (16, 0, 0),
                (112, 80, 16), (56, 24, 8), (64, 128, 0), (256, 256, 32), (352, 96, 32)]
                

@ddt
class TestMoeUtil(unittest.TestCase):
    def test_parse_ok(self):
        ep_file_json = "balance_file.json"
        with open(ep_file_json, "w") as file:
            json.dump(JOSN_DICT, file)
        ep_info = parse_ep_balance_file(ep_file_json, 2, 0)
        self.assertNotEqual(ep_info, None)

    def test_parse_faild_case1(self):
        new_json = copy.deepcopy(JOSN_DICT)
        new_json["moe_layer_count"] = 2
        ep_file_json = "balance_file.json"
        with open(ep_file_json, "w") as file:
            json.dump(new_json, file)
        ep_info = parse_ep_balance_file(ep_file_json, 2, 0)
        self.assertEqual(ep_info, None)

    def test_parse_faild_case2(self):
        new_json = copy.deepcopy(JOSN_DICT)
        new_json["layer_list"][0]["device_count"] = 1
        ep_file_json = "balance_file.json"
        with open(ep_file_json, "w") as file:
            json.dump(new_json, file)
        ep_info = parse_ep_balance_file(ep_file_json, 2, 0)
        self.assertEqual(ep_info, None)

    def test_assign(self):
        assignment = assign(64, 8)
        expected_assignment = [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20, 21, 22, 23],
            [24, 25, 26, 27, 28, 29, 30, 31],
            [32, 33, 34, 35, 36, 37, 38, 39],
            [40, 41, 42, 43, 44, 45, 46, 47],
            [48, 49, 50, 51, 52, 53, 54, 55],
            [56, 57, 58, 59, 60, 61, 62, 63]
        ]
        self.assertEqual(assignment, expected_assignment)

    def test_random_generation(self):
        experts_table = random_generation(n_layer=8, n_expert=32, device_count=8, n_redundant=8)
        expected_experts_table = [
         [[0, 1, 2, 3, 0], [4, 5, 6, 7, 4], [8, 9, 10, 11, 8], [12, 13, 14, 15, 12],
          [16, 17, 18, 19, 16], [20, 21, 22, 23, 20], [24, 25, 26, 27, 24], [28, 29, 30, 31, 28]],
         [[0, 1, 2, 3, 0], [4, 5, 6, 7, 4], [8, 9, 10, 11, 8], [12, 13, 14, 15, 12],
          [16, 17, 18, 19, 16], [20, 21, 22, 23, 20], [24, 25, 26, 27, 24], [28, 29, 30, 31, 28]],
         [[0, 1, 2, 3, 0], [4, 5, 6, 7, 4], [8, 9, 10, 11, 8], [12, 13, 14, 15, 12],
          [16, 17, 18, 19, 16], [20, 21, 22, 23, 20], [24, 25, 26, 27, 24], [28, 29, 30, 31, 28]],
         [[0, 1, 2, 3, 0], [4, 5, 6, 7, 4], [8, 9, 10, 11, 8], [12, 13, 14, 15, 12],
          [16, 17, 18, 19, 16], [20, 21, 22, 23, 20], [24, 25, 26, 27, 24], [28, 29, 30, 31, 28]],
         [[0, 1, 2, 3, 0], [4, 5, 6, 7, 4], [8, 9, 10, 11, 8], [12, 13, 14, 15, 12],
          [16, 17, 18, 19, 16], [20, 21, 22, 23, 20], [24, 25, 26, 27, 24], [28, 29, 30, 31, 28]],
         [[0, 1, 2, 3, 0], [4, 5, 6, 7, 4], [8, 9, 10, 11, 8], [12, 13, 14, 15, 12],
          [16, 17, 18, 19, 16], [20, 21, 22, 23, 20], [24, 25, 26, 27, 24], [28, 29, 30, 31, 28]],
         [[0, 1, 2, 3, 0], [4, 5, 6, 7, 4], [8, 9, 10, 11, 8], [12, 13, 14, 15, 12],
          [16, 17, 18, 19, 16], [20, 21, 22, 23, 20], [24, 25, 26, 27, 24], [28, 29, 30, 31, 28]],
         [[0, 1, 2, 3, 0], [4, 5, 6, 7, 4], [8, 9, 10, 11, 8], [12, 13, 14, 15, 12],
          [16, 17, 18, 19, 16], [20, 21, 22, 23, 20], [24, 25, 26, 27, 24], [28, 29, 30, 31, 28]]
        ]
        self.assertEqual(experts_table, expected_experts_table)

    def test_save_eplb_data(self):
        ENV.enable_expert_hotpot_gather = True
        ENV.expert_hotpot_dump_path = './hotpot'
        prefill_cumsum = torch.rand(2,2)
        decode_cumsum = torch.rand(2,2)
        save_eplb_data(0, prefill_cumsum, "prefill", 0)
        save_eplb_data(0, decode_cumsum, "decode", 0)

    @data(*COMBINATIONS)
    @unpack
    @unittest.mock.patch("atb_llm.utils.moe_utils.parse_ep_file")
    def test_calculate_eplb_param_internal(self, device_count, n_redundant, n_dangling, mock_epfile):
        mock_epfile.return_value = random_generation(8, 256, device_count, n_redundant, n_dangling=n_dangling)
        mix_shared_routing, num_dangling_experts, num_redundant_experts = calculate_eplb_param("", 256)
        self.assertFalse(mix_shared_routing)
        self.assertEqual(num_dangling_experts, n_dangling)
        self.assertEqual(num_redundant_experts, n_redundant)

    @unittest.mock.patch("atb_llm.utils.moe_utils.parse_ep_file",
                         return_value=random_generation(n_layer=8, n_expert=257, device_count=64, n_redundant=63))
    def test_calculate_eplb_param_mix(self, mock_epfile):
        mix_shared_routing, num_dangling_experts, num_redundant_experts = calculate_eplb_param("", 256)
        self.assertTrue(mix_shared_routing)
        self.assertEqual(num_dangling_experts, 0)
        self.assertEqual(num_redundant_experts, 64)


if __name__ == "__main__":
    unittest.main()
