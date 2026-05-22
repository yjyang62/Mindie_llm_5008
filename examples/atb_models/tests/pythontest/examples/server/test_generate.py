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

import torch
import numpy as np

from atb_llm.utils.env import ENV
from examples.server.generate import partition_data, gather_dp_data, get_dep_inputs, split_input_batch_by_dp, Batch
from examples.server.request import request_from_token


class TestGenerate(unittest.TestCase):
    def setUp(self):
        attn_dp = MagicMock()
        attn_dp.rank = 0
        attn_dp.group_size = 4

        attn_tp = MagicMock()
        attn_tp.rank = 0
        attn_tp.group_size = 1

        mapping = MagicMock()
        mapping.attn_dp = attn_dp
        mapping.attn_tp = attn_tp

        self.model = MagicMock()
        self.model.mapping = mapping

        self.batch_size = 4
        self.block_size = 128


    def test_partition_data(self):
        dp_rank_ids = torch.tensor([0, 1, 2, 3])
        dp_rank_ids_per_token = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])
        dp_rank_ids_per_token_sp = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])
        input_ids = torch.tensor([100000, 26503, 335, 279, 33396, 37576, 1593, 429, 39166,
            4403, 643, 895, 1377, 4712, 8254, 285, 33396, 433,
            9017, 418, 839, 3430, 276, 2622, 742, 2616, 11010,
            15821, 11, 5802, 1094, 12191, 536, 441, 463, 276,
            2622, 254, 11010, 3675, 9880, 4712, 13, 685, 207,
            17, 15, 15, 24, 11, 33396, 37576, 6972, 363,
            18, 13, 22, 19, 17, 10532, 881, 254, 2616,
            39696, 13, 79073, 280, 33396, 37576, 2622, 881, 9798,
            12178, 11, 285, 418, 4117, 19336, 327, 9798, 12178,
            7462, 2065, 20234, 13, 3159, 11, 657, 418, 25541,
            473, 254, 41198, 266, 12178, 47570, 13, 185, 23853,
            25, 317, 11010, 9880, 4712, 254, 1246, 372, 3613,
            5424, 30, 185, 32349, 25, 100000, 2640, 6, 82,
            4399, 4526, 30, 100000, 26503, 335, 279, 33396, 37576,
            1593, 429, 39166, 4403, 643, 895, 1377, 4712, 8254,
            285, 33396, 433, 9017, 418, 839, 3430, 276, 2622,
            742, 2616, 11010, 15821, 11, 5802, 1094, 12191, 536,
            441, 463, 276, 2622, 254, 11010, 3675, 9880, 4712,
            13, 685, 207, 17, 15, 15, 24, 11, 33396,
            37576, 6972, 363, 18, 13, 22, 19, 17, 10532,
            881, 254, 2616, 39696, 13, 79073, 280, 33396, 37576,
            2622, 881, 9798, 12178, 11, 285, 418, 4117, 19336,
            327, 9798, 12178, 7462, 2065, 20234, 13, 3159, 11,
            657, 418, 25541, 473, 254, 41198, 266, 12178, 47570,
            13, 185, 23853, 25, 317, 11010, 9880, 4712, 254,
            1246, 372, 3613, 5424, 30, 185, 32349, 25, 100000,
            2819, 418, 340, 30])
        position_ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
            28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
            42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
            56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
            70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
            84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97,
            98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
            112, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5,
            6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
            20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
            34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
            62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
            76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
            90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103,
            104, 105, 106, 107, 108, 109, 110, 111, 112, 0, 1, 2, 3, 4])
        is_prefill = True
        num_blocks = 10
        block_size = 128
        block_tables = torch.tensor([[0, 1],
            [0, 0],
            [0, 1],
            [0, 0]])
        slots = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
            28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
            42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
            56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
            70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
            84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97,
            98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
            112, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5,
            6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
            20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
            34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
            62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
            76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
            90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103,
            104, 105, 106, 107, 108, 109, 110, 111, 112, 0, 1, 2, 3, 4])
        input_lengths = torch.tensor([113, 7, 113, 5])
        input_lengths_sp = torch.tensor([113, 7, 113, 5])
        res = partition_data(self.model,
            dp_rank_ids,
            dp_rank_ids_per_token,
            dp_rank_ids_per_token_sp,
            input_ids,
            position_ids,
            is_prefill,
            num_blocks,
            block_size,
            block_tables,
            slots,
            input_lengths,
            input_lengths_sp)
        shard_input_ids, shard_position_ids, shard_is_prefill, shard_block_tables, \
            shard_slots, shard_input_lengths, _, shard_max_seq_len, shard_lm_head_indices, _, _ = res

        golden_input_ids = torch.tensor([100000, 26503, 335, 279, 33396, 37576, 1593, 429, 39166,
            4403, 643, 895, 1377, 4712, 8254, 285, 33396, 433,
            9017, 418, 839, 3430, 276, 2622, 742, 2616, 11010,
            15821, 11, 5802, 1094, 12191, 536, 441, 463, 276,
            2622, 254, 11010, 3675, 9880, 4712, 13, 685, 207,
            17, 15, 15, 24, 11, 33396, 37576, 6972, 363,
            18, 13, 22, 19, 17, 10532, 881, 254, 2616,
            39696, 13, 79073, 280, 33396, 37576, 2622, 881, 9798,
            12178, 11, 285, 418, 4117, 19336, 327, 9798, 12178,
            7462, 2065, 20234, 13, 3159, 11, 657, 418, 25541,
            473, 254, 41198, 266, 12178, 47570, 13, 185, 23853,
            25, 317, 11010, 9880, 4712, 254, 1246, 372, 3613,
            5424, 30, 185, 32349, 25])
        self.assertTrue(torch.equal(shard_input_ids, golden_input_ids))
        golden_position_ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
            28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
            42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
            56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
            70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
            84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97,
            98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
            112])
        self.assertTrue(torch.equal(shard_position_ids, golden_position_ids))
        self.assertTrue(shard_is_prefill)
        self.assertTrue(torch.equal(shard_block_tables, torch.tensor([[0, 1]])))
        golden_slots = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
            28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
            42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
            56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
            70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
            84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97,
            98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
            112])
        self.assertTrue(torch.equal(shard_slots, golden_slots))
        self.assertTrue(torch.equal(shard_input_lengths, torch.tensor([113])))
        self.assertEqual(shard_max_seq_len, 113)
        self.assertTrue(torch.equal(shard_lm_head_indices, torch.tensor([112, 119, 232, 237])))

    def test_partition_data_dummy(self):
        self.model.mapping.attn_dp.rank = 3
        dp_rank_ids = torch.tensor([0])
        dp_rank_ids_per_token = torch.tensor([0])
        dp_rank_ids_per_token_sp = torch.tensor([0])

        input_ids = torch.tensor([185])
        position_ids = torch.tensor([7])
        is_prefill = False
        num_blocks = 10
        block_size = 128
        block_tables = torch.tensor([[0]])
        slots = torch.tensor([7])
        input_lengths = torch.tensor([8])
        input_lengths_sp = torch.tensor([8])

        res = partition_data(self.model,
            dp_rank_ids,
            dp_rank_ids_per_token,
            dp_rank_ids_per_token_sp,
            input_ids,
            position_ids,
            is_prefill,
            num_blocks,
            block_size,
            block_tables,
            slots,
            input_lengths,
            input_lengths_sp)
        shard_input_ids, shard_position_ids, shard_is_prefill, shard_block_tables, \
            shard_slots, shard_input_lengths, _, shard_max_seq_len, shard_lm_head_indices, _, _ = res

        self.assertTrue(torch.equal(shard_input_ids, torch.tensor([1])))
        self.assertTrue(torch.equal(shard_position_ids, torch.tensor([0], dtype=torch.int32)))
        self.assertTrue(torch.equal(shard_block_tables, torch.tensor([[9]], dtype=torch.int32)))
        self.assertTrue(torch.equal(shard_slots, torch.tensor([1152], dtype=torch.int32)))
        self.assertTrue(torch.equal(shard_input_lengths, torch.tensor([1], dtype=torch.int32)))
        self.assertEqual(shard_max_seq_len, 1)
        self.assertTrue(torch.equal(shard_lm_head_indices, torch.tensor([0])))

    def test_gather_dp_data(self):
        dp_rank_ids_per_token = torch.tensor([0, 0, 0, 0, 1, 2, 2, 3, 3, 3, 3, 3])
        res = gather_dp_data(self.model, dp_rank_ids_per_token)
        golden = {
            'sum_token_size_per_dp_group': 12,
            'shard_effective_token_indices': torch.tensor([0, 1, 2, 3]).npu(),
            'token_index_with_padding': torch.tensor([0, 1, 2, 3, 0]).npu(),
            'skip_padding_token_indices': torch.tensor([0, 1, 2, 3, 5, 10, 11, 15, 16, 17, 18, 19]).npu()
        }
        self.assertEqual(res.get("sum_token_size_per_dp_group"), golden.get("sum_token_size_per_dp_group"))
        self.assertTrue(torch.equal(
            res.get("shard_effective_token_indices"), golden.get("shard_effective_token_indices")))
        self.assertTrue(torch.equal(res.get("token_index_with_padding"), golden.get("token_index_with_padding")))
        self.assertTrue(torch.equal(res.get("skip_padding_token_indices"), golden.get("skip_padding_token_indices")))

    def test_gather_dp_data_dummy(self):
        dp_rank_ids_per_token = torch.tensor([0, 0, 0, 0, 2, 2])
        res = gather_dp_data(self.model, dp_rank_ids_per_token)
        golden = {
            'sum_token_size_per_dp_group': 8,
            'shard_effective_token_indices': torch.tensor([0, 1, 2, 3]).npu(),
            'token_index_with_padding': torch.tensor([0, 1, 2, 3]).npu(),
            'skip_padding_token_indices': torch.tensor([0, 1, 2, 3, 4, 8, 9, 12]).npu()
        }
        self.assertEqual(res.get("sum_token_size_per_dp_group"), golden.get("sum_token_size_per_dp_group"))
        self.assertTrue(torch.equal(
            res.get("shard_effective_token_indices"), golden.get("shard_effective_token_indices")))
        self.assertTrue(torch.equal(res.get("token_index_with_padding"), golden.get("token_index_with_padding")))
        self.assertTrue(torch.equal(res.get("skip_padding_token_indices"), golden.get("skip_padding_token_indices")))


    @patch('examples.server.generate.MTP', 1)
    def test_get_dep_inputs(self):
        ENV.enable_dp_move_up = True
        cache_manager = MagicMock()
        cache_manager.block_size = self.block_size
        context_length = 1
        request_list = []
        for i in range(1, self.batch_size + 1):
            input_ids = request_from_token(np.ones(context_length, dtype=np.int64), 1, self.block_size, i - 1)
            input_ids.dp_rank = (i - 1) % self.model.mapping.attn_dp.group_size
            request_list.append(input_ids)
        input_batch = Batch(request_list)

        input_batch.batch_block_tables = np.array([[0, 1],
            [0, 0],
            [0, 1],
            [0, 0]])

        input_batch_dp, dp_rank_ids_per_token = split_input_batch_by_dp(self.model, cache_manager, input_batch)
        tensor_dep_inputs, token_size_per_dp_group_tensor, shard_effective_token_indices, max_dp_batch_size = \
            get_dep_inputs(self.model, input_batch, input_batch_dp, dp_rank_ids_per_token)

        self.assertTrue(torch.equal(
            tensor_dep_inputs[0].cpu(), torch.tensor([0])))
        self.assertTrue(torch.equal(
            shard_effective_token_indices.cpu(), torch.arange(1)))
        self.assertTrue(torch.equal(
            token_size_per_dp_group_tensor.cpu(), torch.ones(self.batch_size) * context_length))
        self.assertEqual(max_dp_batch_size, context_length)


if __name__ == "__main__":
    unittest.main()