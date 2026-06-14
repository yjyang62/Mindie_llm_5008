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

from examples.server.batch import Batch
from examples.server.request import Request


class TestBatch(unittest.TestCase):
    def setUp(self):
        self.req1 = Request(
            max_out_length=20,
            block_size=128,
            req_id=0,
            input_ids=torch.tensor(list(range(4)), dtype=torch.int64),
            adapter_id='0'
        )
        
        self.req2 = Request(
            max_out_length=20,
            block_size=128,
            req_id=1,
            input_ids=torch.tensor(list(range(4, 8)), dtype=torch.int64),
            adapter_id='0'
        )
        
        self.req_list = [self.req1, self.req2]
        self.block_tables = torch.tensor([[0, 1], [0, 0]])

    @patch('examples.server.batch.ENV.deepseek_mtp', True)    
    def test_init_mtp(self):
        batch = Batch(self.req_list)
        golden_batch_input_ids_mtp_list = torch.tensor([
            1, 2, 3, 0, 5, 6, 7, 4
        ])
        self.assertTrue(torch.equal(batch.mtp.batch_input_ids_mtp, golden_batch_input_ids_mtp_list))

    @patch('examples.server.batch.ENV.deepseek_mtp', True)
    def test_concatenate_mtp(self):
        batch1 = Batch([self.req1])
        batch1.mtp.block_tables = torch.tensor([[0, 1]])
        batch1.batch_block_tables = torch.tensor([[0, 1]])
        batch1.batch_slots_tables = torch.tensor(list(range(4)), dtype=torch.int64)
        batch1.mtp.accepted_lens = torch.zeros(batch1.mtp.block_tables.size(0), dtype=torch.int64)
        batch2 = Batch([self.req2])
        batch2.mtp.block_tables = torch.tensor([[0, 0]])
        batch2.batch_block_tables = torch.tensor([[0, 0]])
        batch2.batch_slots_tables = torch.tensor(list(range(4)), dtype=torch.int64)
        batch2.mtp.accepted_lens = torch.zeros(batch2.mtp.block_tables.size(0), dtype=torch.int64)
        batches = [batch1, batch2]
        Batch.concatenate(batches)
        golden_batch_input_ids_mtp_list = torch.tensor([
            1, 2, 3, 0, 5, 6, 7, 4
        ])
        golden_accepted_lens = torch.tensor([0, 0], dtype=torch.int64)
        self.assertEqual(len(batches), 1)
        self.assertTrue(torch.equal(batches[0].mtp.batch_input_ids_mtp, golden_batch_input_ids_mtp_list))
        self.assertTrue(torch.equal(batches[0].mtp.accepted_lens, golden_accepted_lens))
    
    @patch('examples.server.batch.ENV.deepseek_mtp', True)
    def test_filter_mtp(self):
        batch = Batch(self.req_list)
        batch.req_list[0].block_tables = torch.tensor([[0, 1], [0, 0]])
        batch.req_list[1].block_tables = torch.tensor([[0, 1], [0, 0]])
        batch.req_list[0].slot_tables = torch.tensor(list(range(4)), dtype=torch.int64)
        batch.req_list[1].slot_tables = torch.tensor(list(range(4)), dtype=torch.int64)
        batch.mtp.block_tables = torch.tensor([[0, 1], [0, 0]])
        batch.batch_block_tables = torch.tensor([[0, 1], [0, 0]])
        batch.req_list[0].out_token_list = list(range(20))
        batch.mtp.accepted_lens = torch.zeros(batch.mtp.block_tables.size(0), dtype=torch.int64)
        mock_postprocessor = MagicMock()
        mock_postprocessor.stopping_criteria.return_value = False
        mock_postprocessor.max_new_tokens = 20

        cache_manger = MagicMock()

        finish_num = batch.filter(mock_postprocessor, cache_manger)
        self.assertEqual(finish_num, 1)


if __name__ == "__main__":
    unittest.main()