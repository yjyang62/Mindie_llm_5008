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
from unittest.mock import MagicMock, patch

import torch

from atb_llm.layers.embedding.word_embedding import ParallelEmbedding
from atb_llm.utils.loader.safetensor_file_loader import SafetensorFileLoader
from atb_llm.models.base.config import BaseConfig
from atb_llm.nn.tensor import Tensor


class TestWordEmbedding(unittest.TestCase):
    def setUp(self):
        self.config = BaseConfig(torch_dtype=torch.float16)

    @patch("atb_llm.layers.embedding.word_embedding")
    def test_word_embedding(self, mock_ops):
        weight_tool_cls = MagicMock(spec=SafetensorFileLoader)
        mock_weight_tool_obj = weight_tool_cls()
        embedding_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float16)
        mock_weight_tool_obj.get_tensor = MagicMock(return_value=embedding_tensor)
        mapping = MagicMock()
        module = ParallelEmbedding(self.config, mock_weight_tool_obj, "word_embedding", mapping)
        input_ids = Tensor("input_ids")
        golden_output = Tensor()
        mock_ops.gather.return_value = golden_output
        out = module.forward(input_ids)
        self.assertEqual(golden_output, out)
    
    def test_parallel_embedding(self):
        weight_tool_cls = MagicMock(spec=SafetensorFileLoader)
        mock_weight_tool_obj = weight_tool_cls()
        embedding_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float16)
        mock_weight_tool_obj.get_sharded = MagicMock(return_value=embedding_tensor)
        mapping = MagicMock()
        _ = ParallelEmbedding(self.config, mock_weight_tool_obj, "word_embedding",
                                   mapping, parallel_embedding=True)


if __name__ == '__main__':
    unittest.main()