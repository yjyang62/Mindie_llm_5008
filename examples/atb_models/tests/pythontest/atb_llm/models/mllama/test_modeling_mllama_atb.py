# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import torch

path = os.getenv('ATB_SPEED_HOME_PATH')
sys.path.append(os.path.join(path, 'lib'))

from atb_llm.models.mllama.modeling_mllama_atb import MllamaModelATB
from atb_llm.models.mllama.config_mllama import MllamaTextConfig
from atb_llm.utils.dist import FakeGroup


class TestFlashMllamaForCausal(unittest.TestCase):
    def setUp(self):
        self.config = MllamaTextConfig(
            max_position_embeddings=1024,
        )
        self.weights = MagicMock()
        self.dtype = torch.bfloat16
        self.weights.device = torch.device("npu")
        self.weights.dtype = self.dtype
        self.weights.get_tensor.return_value = torch.empty(100, 100, dtype=self.dtype)
        self.weights.get_partial_sharded.return_value = torch.empty(100, 100, dtype=self.dtype)
        self.weights.get_whole_tensor.return_value = torch.empty(100, 100, dtype=self.dtype)
        self.weights.get_multi_weights_col.return_value = torch.empty(100, 100, dtype=self.dtype)
        self.weights.get_replicated_weights.return_value = torch.empty(100, 100, dtype=self.dtype)
        self.weights.get_multi_weights_row.return_value = torch.empty(100, 100, dtype=self.dtype)
        self.weights.get_shape.return_value = (100, 100)
        self.weights.process_group = FakeGroup(1, 1)
    
    @patch("_libatb_torch.BaseOperation")
    @patch('atb_llm.models.llama.modeling_llama_atb.AtbGraph')
    @patch('atb_llm.models.mllama.modeling_mllama_atb.AtbGraph')
    @patch('atb_llm.models.base.modeling_atb.TensorParallelEmbedding')
    def test_init_mllama_model(self, mock_embedding, mock_mllama_atb_graph, mock_llama_atb_graph, mock_base_operation):

        mock_embedding_obj = mock_embedding.return_value
        mock_embedding_obj.get_weights.return_value = dict()

        mock_mllama_atb_graph_obj = mock_mllama_atb_graph.return_value
        mock_mllama_atb_graph_obj.add_operation.return_value = None
        mock_mllama_atb_graph_obj.build.return_value = None

        mock_llama_atb_graph_obj = mock_llama_atb_graph.return_value
        mock_llama_atb_graph_obj.add_operation.return_value = None
        mock_llama_atb_graph_obj.build.return_value = None

        mllama_model = MllamaModelATB(self.config, self.weights)
        mllama_model.get_weights('model')
        mllama_model.build_graph(mock_mllama_atb_graph_obj, True, True)

        mock_embedding.assert_called_once_with(
            prefix="model.embed_tokens",
            weights=self.weights
        )
        self.assertEqual(mock_mllama_atb_graph.call_count, 8)
        self.assertEqual(mock_llama_atb_graph.call_count, 32)
        mock_base_operation.assert_called()


if __name__ == '__main__':
    unittest.main()