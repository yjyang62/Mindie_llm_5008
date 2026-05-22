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

from atb_llm.models.base.model_input_managers import AttentionMaskGenerator, PositionEmbeddingGenerator, KVCacheUpdater
from atb_llm.utils.layers.embedding.position_rotary_embedding import PositionEmbeddingType
from tests.pythontest.atb_llm.models.base.test_flash_causal_lm_v3 import MockModelParam, MockModelStatus
from tests.pythontest.atb_llm.models.base.mock_class import MockTorchClasses


class TestAttentionMaskGenerator(unittest.TestCase):
    def setUp(self):
        torch.classes = MockTorchClasses()
        self.mindie_llm_config = MockModelParam()
        self.mindie_llm_config.soc_info.need_nz = False
        self.torch_device = torch.device("npu")
        model_status = MockModelStatus()
        self.attention_mask_manager = AttentionMaskGenerator(self.mindie_llm_config, model_status, self.torch_device)

    def test_generate_mask_prefill(self):
        with patch.object(self.attention_mask_manager.attention_mask_ins, 'get_rope_prefill_mask',
                          return_value=torch.tensor([[1, 0], [1, 1]])) as mock_method:
            mask = self.attention_mask_manager.generate_mask(None, is_prefill=True)
            mock_method.assert_called_once()
            self.assertIsInstance(mask, torch.Tensor)
    
    def test_generate_mask_decode(self):
        with patch.object(self.attention_mask_manager.attention_mask_ins, 'get_rope_decode_mask',
                          return_value=torch.tensor([[1, 0], [1, 1]])) as mock_method:
            mask = self.attention_mask_manager.generate_mask(None, is_prefill=False)
            mock_method.assert_called_once()
            self.assertIsInstance(mask, torch.Tensor)


@ddt
class TestPositionEmbeddingGenerator(unittest.TestCase):
    def setUp(self):
        self.mindie_llm_config = MockModelParam()
        self.torch_device = torch.device("npu")
        model_status = MockModelStatus()
        self.position_embedding_manager = PositionEmbeddingGenerator(self.mindie_llm_config, model_status, self.torch_device)
    
    @patch("atb_llm.models.base.model_input_managers.PositionRotaryEmbedding.get_cos_cached_total", return_value=None)
    @patch("atb_llm.models.base.model_input_managers.PositionRotaryEmbedding.get_sin_cached_total", return_value=None)
    @data(
        ("linear", "update_cos_sin_cache_total", False),
        ("llama3", "update_llama3_cos_sin_cache_total", False),
        ("yarn", "yarn_scaling_rotary_embedding", True),
        ("dynamic", "dynamic_ntk_rotary_embedding", True)
    )
    @unpack
    def test_generate_positional_embedding_rope(self, rope_type, expected_method, expected_placeholder,
                                                mock_get_sin, mock_get_cos):
        self.position_embedding_manager.position_embedding_type = PositionEmbeddingType.ROPE
        self.position_embedding_manager.rope_type = rope_type

        with patch.object(
            self.position_embedding_manager.position_embedding_ins,
            expected_method, return_value=None
        ) as mock_method:
            self.position_embedding_manager.generate_position_embedding(max_seq_len=128)
            mock_method.assert_called_once()
            
        if expected_placeholder:
            self.assertIs(self.position_embedding_manager.cosine_table, self.position_embedding_manager.placeholder)
            self.assertIs(self.position_embedding_manager.sine_table, self.position_embedding_manager.placeholder)
        else:
            self.assertIsNone(self.position_embedding_manager.cosine_table)
            self.assertIsNone(self.position_embedding_manager.sine_table)

    def test_generate_positional_embedding_alibi(self):
        self.position_embedding_manager.position_embedding_type = PositionEmbeddingType.ALIBI
        self.position_embedding_manager.generate_position_embedding(max_seq_len=128)
        self.assertIs(self.position_embedding_manager.cosine_table, self.position_embedding_manager.placeholder)
        self.assertIs(self.position_embedding_manager.sine_table, self.position_embedding_manager.placeholder)


class TestKVCacheUpdater(unittest.TestCase):
    def setUp(self):
        self.mindie_llm_config = MockModelParam()
        self.kv_cache_manager = KVCacheUpdater(self.mindie_llm_config)
        self.kv_cache_manager.soc_info.need_nz = True

    @patch("torch_npu.get_npu_format")
    @patch("torch_npu.npu_format_cast_", return_value=torch.rand(2, 3, 4, 5))
    def test_update_kv_cache(self, mock_cast, _):
        mock_engine_wrappers = [MagicMock()]

        kv_cache = [
            (torch.rand(2, 3, 4, 5), torch.rand(2, 3, 4, 5)), (torch.rand(2, 3, 4, 5), torch.rand(2, 3, 4, 5))
        ]
        self.kv_cache_manager.update_kv_cache(kv_cache, mock_engine_wrappers)
        mock_cast.assert_called()
        mock_engine_wrappers[0].set_kv_caches.assert_called_once()


if __name__ == "__main__":
    unittest.main()