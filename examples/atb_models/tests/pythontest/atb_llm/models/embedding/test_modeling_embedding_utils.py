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
from unittest import TestCase

import torch
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.xlm_roberta.configuration_xlm_roberta import XLMRobertaConfig
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaModel
from torch import nn

from atb_llm.models.embedding.modeling_embedding_utils import (
    create_position_ids_from_input_ids,
    create_padded_output,
    KVAttentionManager,
    BaseEmbeddings,
    BaseSelfAttention,
    BaseSelfOutput,
    BaseAttention,
    BaseIntermediate,
    BaseOutput,
    BaseLayer,
    BaseEncoder,
    BasePooler,
    BaseModel,
    BasePreTrainedModel,
    BaseForSequenceClassification
)

FAKE_BERT_CONFIG = BertConfig(
    vocab_size=1000,
    hidden_size=256,
    num_hidden_layers=2,
    num_attention_heads=8,
    intermediate_size=512,
    max_position_embeddings=512,
    pad_token_id=0,
)

FAKE_XLM_CONFIG = XLMRobertaConfig(
    vocab_size=1000,
    hidden_size=256,
    num_hidden_layers=2,
    num_attention_heads=8,
    intermediate_size=512,
    max_position_embeddings=512,
    pad_token_id=0,
)


class SimpleModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear = torch.nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, input_ids, attention_mask):
        return self.linear(input_ids)


class TestModelingEmbeddingUtils(TestCase):
    def setUp(self):
        self.config_bert = FAKE_BERT_CONFIG
        self.config_xlm = FAKE_XLM_CONFIG
        self.weights = MagicMock()
        self.weights.device = torch.device("npu")
        self.weights.dtype = torch.float16
        self.weights.get_tensor.return_value = torch.empty(100, 100, dtype=torch.bfloat16)
        self.weights.get_multi_weights_col.return_value = torch.empty(100, 100, dtype=torch.bfloat16)
        self.weights.get_replicated_weights.return_value = torch.empty(100, 100, dtype=torch.bfloat16)
        self.weights.get_multi_weights_row.return_value = torch.empty(100, 100, dtype=torch.bfloat16)
        self.weights.get_whole_tensor.return_value = torch.empty(100, 100, dtype=torch.bfloat16)
        self.weights.get_shape.return_value = [100, 100]
        self.weights.switch_process_group.return_value = None

    def test_create_position_ids_from_input_ids(self):
        input_ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]])
        padding_idx = 0
        past_key_values_length = 0
        position_ids = create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length)
        expected_position_ids = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 3, 4, 5]])
        self.assertTrue(torch.equal(position_ids, expected_position_ids))

    def test_create_padded_output(self):
        encoder_outputs = torch.tensor([[[1, 2], [3, 4], [5, 6]]])
        nonzero_seq_len = torch.tensor([3])
        padded_encoder_outputs = create_padded_output(encoder_outputs, nonzero_seq_len)
        expected_padded_encoder_outputs = torch.tensor([[[1, 2], [3, 4], [5, 6]]])
        self.assertTrue(torch.equal(padded_encoder_outputs, expected_padded_encoder_outputs))

    def test_kv_attention_manager(self):
        input_shape = (2, 5)
        kv_attention_manager = KVAttentionManager(self.config_bert, input_shape)
        self.assertEqual(kv_attention_manager.batch_size, 2)
        self.assertEqual(kv_attention_manager.seq_len, 5)

    def test_kv_attention_manager_with_different_input_shape(self):
        input_shape = (1, 10)
        kv_attention_manager = KVAttentionManager(self.config_bert, input_shape)
        self.assertEqual(kv_attention_manager.batch_size, 1)
        self.assertEqual(kv_attention_manager.seq_len, 10)

    def test_kv_attention_manager_with_different_device(self):
        input_shape = (2, 5)
        kv_attention_manager = KVAttentionManager(self.config_bert, input_shape)
        kv_attention_manager.device = torch.device("cpu")
        self.assertEqual(kv_attention_manager.device, torch.device("cpu"))

    def test_kv_attention_manager_with_different_config(self):
        config = FAKE_BERT_CONFIG
        input_shape = (2, 5)
        kv_attention_manager = KVAttentionManager(config, input_shape)
        self.assertEqual(kv_attention_manager.batch_size, 2)
        self.assertEqual(kv_attention_manager.seq_len, 5)

    def test_kv_attention_manager_with_different_config_and_input_shape(self):
        config = FAKE_BERT_CONFIG
        input_shape = (3, 7)
        kv_attention_manager = KVAttentionManager(config, input_shape)
        self.assertEqual(kv_attention_manager.batch_size, 3)
        self.assertEqual(kv_attention_manager.seq_len, 7)

    def test_kv_attention_manager_get_seq_len_list_and_tensor(self):
        input_shape = (2, 5)
        kv_attention_manager = KVAttentionManager(self.config_bert, input_shape)
        kv_attention_manager.init_seq_len_and_token_offset(input_shape[1])
        kv_attention_manager.is_full = True
        self.assertIsNotNone(kv_attention_manager.seq_len_list)
        self.assertIsNotNone(kv_attention_manager.seq_len_tensor)
        kv_attention_manager.is_full = False
        self.assertIsNotNone(kv_attention_manager.seq_len_list)
        self.assertIsNotNone(kv_attention_manager.seq_len_tensor)
        self.assertIsNotNone(kv_attention_manager.token_offset_list)

    def test_kv_attention_manager_init_attention_mask(self):
        input_shape = (2, 5)
        kv_attention_manager = KVAttentionManager(self.config_bert, input_shape)
        kv_attention_manager.init_attention_mask()

    def test_kv_attention_manager_get_attention_mask(self):
        input_shape = (2, 5)
        attention_mask = torch.ones(1, 3, dtype=torch.float)
        kv_attention_manager = KVAttentionManager(self.config_bert, input_shape)
        kv_attention_manager.is_full = False
        return_attention_mask = kv_attention_manager.get_attention_mask(attention_mask)
        self.assertIsNotNone(return_attention_mask)

    def test_base_embeddings(self):
        base_embeddings = BaseEmbeddings(self.config_bert)
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        token_type_ids = torch.tensor([[0, 0, 1], [0, 1, 1]])
        position_ids = torch.tensor([[1, 2, 3], [1, 2, 3]])
        embeddings = base_embeddings(input_ids, token_type_ids, position_ids)
        self.assertIsNotNone(embeddings)

    def test_base_self_attention(self):
        base_self_attention = BaseSelfAttention(self.config_bert)
        hidden_states = torch.randn(1, 3, self.config_bert.hidden_size, dtype=torch.float)
        attention_mask = torch.ones(1, 3, dtype=torch.float)
        outputs = base_self_attention(hidden_states, attention_mask)
        self.assertIsNotNone(outputs)

    def test_base_self_attention_with_different_hidden_size(self):
        config = FAKE_BERT_CONFIG
        base_self_attention = BaseSelfAttention(config)
        hidden_states = torch.randn(1, 3, config.hidden_size, dtype=torch.float)
        attention_mask = torch.ones(1, 3, dtype=torch.float)
        outputs = base_self_attention(hidden_states, attention_mask)
        self.assertIsNotNone(outputs)

    def test_base_self_attention_with_different_device(self):
        config = FAKE_BERT_CONFIG
        base_self_attention = BaseSelfAttention(config)
        base_self_attention.device = torch.device("cpu")
        hidden_states = torch.randn(1, 3, config.hidden_size, dtype=torch.float)
        attention_mask = torch.ones(1, 3, dtype=torch.float)
        outputs = base_self_attention(hidden_states, attention_mask)
        self.assertIsNotNone(outputs)

    def test_base_self_attention_with_different_hidden_states(self):
        config = FAKE_BERT_CONFIG
        base_self_attention = BaseSelfAttention(config)
        hidden_states = torch.randn(1, 3, config.hidden_size, dtype=torch.float)
        attention_mask = torch.ones(1, 3, dtype=torch.float)
        outputs = base_self_attention(hidden_states, attention_mask)
        self.assertIsNotNone(outputs)

    def test_base_self_attention_with_different_config_and_hidden_states(self):
        config = FAKE_BERT_CONFIG
        base_self_attention = BaseSelfAttention(config)
        hidden_states = torch.randn(1, 3, config.hidden_size, dtype=torch.float)
        attention_mask = torch.ones(1, 3, dtype=torch.float)
        outputs = base_self_attention(hidden_states, attention_mask)
        self.assertIsNotNone(outputs)

    def test_base_self_attention_with_different_config_and_hidden_states_and_device(self):
        config = FAKE_BERT_CONFIG
        base_self_attention = BaseSelfAttention(config)
        base_self_attention.device = torch.device("cpu")
        hidden_states = torch.randn(1, 3, config.hidden_size, dtype=torch.float)
        attention_mask = torch.ones(1, 3, dtype=torch.float)
        outputs = base_self_attention(hidden_states, attention_mask)
        self.assertIsNotNone(outputs)

    def test_base_self_attention_with_different_hidden_states_and_attention_mask(self):
        config = FAKE_BERT_CONFIG
        base_self_attention = BaseSelfAttention(config)
        hidden_states = torch.randn(1, 3, config.hidden_size, dtype=torch.float)
        attention_mask = torch.ones(1, 3, dtype=torch.float)
        outputs = base_self_attention(hidden_states, attention_mask)
        self.assertIsNotNone(outputs)

    def test_base_self_attention_with_different_hidden_states_and_no_attention_mask(self):
        config = FAKE_BERT_CONFIG
        base_self_attention = BaseSelfAttention(config)
        hidden_states = torch.randn(1, 3, config.hidden_size, dtype=torch.float)
        outputs = base_self_attention(hidden_states)
        self.assertIsNotNone(outputs)

    def test_base_self_output(self):
        base_self_output = BaseSelfOutput(self.config_bert)
        hidden_states = torch.randn(1, 3, self.config_bert.hidden_size, dtype=torch.float)
        input_tensor = torch.randn(1, 3, self.config_bert.hidden_size, dtype=torch.float)
        outputs = base_self_output(hidden_states, input_tensor)
        self.assertIsNotNone(outputs)

    def test_base_attention(self):
        base_attention = BaseAttention(self.config_bert)
        hidden_states = torch.randn(1, 3, self.config_bert.hidden_size, dtype=torch.float)
        attention_mask = torch.ones(1, 3, dtype=torch.float)
        outputs = base_attention(hidden_states, attention_mask)
        self.assertIsNotNone(outputs)

    def test_base_intermediate(self):
        base_intermediate = BaseIntermediate(self.config_bert)
        hidden_states = torch.randn(1, 3, self.config_bert.hidden_size, dtype=torch.float)
        outputs = base_intermediate(hidden_states)
        self.assertIsNotNone(outputs)

    def test_base_output(self):
        base_output = BaseOutput(self.config_bert)
        hidden_states = torch.randn(1, 3, 512, dtype=torch.float)  # hidden_size = 512
        input_tensor = torch.randn(1, 3, 256, dtype=torch.float)  # hidden_size = 256
        outputs = base_output(hidden_states, input_tensor)
        self.assertIsNotNone(outputs)

    def test_base_layer(self):
        base_layer = BaseLayer(self.config_bert)
        hidden_states = torch.randn(1, 3, self.config_bert.hidden_size, dtype=torch.float)
        attention_mask = torch.ones(1, 3, dtype=torch.float)
        outputs = base_layer(hidden_states, attention_mask)
        self.assertIsNotNone(outputs)

    def test_base_encoder(self):
        base_encoder = BaseEncoder(self.config_bert)
        hidden_states = torch.randn(1, 3, self.config_bert.hidden_size, dtype=torch.float)
        attention_mask = torch.ones(1, 3, dtype=torch.float)
        outputs = base_encoder(hidden_states, attention_mask)
        self.assertIsNotNone(outputs)

    def test_base_pooler(self):
        base_pooler = BasePooler(self.config_bert)
        hidden_states = torch.randn(1, 3, self.config_bert.hidden_size, dtype=torch.float)
        outputs = base_pooler(hidden_states)
        self.assertIsNotNone(outputs)

    def test_base_pretrained_model(self):
        base_pretrained_model = BasePreTrainedModel(self.config_bert)
        self.assertIsNotNone(base_pretrained_model)

    def test_base_pretrained_model_init_weights(self):
        base_pretrained_model = BasePreTrainedModel(self.config_bert)
        module = nn.Linear(10, 10)
        base_pretrained_model._init_weights(module)
        module = nn.Embedding(10, 10)
        base_pretrained_model._init_weights(module)
        module = nn.LayerNorm(10, 1e-5)
        base_pretrained_model._init_weights(module)

    def test_base_model(self):
        base_model = SimpleModel(self.config_bert)
        input_ids = torch.randn(1, 3, self.config_bert.hidden_size, dtype=torch.float)
        attention_mask = torch.ones(1, 3, dtype=torch.float)
        outputs = base_model(input_ids, attention_mask)
        self.assertIsNotNone(outputs)

    def test_base_for_sequence_classification(self):
        base_for_sequence_classification = BaseForSequenceClassification(self.config_bert)
        self.assertIsNotNone(base_for_sequence_classification)

    def test_base_attention_with_different_config(self):
        config = FAKE_BERT_CONFIG
        base_attention = BaseAttention(config)
        hidden_states = torch.randn(1, 3, config.hidden_size, dtype=torch.float)
        attention_mask = torch.ones(1, 3, dtype=torch.float)
        outputs = base_attention(hidden_states, attention_mask)
        self.assertIsNotNone(outputs)

    def test_base_encoder_with_different_config(self):
        config = FAKE_BERT_CONFIG
        base_encoder = BaseEncoder(config)
        hidden_states = torch.randn(1, 3, config.hidden_size, dtype=torch.float)
        attention_mask = torch.ones(1, 3, dtype=torch.float)
        outputs = base_encoder(hidden_states, attention_mask)
        self.assertIsNotNone(outputs)

    def test_base_model_with_different_config(self):
        config = FAKE_BERT_CONFIG
        base_model = SimpleModel(config)
        input_ids = torch.randn(1, 3, config.hidden_size, dtype=torch.float)
        attention_mask = torch.ones(1, 3, dtype=torch.float)
        outputs = base_model(input_ids, attention_mask)
        self.assertIsNotNone(outputs)

    def test_create_position_ids_from_input_ids_with_padding(self):
        input_ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]])
        padding_idx = 0
        past_key_values_length = 2
        position_ids = create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length)
        expected_position_ids = torch.tensor([[3, 4, 5, 0, 0], [3, 4, 5, 6, 7]])
        self.assertTrue(torch.equal(position_ids, expected_position_ids))

    def test_create_padded_output_with_different_seq_len(self):
        encoder_outputs = torch.tensor([[[1, 2], [3, 4], [5, 6]]], dtype=torch.float)
        nonzero_seq_len = torch.tensor([3])
        padded_encoder_outputs = create_padded_output(encoder_outputs, nonzero_seq_len)
        expected_padded_encoder_outputs = torch.tensor([[[1, 2], [3, 4], [5, 6]]], dtype=torch.float)
        self.assertTrue(torch.equal(padded_encoder_outputs, expected_padded_encoder_outputs))

    def test_base_embeddings_with_different_config(self):
        config = FAKE_BERT_CONFIG
        base_embeddings = BaseEmbeddings(config)
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        token_type_ids = torch.tensor([[0, 0, 1], [0, 1, 1]])
        position_ids = torch.tensor([[1, 2, 3], [1, 2, 3]])
        embeddings = base_embeddings(input_ids, token_type_ids, position_ids)
        self.assertIsNotNone(embeddings)

    def test_base_self_output_with_different_hidden_size(self):
        config = FAKE_BERT_CONFIG
        base_self_output = BaseSelfOutput(config)
        hidden_states = torch.randn(1, 3, config.hidden_size, dtype=torch.float)
        input_tensor = torch.randn(1, 3, config.hidden_size, dtype=torch.float)
        outputs = base_self_output(hidden_states, input_tensor)
        self.assertIsNotNone(outputs)

    def test_base_attention_with_different_hidden_size(self):
        config = FAKE_BERT_CONFIG
        base_attention = BaseAttention(config)
        hidden_states = torch.randn(1, 3, config.hidden_size, dtype=torch.float)
        attention_mask = torch.ones(1, 3, dtype=torch.float)
        outputs = base_attention(hidden_states, attention_mask)
        self.assertIsNotNone(outputs)

    def test_base_intermediate_with_different_hidden_size(self):
        config = FAKE_BERT_CONFIG
        base_intermediate = BaseIntermediate(config)
        hidden_states = torch.randn(1, 3, config.hidden_size, dtype=torch.float)
        outputs = base_intermediate(hidden_states)
        self.assertIsNotNone(outputs)

    def test_base_output_with_different_hidden_size(self):
        config = FAKE_BERT_CONFIG
        base_output = BaseOutput(config)
        hidden_states = torch.randn(1, 3, 512, dtype=torch.float)  # hidden_size = 512
        input_tensor = torch.randn(1, 3, 256, dtype=torch.float)  # hidden_size = 256
        outputs = base_output(hidden_states, input_tensor)
        self.assertIsNotNone(outputs)

    def test_base_layer_with_different_hidden_size(self):
        config = FAKE_BERT_CONFIG
        base_layer = BaseLayer(config)
        hidden_states = torch.randn(1, 3, config.hidden_size, dtype=torch.float)
        attention_mask = torch.ones(1, 3, dtype=torch.float)
        outputs = base_layer(hidden_states, attention_mask)
        self.assertIsNotNone(outputs)

    def test_base_encoder_with_different_hidden_size(self):
        config = FAKE_BERT_CONFIG
        base_encoder = BaseEncoder(config)
        hidden_states = torch.randn(1, 3, config.hidden_size, dtype=torch.float)
        attention_mask = torch.ones(1, 3, dtype=torch.float)
        outputs = base_encoder(hidden_states, attention_mask)
        self.assertIsNotNone(outputs)

    def test_base_pooler_with_different_hidden_size(self):
        config = FAKE_BERT_CONFIG
        base_pooler = BasePooler(config)
        hidden_states = torch.randn(1, 3, config.hidden_size, dtype=torch.float)
        outputs = base_pooler(hidden_states)
        self.assertIsNotNone(outputs)

    def test_base_pretrained_model_with_different_config(self):
        config = FAKE_BERT_CONFIG
        base_pretrained_model = BasePreTrainedModel(config)
        self.assertIsNotNone(base_pretrained_model)

    def test_base_model_with_different_hidden_size(self):
        config = FAKE_BERT_CONFIG
        base_model = SimpleModel(config)
        input_ids = torch.randn(1, 3, config.hidden_size, dtype=torch.float)
        attention_mask = torch.ones(1, 3, dtype=torch.float)
        outputs = base_model(input_ids, attention_mask)
        self.assertIsNotNone(outputs)

    def test_base_for_sequence_classification_with_different_config(self):
        config = FAKE_BERT_CONFIG
        base_for_sequence_classification = BaseForSequenceClassification(config)
        self.assertIsNotNone(base_for_sequence_classification)

    def test_create_position_ids_from_input_ids_with_different_padding_idx(self):
        input_ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]])
        padding_idx = 0
        past_key_values_length = 0
        position_ids = create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length)
        expected_position_ids = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 3, 4, 5]])
        self.assertTrue(torch.equal(position_ids, expected_position_ids))

    def test_base_embeddings_with_different_device(self):
        config = FAKE_BERT_CONFIG
        base_embeddings = BaseEmbeddings(config)
        base_embeddings.device = torch.device("cpu")
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        token_type_ids = torch.tensor([[0, 0, 1], [0, 1, 1]])
        position_ids = torch.tensor([[1, 2, 3], [1, 2, 3]])
        embeddings = base_embeddings(input_ids, token_type_ids, position_ids)
        self.assertIsNotNone(embeddings)

    def test_base_self_output_with_different_device(self):
        config = FAKE_BERT_CONFIG
        base_self_output = BaseSelfOutput(config)
        base_self_output.device = torch.device("cpu")
        hidden_states = torch.randn(1, 3, config.hidden_size, dtype=torch.float)
        input_tensor = torch.randn(1, 3, config.hidden_size, dtype=torch.float)
        outputs = base_self_output(hidden_states, input_tensor)
        self.assertIsNotNone(outputs)

    def test_base_attention_with_different_device(self):
        config = FAKE_BERT_CONFIG
        base_attention = BaseAttention(config)
        base_attention.device = torch.device("cpu")
        hidden_states = torch.randn(1, 3, config.hidden_size, dtype=torch.float)
        attention_mask = torch.ones(1, 3, dtype=torch.float)
        outputs = base_attention(hidden_states, attention_mask)
        self.assertIsNotNone(outputs)

    def test_base_intermediate_with_different_device(self):
        config = FAKE_BERT_CONFIG
        base_intermediate = BaseIntermediate(config)
        base_intermediate.device = torch.device("cpu")
        hidden_states = torch.randn(1, 3, config.hidden_size, dtype=torch.float)
        outputs = base_intermediate(hidden_states)
        self.assertIsNotNone(outputs)

    def test_base_output_with_different_device(self):
        config = FAKE_BERT_CONFIG
        base_output = BaseOutput(config)
        base_output.device = torch.device("cpu")
        hidden_states = torch.randn(1, 3, 512, dtype=torch.float)  # hidden_size = 512
        input_tensor = torch.randn(1, 3, 256, dtype=torch.float)  # hidden_size = 256
        outputs = base_output(hidden_states, input_tensor)
        self.assertIsNotNone(outputs)

    def test_base_layer_with_different_device(self):
        config = FAKE_BERT_CONFIG
        base_layer = BaseLayer(config)
        base_layer.device = torch.device("cpu")
        hidden_states = torch.randn(1, 3, config.hidden_size, dtype=torch.float)
        attention_mask = torch.ones(1, 3, dtype=torch.float)
        outputs = base_layer(hidden_states, attention_mask)
        self.assertIsNotNone(outputs)

    def test_base_encoder_with_different_device(self):
        config = FAKE_BERT_CONFIG
        base_encoder = BaseEncoder(config)
        base_encoder.device = torch.device("cpu")
        hidden_states = torch.randn(1, 3, config.hidden_size, dtype=torch.float)
        attention_mask = torch.ones(1, 3, dtype=torch.float)
        outputs = base_encoder(hidden_states, attention_mask)
        self.assertIsNotNone(outputs)

    def test_base_pooler_with_different_device(self):
        config = FAKE_BERT_CONFIG
        base_pooler = BasePooler(config)
        base_pooler.device = torch.device("cpu")
        hidden_states = torch.randn(1, 3, config.hidden_size, dtype=torch.float)
        outputs = base_pooler(hidden_states)
        self.assertIsNotNone(outputs)

    def test_base_pretrained_model_with_different_device(self):
        config = FAKE_BERT_CONFIG
        base_pretrained_model = BasePreTrainedModel(config)
        base_pretrained_model.to(torch.device("cpu"))
        self.assertIsNotNone(base_pretrained_model)


    def test_base_model_with_different_device(self):
        config = FAKE_BERT_CONFIG
        base_model = SimpleModel(config)
        base_model.device = torch.device("cpu")
        input_ids = torch.randn(1, 3, config.hidden_size, dtype=torch.float)
        attention_mask = torch.ones(1, 3, dtype=torch.float)
        outputs = base_model(input_ids, attention_mask)
        self.assertIsNotNone(outputs)

    def test_base_for_sequence_classification_with_different_device(self):
        config = FAKE_BERT_CONFIG
        base_for_sequence_classification = BaseForSequenceClassification(config)
        base_for_sequence_classification.to(torch.device("cpu"))
        self.assertIsNotNone(base_for_sequence_classification)

    def test_create_position_ids_from_input_ids_with_different_past_key_values_length(self):
        input_ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]])
        padding_idx = 0
        past_key_values_length = 3
        position_ids = create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length)
        expected_position_ids = torch.tensor([[4, 5, 6, 0, 0], [4, 5, 6, 7, 8]])
        self.assertTrue(torch.equal(position_ids, expected_position_ids))

    def test_base_embeddings_with_different_input_ids(self):
        config = FAKE_BERT_CONFIG
        base_embeddings = BaseEmbeddings(config)
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        token_type_ids = torch.tensor([[0, 0, 1], [0, 1, 1], [1, 1, 0]])
        position_ids = torch.tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
        embeddings = base_embeddings(input_ids, token_type_ids, position_ids)
        self.assertIsNotNone(embeddings)

    def test_base_self_output_with_different_hidden_states(self):
        config = FAKE_BERT_CONFIG
        base_self_output = BaseSelfOutput(config)
        hidden_states = torch.randn(2, 4, config.hidden_size, dtype=torch.float)
        input_tensor = torch.randn(2, 4, config.hidden_size, dtype=torch.float)
        outputs = base_self_output(hidden_states, input_tensor)
        self.assertIsNotNone(outputs)

    def test_base_attention_with_different_hidden_states(self):
        config = FAKE_BERT_CONFIG
        base_attention = BaseAttention(config)
        hidden_states = torch.randn(1, 3, config.hidden_size, dtype=torch.float)
        attention_mask = torch.ones(1, 3, dtype=torch.float)
        outputs = base_attention(hidden_states, attention_mask)
        self.assertIsNotNone(outputs)

    def test_base_intermediate_with_different_hidden_states(self):
        config = FAKE_BERT_CONFIG
        base_intermediate = BaseIntermediate(config)
        hidden_states = torch.randn(2, 4, config.hidden_size, dtype=torch.float)
        outputs = base_intermediate(hidden_states)
        self.assertIsNotNone(outputs)

    def test_base_output_with_different_hidden_states(self):
        config = FAKE_BERT_CONFIG
        base_output = BaseOutput(config)
        hidden_states = torch.randn(2, 4, 512, dtype=torch.float)  # hidden_size = 512
        input_tensor = torch.randn(2, 4, 256, dtype=torch.float)  # hidden_size = 256
        outputs = base_output(hidden_states, input_tensor)
        self.assertIsNotNone(outputs)

    def test_base_layer_with_different_hidden_states(self):
        config = FAKE_BERT_CONFIG
        base_layer = BaseLayer(config)
        hidden_states = torch.randn(1, 3, config.hidden_size, dtype=torch.float)
        attention_mask = torch.ones(1, 3, dtype=torch.float)
        outputs = base_layer(hidden_states, attention_mask)
        self.assertIsNotNone(outputs)

    def test_base_encoder_with_different_hidden_states(self):
        config = FAKE_BERT_CONFIG
        base_encoder = BaseEncoder(config)
        hidden_states = torch.randn(1, 3, config.hidden_size, dtype=torch.float)
        attention_mask = torch.ones(1, 3, dtype=torch.float)
        outputs = base_encoder(hidden_states, attention_mask)
        self.assertIsNotNone(outputs)

    def test_base_pooler_with_different_hidden_states(self):
        config = FAKE_BERT_CONFIG
        base_pooler = BasePooler(config)
        hidden_states = torch.randn(2, 4, config.hidden_size, dtype=torch.float)
        outputs = base_pooler(hidden_states)
        self.assertIsNotNone(outputs)

    def test_base_pretrained_model_with_different_hidden_states(self):
        config = FAKE_BERT_CONFIG
        base_pretrained_model = BasePreTrainedModel(config)
        self.assertIsNotNone(base_pretrained_model)

    def test_base_model_with_different_hidden_states(self):
        config = FAKE_BERT_CONFIG
        base_model = SimpleModel(config)
        input_ids = torch.randn(2, 4, config.hidden_size, dtype=torch.float)
        attention_mask = torch.ones(2, 4, dtype=torch.float)
        outputs = base_model(input_ids, attention_mask)
        self.assertIsNotNone(outputs)

    def test_base_for_sequence_classification_with_different_hidden_states(self):
        config = FAKE_BERT_CONFIG
        base_for_sequence_classification = BaseForSequenceClassification(config)
        self.assertIsNotNone(base_for_sequence_classification)

    def test_create_position_ids_from_input_ids_with_different_input_ids(self):
        input_ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        padding_idx = 0
        past_key_values_length = 0
        position_ids = create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length)
        expected_position_ids = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
        self.assertTrue(torch.equal(position_ids, expected_position_ids))

    def test_base_embeddings_with_different_config_and_input_ids(self):
        config = FAKE_BERT_CONFIG
        base_embeddings = BaseEmbeddings(config)
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        token_type_ids = torch.tensor([[0, 0, 1], [0, 1, 1]])
        position_ids = torch.tensor([[1, 2, 3], [1, 2, 3]])
        embeddings = base_embeddings(input_ids, token_type_ids, position_ids)
        self.assertIsNotNone(embeddings)

    def test_base_self_output_with_different_config_and_hidden_states(self):
        config = FAKE_BERT_CONFIG
        base_self_output = BaseSelfOutput(config)
        hidden_states = torch.randn(1, 3, config.hidden_size, dtype=torch.float)
        input_tensor = torch.randn(1, 3, config.hidden_size, dtype=torch.float)
        outputs = base_self_output(hidden_states, input_tensor)
        self.assertIsNotNone(outputs)

    def test_base_attention_with_different_config_and_hidden_states(self):
        config = FAKE_BERT_CONFIG
        base_attention = BaseAttention(config)
        hidden_states = torch.randn(1, 3, config.hidden_size, dtype=torch.float)
        attention_mask = torch.ones(1, 3, dtype=torch.float)
        outputs = base_attention(hidden_states, attention_mask)
        self.assertIsNotNone(outputs)

    def test_base_intermediate_with_different_config_and_hidden_states(self):
        config = FAKE_BERT_CONFIG
        base_intermediate = BaseIntermediate(config)
        hidden_states = torch.randn(1, 3, config.hidden_size, dtype=torch.float)
        outputs = base_intermediate(hidden_states)
        self.assertIsNotNone(outputs)

    def test_base_output_with_different_config_and_hidden_states(self):
        config = FAKE_BERT_CONFIG
        base_output = BaseOutput(config)
        hidden_states = torch.randn(1, 3, 512, dtype=torch.float)  # hidden_size = 512
        input_tensor = torch.randn(1, 3, 256, dtype=torch.float)  # hidden_size = 256
        outputs = base_output(hidden_states, input_tensor)
        self.assertIsNotNone(outputs)

    def test_base_layer_with_different_config_and_hidden_states(self):
        config = FAKE_BERT_CONFIG
        base_layer = BaseLayer(config)
        hidden_states = torch.randn(1, 3, config.hidden_size, dtype=torch.float)
        attention_mask = torch.ones(1, 3, dtype=torch.float)
        outputs = base_layer(hidden_states, attention_mask)
        self.assertIsNotNone(outputs)

    def test_base_encoder_with_different_config_and_hidden_states(self):
        config = FAKE_BERT_CONFIG
        base_encoder = BaseEncoder(config)
        hidden_states = torch.randn(1, 3, config.hidden_size, dtype=torch.float)
        attention_mask = torch.ones(1, 3, dtype=torch.float)
        outputs = base_encoder(hidden_states, attention_mask)
        self.assertIsNotNone(outputs)

    def test_base_pooler_with_different_config_and_hidden_states(self):
        config = FAKE_BERT_CONFIG
        base_pooler = BasePooler(config)
        hidden_states = torch.randn(1, 3, config.hidden_size, dtype=torch.float)
        outputs = base_pooler(hidden_states)
        self.assertIsNotNone(outputs)

    def test_base_pretrained_model_with_different_config_and_hidden_states(self):
        config = FAKE_BERT_CONFIG
        base_pretrained_model = BasePreTrainedModel(config)
        self.assertIsNotNone(base_pretrained_model)

    def test_base_model_with_different_config_and_hidden_states(self):
        config = FAKE_BERT_CONFIG
        base_model = SimpleModel(config)
        input_ids = torch.randn(1, 3, config.hidden_size, dtype=torch.float)
        attention_mask = torch.ones(1, 3, dtype=torch.float)
        outputs = base_model(input_ids, attention_mask)
        self.assertIsNotNone(outputs)

    def test_base_for_sequence_classification_with_different_config_and_hidden_states(self):
        config = FAKE_BERT_CONFIG
        base_for_sequence_classification = BaseForSequenceClassification(config)
        self.assertIsNotNone(base_for_sequence_classification)

    def test_create_position_ids_from_input_ids_with_different_padding_and_past_length(self):
        input_ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]])
        padding_idx = 0
        past_key_values_length = 2
        position_ids = create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length)
        expected_position_ids = torch.tensor([[3, 4, 5, 0, 0], [3, 4, 5, 6, 7]])
        self.assertTrue(torch.equal(position_ids, expected_position_ids))

    def test_base_embeddings_with_different_config_and_input_ids_and_device(self):
        config = FAKE_BERT_CONFIG
        base_embeddings = BaseEmbeddings(config)
        base_embeddings.device = torch.device("cpu")
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        token_type_ids = torch.tensor([[0, 0, 1], [0, 1, 1]])
        position_ids = torch.tensor([[1, 2, 3], [1, 2, 3]])
        embeddings = base_embeddings(input_ids, token_type_ids, position_ids)
        self.assertIsNotNone(embeddings)

    def test_base_self_output_with_different_config_and_hidden_states_and_device(self):
        config = FAKE_BERT_CONFIG
        base_self_output = BaseSelfOutput(config)
        base_self_output.device = torch.device("cpu")
        hidden_states = torch.randn(1, 3, config.hidden_size, dtype=torch.float)
        input_tensor = torch.randn(1, 3, config.hidden_size, dtype=torch.float)
        outputs = base_self_output(hidden_states, input_tensor)
        self.assertIsNotNone(outputs)

    def test_base_attention_with_different_config_and_hidden_states_and_device(self):
        config = FAKE_BERT_CONFIG
        base_attention = BaseAttention(config)
        base_attention.device = torch.device("cpu")
        hidden_states = torch.randn(1, 3, config.hidden_size, dtype=torch.float)
        attention_mask = torch.ones(1, 3, dtype=torch.float)
        outputs = base_attention(hidden_states, attention_mask)
        self.assertIsNotNone(outputs)

    def test_base_intermediate_with_different_config_and_hidden_states_and_device(self):
        config = FAKE_BERT_CONFIG
        base_intermediate = BaseIntermediate(config)
        base_intermediate.device = torch.device("cpu")
        hidden_states = torch.randn(1, 3, config.hidden_size, dtype=torch.float)
        outputs = base_intermediate(hidden_states)
        self.assertIsNotNone(outputs)

    def test_base_output_with_different_config_and_hidden_states_and_device(self):
        config = FAKE_BERT_CONFIG
        base_output = BaseOutput(config)
        base_output.device = torch.device("cpu")
        hidden_states = torch.randn(1, 3, 512, dtype=torch.float)  # hidden_size = 512
        input_tensor = torch.randn(1, 3, 256, dtype=torch.float)  # hidden_size = 256
        outputs = base_output(hidden_states, input_tensor)
        self.assertIsNotNone(outputs)

    def test_base_layer_with_different_config_and_hidden_states_and_device(self):
        config = FAKE_BERT_CONFIG
        base_layer = BaseLayer(config)
        base_layer.device = torch.device("cpu")
        hidden_states = torch.randn(1, 3, config.hidden_size, dtype=torch.float)
        attention_mask = torch.ones(1, 3, dtype=torch.float)
        outputs = base_layer(hidden_states, attention_mask)
        self.assertIsNotNone(outputs)

    def test_base_encoder_with_different_config_and_hidden_states_and_device(self):
        config = FAKE_BERT_CONFIG
        base_encoder = BaseEncoder(config)
        base_encoder.device = torch.device("cpu")
        hidden_states = torch.randn(1, 3, config.hidden_size, dtype=torch.float)
        attention_mask = torch.ones(1, 3, dtype=torch.float)
        outputs = base_encoder(hidden_states, attention_mask)
        self.assertIsNotNone(outputs)

    def test_base_pooler_with_different_config_and_hidden_states_and_device(self):
        config = FAKE_BERT_CONFIG
        base_pooler = BasePooler(config)
        base_pooler.device = torch.device("cpu")
        hidden_states = torch.randn(1, 3, config.hidden_size, dtype=torch.float)
        outputs = base_pooler(hidden_states)
        self.assertIsNotNone(outputs)

    def test_base_pretrained_model_with_different_config_and_hidden_states_and_device(self):
        config = FAKE_BERT_CONFIG
        base_pretrained_model = BasePreTrainedModel(config)
        base_pretrained_model.to(torch.device("cpu"))
        self.assertIsNotNone(base_pretrained_model)

    def test_base_model_with_different_config_and_hidden_states_and_device(self):
        config = FAKE_BERT_CONFIG
        base_model = SimpleModel(config)
        base_model.device = torch.device("cpu")
        input_ids = torch.randn(1, 3, config.hidden_size, dtype=torch.float)
        attention_mask = torch.ones(1, 3, dtype=torch.float)
        outputs = base_model(input_ids, attention_mask)
        self.assertIsNotNone(outputs)

    def test_base_for_sequence_classification_with_different_config_and_hidden_states_and_device(self):
        config = FAKE_BERT_CONFIG
        base_for_sequence_classification = BaseForSequenceClassification(config)
        base_for_sequence_classification.to(torch.device("cpu"))
        self.assertIsNotNone(base_for_sequence_classification)

    def test_create_position_ids_from_input_ids_with_different_padding_idx_and_past_length(self):
        input_ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]])
        padding_idx = 0
        past_key_values_length = 2
        position_ids = create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length)
        expected_position_ids = torch.tensor([[3, 4, 5, 0, 0], [3, 4, 5, 6, 7]])
        self.assertTrue(torch.equal(position_ids, expected_position_ids))

    def test_create_position_ids_from_input_ids_with_different_padding_idx_and_no_past_length(self):
        input_ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]])
        padding_idx = 0
        past_key_values_length = 0
        position_ids = create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length)
        expected_position_ids = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 3, 4, 5]])
        self.assertTrue(torch.equal(position_ids, expected_position_ids))

    def test_base_embeddings_with_different_input_ids_and_token_type_ids(self):
        config = FAKE_BERT_CONFIG
        base_embeddings = BaseEmbeddings(config)
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        token_type_ids = torch.tensor([[0, 0, 1], [0, 1, 1]])
        position_ids = torch.tensor([[1, 2, 3], [1, 2, 3]])
        embeddings = base_embeddings(input_ids, token_type_ids, position_ids)
        self.assertIsNotNone(embeddings)

    def test_base_embeddings_with_different_input_ids_and_no_token_type_ids(self):
        config = FAKE_BERT_CONFIG
        base_embeddings = BaseEmbeddings(config)
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        embeddings = base_embeddings(input_ids)
        self.assertIsNotNone(embeddings)

    def test_base_attention_with_different_hidden_states_and_attention_mask(self):
        config = FAKE_BERT_CONFIG
        base_attention = BaseAttention(config)
        hidden_states = torch.randn(1, 3, config.hidden_size, dtype=torch.float)
        attention_mask = torch.ones(1, 3, dtype=torch.float)
        outputs = base_attention(hidden_states, attention_mask)
        self.assertIsNotNone(outputs)

    def test_base_attention_with_different_hidden_states_and_no_attention_mask(self):
        config = FAKE_BERT_CONFIG
        base_attention = BaseAttention(config)
        hidden_states = torch.randn(1, 3, config.hidden_size, dtype=torch.float)
        outputs = base_attention(hidden_states)
        self.assertIsNotNone(outputs)


class TestBaseEmbeddings(unittest.TestCase):
    def setUp(self):
        self.bert_config = FAKE_BERT_CONFIG
        self.xlm_roberta_config = FAKE_XLM_CONFIG
        self.input_ids = torch.LongTensor([[1, 2, 3, 4, 5]])
        self.token_type_ids = torch.LongTensor([[0, 1, 1, 1, 0]])
        self.position_ids = torch.LongTensor([[0, 1, 2, 3, 4]])
        self.past_key_values_length = 2

    def test_init(self):
        # 测试初始化方法
        embeddings = BaseEmbeddings(self.bert_config)
        assert embeddings.config == self.bert_config
        assert isinstance(embeddings.word_embeddings, nn.Embedding)
        assert isinstance(embeddings.position_embeddings, nn.Embedding)
        assert isinstance(embeddings.token_type_embeddings, nn.Embedding)
        assert isinstance(embeddings.LayerNorm, nn.LayerNorm)
        assert isinstance(embeddings.dropout, nn.Dropout)
        assert embeddings.position_embedding_type == "absolute"
        assert embeddings.position_ids.size() == torch.Size([1, self.bert_config.max_position_embeddings])
        assert embeddings.token_type_ids.size() == torch.Size([1, self.bert_config.max_position_embeddings])

    def test_forward_bert_config(self):
        # 测试使用BertConfig的情况
        embeddings = BaseEmbeddings(self.bert_config)
        embeddings_output = embeddings(self.input_ids)
        assert embeddings_output.size() == torch.Size([1, 5, self.bert_config.hidden_size])

    def test_forward_xlm_roberta_config(self):
        # 测试使用XLMRobertaConfig的情况
        embeddings = BaseEmbeddings(self.xlm_roberta_config)
        embeddings_output = embeddings(self.input_ids)
        assert embeddings_output.size() == torch.Size([1, 5, self.xlm_roberta_config.hidden_size])

    def test_forward_with_token_type_ids(self):
        # 测试使用token_type_ids的情况
        embeddings = BaseEmbeddings(self.bert_config)
        embeddings_output = embeddings(self.input_ids, token_type_ids=self.token_type_ids)
        assert embeddings_output.size() == torch.Size([1, 5, self.bert_config.hidden_size])

    def test_forward_with_position_ids(self):
        # 测试使用position_ids的情况
        embeddings = BaseEmbeddings(self.bert_config)
        embeddings_output = embeddings(self.input_ids, position_ids=self.position_ids)
        assert embeddings_output.size() == torch.Size([1, 5, self.bert_config.hidden_size])


class TestBaseSelfAttention(unittest.TestCase):
    def setUp(self):
        self.config_bert = FAKE_BERT_CONFIG
        self.config_xlmr = FAKE_XLM_CONFIG
        self.model_bert = BaseSelfAttention(self.config_bert)
        self.model_xlmr = BaseSelfAttention(self.config_xlmr)
        self.model_bert_relative_key = BaseSelfAttention(self.config_bert, position_embedding_type="relative_key")
        self.model_bert_relative_kq = BaseSelfAttention(self.config_bert, position_embedding_type="relative_key_query")

    def test_init(self):
        # Test if the model is initialized correctly
        self.assertEqual(self.model_bert.num_attention_heads, self.config_bert.num_attention_heads)
        self.assertEqual(self.model_bert.attention_head_size,
                         self.config_bert.hidden_size //
                         self.config_bert.num_attention_heads)
        self.assertEqual(self.model_bert.all_head_size,
                         self.config_bert.num_attention_heads *
                         self.config_bert.hidden_size // self.config_bert.num_attention_heads)
        self.assertEqual(self.model_bert.position_embedding_type, "absolute")

        self.assertEqual(self.model_xlmr.num_attention_heads, self.config_xlmr.num_attention_heads)
        self.assertEqual(self.model_xlmr.attention_head_size,
                         self.config_xlmr.hidden_size // self.config_xlmr.num_attention_heads)
        self.assertEqual(self.model_xlmr.all_head_size,
                         self.config_xlmr.num_attention_heads *
                         self.config_xlmr.hidden_size //
                         self.config_xlmr.num_attention_heads)
        self.assertEqual(self.model_xlmr.position_embedding_type, "absolute")

    def test_forward(self):
        # Test if the forward method works correctly
        hidden_states = torch.randn(1, 1, self.config_bert.hidden_size)
        attention_mask = torch.ones(1, 1)
        output_bert = self.model_bert(hidden_states, attention_mask=attention_mask)
        output_xlmr = self.model_xlmr(hidden_states, attention_mask=attention_mask)
        self.model_bert_relative_key(hidden_states, attention_mask=attention_mask)
        self.model_bert_relative_kq(hidden_states, attention_mask=attention_mask)
        self.assertEqual(output_bert[0].shape, output_xlmr[0].shape)


class TestBaseSelfOutput(unittest.TestCase):
    def setUp(self):
        self.config = FAKE_BERT_CONFIG
        self.model = BaseSelfOutput(self.config)


class TestBaseAttention:
    def __init__(self):
        self.config = FAKE_BERT_CONFIG
        self.attention = BaseAttention(self.config)

    def setup_method(self):
        pass

    def test_prune_heads(self):
        self.attention.prune_heads([])

        self.attention.prune_heads([0, 1])
        assert self.attention.self.num_attention_heads == self.config.num_attention_heads - 2
        assert (self.attention.self.all_head_size == self.config.hidden_size - 2 * self.config.hidden_size //
                self.config.num_attention_heads)
        assert 0 in self.attention.pruned_heads and 1 in self.attention.pruned_heads

    def test_forward(self):
        hidden_states = torch.randn(1, 1, self.config.hidden_size)
        attention_mask = torch.ones(1, 1)
        outputs = self.attention(hidden_states, attention_mask)
        assert outputs[0].shape == (1, 1, self.config.hidden_size)

    def test_forward_with_pruned_heads(self):
        self.attention.prune_heads([0, 1])
        hidden_states = torch.randn(1, 1, self.config.hidden_size)
        attention_mask = torch.ones(1, 1)
        outputs = self.attention(hidden_states, attention_mask)
        assert outputs[0].shape == (1, 1, self.config.hidden_size)


class TestBaseOutput(unittest.TestCase):
    def setUp(self):
        self.config = FAKE_BERT_CONFIG
        self.model = BaseOutput(self.config)

    def test_forward(self):
        # 创建一些随机的输入张量
        hidden_states = torch.randn(1, 1, self.config.intermediate_size)
        input_tensor = torch.randn(1, 1, self.config.hidden_size)

        # 运行模型的前向传播
        output = self.model(hidden_states, input_tensor)

        # 检查输出的形状是否正确
        self.assertEqual(output.shape, (1, 1, self.config.hidden_size))

        # 检查是否使用了正确的参数
        self.assertEqual(self.model.dense.in_features, self.config.intermediate_size)
        self.assertEqual(self.model.dense.out_features, self.config.hidden_size)
        self.assertEqual(self.model.LayerNorm.normalized_shape, (self.config.hidden_size,))
        self.assertEqual(self.model.LayerNorm.eps, self.config.layer_norm_eps)
        self.assertEqual(self.model.dropout.p, self.config.hidden_dropout_prob)

    def test_different_config(self):
        # 使用不同的配置创建模型
        config = FAKE_XLM_CONFIG
        model = BaseOutput(config)
        hidden_states = torch.randn(1, 1, config.intermediate_size)
        input_tensor = torch.randn(1, 1, config.hidden_size)
        output = model(hidden_states, input_tensor)
        self.assertEqual(output.shape, (1, 1, config.hidden_size))
        self.assertEqual(model.dense.in_features, config.intermediate_size)
        self.assertEqual(model.dense.out_features, config.hidden_size)
        self.assertEqual(model.LayerNorm.normalized_shape, (config.hidden_size,))
        self.assertEqual(model.LayerNorm.eps, config.layer_norm_eps)
        self.assertEqual(model.dropout.p, config.hidden_dropout_prob)


class TestBaseEncoder(unittest.TestCase):
    def setUp(self):
        self.config = FAKE_BERT_CONFIG
        self.encoder = BaseEncoder(self.config)

    def test_forward_with_no_optional_params(self):
        hidden_states = torch.randn(1, 1, self.config.hidden_size)
        output = self.encoder(hidden_states)
        self.assertEqual(output.last_hidden_state.shape, (1, 1, self.config.hidden_size))

    def test_with_xlm_roberta_config(self):
        self.config = FAKE_XLM_CONFIG
        self.encoder = BaseEncoder(self.config)
        hidden_states = torch.randn(1, 1, self.config.hidden_size)
        output = self.encoder(hidden_states)
        self.assertEqual(output.last_hidden_state.shape, (1, 1, self.config.hidden_size))


class TestBasePooler(unittest.TestCase):
    def setUp(self):
        self.config_bert = FAKE_BERT_CONFIG
        self.config_xlmr = FAKE_XLM_CONFIG
        self.pooler_bert = BasePooler(self.config_bert)
        self.pooler_xlmr = BasePooler(self.config_xlmr)

    def test_forward_multi_dim(self):
        hidden_states = torch.randn(5, 10, self.config_bert.hidden_size)
        output = self.pooler_bert(hidden_states)
        self.assertEqual(output.shape, (5, self.config_bert.hidden_size))


class TestBaseModel(unittest.TestCase):
    def setUp(self):
        from tests.pythontest.atb_llm.models.base.mock_class import MockTorchClasses
        self.mock_torch_classes = MockTorchClasses()
        torch.classes = self.mock_torch_classes
        self.config_bert = FAKE_BERT_CONFIG
        self.config_xlmr = FAKE_XLM_CONFIG
        self.model_bert = BertModel(self.config_bert)
        self.model_xlmr = XLMRobertaModel(self.config_xlmr)

    def test_init(self):
        model = BaseModel(self.config_bert)
        self.assertEqual(model.config, self.config_bert)

    def test_init_ascend_weight(self):
        pass # model.state_dict 为空

    def test_prepare_inputs_for_ascend(self):
        model = BaseModel(self.config_bert)
        input_shape = (2, 10)
        input_ids = torch.randint(0, 100, input_shape)
        position_ids = torch.randint(0, 10, input_shape)
        token_type_ids = torch.ones(input_shape)
        attention_mask = torch.ones(input_shape)
        model.kv_attention_manager = KVAttentionManager(self.config_bert, input_shape)
        inputs = model.prepare_inputs_for_ascend(input_ids, position_ids, token_type_ids, attention_mask)
        self.assertIsNotNone(inputs)

    def test_execute_ascend_operator(self):
        input_shape = (2, 10)
        model = BaseModel(self.config_bert)
        model.kv_attention_manager = KVAttentionManager(self.config_bert, input_shape)

        input_ids = torch.randint(0, 100, input_shape)
        position_ids = torch.randint(0, 10, input_shape)
        token_type_ids = torch.ones(input_shape)
        attention_mask = torch.ones(input_shape)
        inputs = model.execute_ascend_operator(input_ids, position_ids, token_type_ids, attention_mask)
        self.assertIsNotNone(inputs)

    def test_forward(self):
        input_ids = torch.randint(0, 100, (2, 10))
        attention_mask = torch.ones((2, 10))
        output_bert = self.model_bert(input_ids, attention_mask=attention_mask)
        output_xlmr = self.model_xlmr(input_ids, attention_mask=attention_mask)
        self.assertEqual(output_bert.last_hidden_state.shape, output_xlmr.last_hidden_state.shape)


class TestBaseForSequenceClassification(unittest.TestCase):
    def setUp(self):
        self.config = FAKE_BERT_CONFIG
        self.model = BaseForSequenceClassification(self.config)

    def test_calc_loss(self):
        batch_size, num_classes = 1, 2
        logits = torch.randn(batch_size, num_classes)  # 随机生成一些值作为 logits
        labels = torch.randint(0, num_classes, (batch_size,))  
        loss = self.model._calc_loss(logits, labels)
        self.assertIsNotNone(loss)


class TestKVAttentionManager(unittest.TestCase):
    def setUp(self):
        self.config = FAKE_BERT_CONFIG
        self.input_shape = (2, 10)
        self.attention_manager = KVAttentionManager(self.config, self.input_shape)

    def test_init(self):
        self.assertEqual(self.attention_manager.batch_size, self.input_shape[0])
        self.assertEqual(self.attention_manager.seq_len, self.input_shape[1])
        self.assertEqual(self.attention_manager.hidden_size, self.config.hidden_size)
        self.assertEqual(self.attention_manager.num_layers, self.config.num_hidden_layers)

    def test_init_seq_len_and_token_offset(self):
        seq_len = 5
        self.attention_manager.init_seq_len_and_token_offset(seq_len)
        self.assertEqual(self.attention_manager.token_offset, seq_len)
        self.assertEqual(self.attention_manager.seq_len_list_full, [seq_len] * self.input_shape[0])
        self.assertEqual(self.attention_manager.seq_len_tensor_full.shape, (self.input_shape[0],))
        self.assertEqual(self.attention_manager.seq_len_tensor_full.dtype, torch.int32)
        self.assertEqual(self.attention_manager.seq_len_list_inc, [1] * self.input_shape[0])
        self.assertEqual(self.attention_manager.seq_len_tensor_inc.shape, (self.input_shape[0],))
        self.assertEqual(self.attention_manager.seq_len_tensor_inc.dtype, torch.int32)
        self.assertEqual(self.attention_manager.token_offset_tensor.shape, (self.input_shape[0],))
        self.assertEqual(self.attention_manager.token_offset_tensor.dtype, torch.int32)


if __name__ == '__main__':
    unittest.main()