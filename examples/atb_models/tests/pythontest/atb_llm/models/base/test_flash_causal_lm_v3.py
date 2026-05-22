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
from unittest.mock import patch, MagicMock
from ddt import data, ddt

import torch
from atb_llm.models.base.flash_causal_lm_v3 import FlashCausalLMV3, torch_to_mindie_graph, EngineWrapper
from atb_llm.utils.layers.embedding.position_rotary_embedding import PositionEmbeddingType
from atb_llm.nn.parameter import Parameter
from tests.pythontest.atb_llm.models.base.mock_class import MockTorchClasses
from tests.pythontest.atb_llm.models.base.feature_decorator.v3.test_base_decorator import AliceFeatureDecorator


class MockModelParam(MagicMock):
    def __init__(self, *args, **kw) -> None:
        super().__init__(*args, **kw)
        self.hf_config = MagicMock()
        self.hf_config.torch_dtype = torch.float16
        self.hf_config.rope_theta = 10000.0
        self.hf_config.rope_scaling = MagicMock()
        self.hf_config.rope_scaling.factor = 1.0
        self.hf_config.num_hidden_layers = 12
        self.hf_config.vocab_size = 12345
        self.soc_info = MagicMock()
        self.mapping = MagicMock()
        self.mapping.attn_tp = MagicMock()
        self.mapping.attn_tp.rank = 0


class MockModelStatus(MagicMock):
    def __init__(self, *args, **kw,) -> None:
        super().__init__(*args, **kw)
        self.position_embedding_type = PositionEmbeddingType.ROPE
        self.head_dim = 64
    
    @classmethod
    def from_config(cls, mindie_llm_config):
        return cls()


@torch_to_mindie_graph(AliceFeatureDecorator)
class MockFlashCasualV3(FlashCausalLMV3):
    model_status_cls = MockModelStatus

    def __init__(self, mindie_llm_config, weight_loader, **kwargs) -> None:
        super().__init__(mindie_llm_config, weight_loader, **kwargs)
        self.model = MagicMock()
        self.lm_head = MagicMock()
    
    def forward(self, **kwargs):
        return self.lm_head(self.model(**kwargs))


FAKE_MODEL_INPUT = {
    "input_ids": torch.tensor([1,2,3]).npu(),
    "position_ids": torch.tensor([0,1,2]).npu(),
    "kv_cache": [(torch.tensor([0]), torch.tensor([1])), (torch.tensor([2]), torch.tensor([3]))],
    "block_tables": torch.tensor([1, 2]).npu(),
    "slots": torch.tensor([0, 1, 2]).npu(),
    "input_lengths": torch.tensor([2, 1]).npu(),
    "max_seq_len": 10,
    "is_prefill": True
}


@ddt
class TestFlashCausalLMV3(unittest.TestCase):
    @patch("atb_llm.models.base.flash_causal_lm_v3.load_atb_speed", MagicMock())
    def setUp(self):
        torch.classes = MockTorchClasses()
        self.mindie_llm_config = MockModelParam()
        self.weight_loader = MagicMock()
        self.weight_loader.device = torch.device("npu")

        self.model = MockFlashCasualV3(mindie_llm_config=self.mindie_llm_config, weight_loader=self.weight_loader)
    
    def test_init(self):
        self.assertEqual(self.model._feature_index_map, {})
        self.assertEqual(self.model._engine_wrappers, [])
        self.assertFalse(self.model._ready_for_execute)
    
    def test_get_engine_wrappers(self):
        self.model._engine_wrappers = [MagicMock(), None, MagicMock(), MagicMock()]
        self.assertEqual(len(self.model.get_engine_wrappers()), 3)
    
    def test_get_device_weights(self):
        self.model.model.named_parameters.return_value = [
            ("param1", MagicMock(spec=Parameter)),
            ("param2", MagicMock(spec=Parameter))
        ]
        self.model.lm_head.weight = MagicMock(spec=Parameter)
        model_status = MagicMock()
        model_status.enable_matmul_nz = False
        self.model.model_status = model_status

        self.model._get_device_weights()
        
        for _, param in self.model.model.named_parameters():
            if isinstance(param, Parameter):
                param.weight_format_cast.assert_called_once_with(False)
        
        self.model.lm_head.weight.weight_format_cast.assert_called_once_with(False)

        self.assertEqual(len(self.model.device_weights_dict), 3)

    @data(True, False)
    def test_prepare_default_inputs(self, is_prefill):
        pos_embed_manager = MagicMock()
        pos_embed_manager.cosine_table = None
        pos_embed_manager.sine_table = None
        attn_mask_manager = MagicMock()
        attn_mask_manager.generate_mask = MagicMock(return_value=None)
        self.model.pos_embed_generator = pos_embed_manager
        self.model.attn_mask_generator = attn_mask_manager

        input_ids = torch.tensor([1,2,3]).npu()
        position_ids = torch.tensor([0,1,2]).npu()
        block_tables = torch.tensor([1, 2]).npu()
        slots = torch.tensor([0, 1, 2]).npu()
        input_lengths = torch.tensor([2, 1]).npu()
        max_seq_len = 10

        engine_inputs, engine_outputs, enine_runtime_params = self.model._prepare_default_inputs(
            input_ids, position_ids, is_prefill, block_tables, slots, input_lengths, max_seq_len
        )
        
        self.assertTrue(torch.equal(engine_inputs["input_ids"], input_ids))
        self.assertTrue(torch.equal(engine_inputs["position_ids"], position_ids.to(torch.int64)))
        self.assertEqual(engine_inputs["cosine_table"], None)
        self.assertEqual(engine_inputs["sine_table"], None)
        self.assertTrue(torch.equal(engine_inputs["slots_mapping"], slots.to(torch.int32)))
        self.assertTrue(torch.equal(engine_inputs["seq_len"], input_lengths.to(torch.int32)))
        if is_prefill:
            self.assertEqual(engine_inputs["attention_mask"], None)
            self.assertTrue(torch.equal(engine_inputs["lm_head_indices"],
                                        torch.tensor(range(input_ids.shape[0]), dtype=torch.int64).npu()))
        else:
            self.assertTrue(torch.equal(engine_inputs["block_table"], block_tables.to(torch.int32)))
        self.assertIn("model_out", engine_outputs)
        self.assertIn("seq_len", enine_runtime_params)
    
    def test_get_input_keys(self):
        self.model._update_model_inputs = MagicMock(return_value=({"mock": None}, {}, {}))
        input_keys = self.model._get_input_keys({})
        golden_input_keys = {"input_ids", "position_ids", "slots_mapping", "seq_len", 
                             "cosine_table", "sine_table", "attention_mask", "lm_head_indices", "block_table", "mock"}
        self.assertEqual(input_keys, golden_input_keys)

    def test_build_engines(self):
        self.model._build = MagicMock()
        self.model._build_engines({"input_ids"})
        self.assertEqual(len(self.model._engine_wrappers), 4)
        for engine_wrapper in self.model._engine_wrappers:
            self.assertIsNotNone(engine_wrapper)
        self.assertEqual(self.model._get_engine_index(["decode"]), 0)
        self.assertEqual(self.model._get_engine_index(["prefill"]), 1)
        self.assertEqual(self.model._get_engine_index(["decode", "alice"]), 2)
        self.assertEqual(self.model._get_engine_index(["prefill", "alice"]), 3)
    
    @patch("atb_llm.models.base.flash_causal_lm_v3.get_default_net", MagicMock())
    def test_build(self):
        mindie_llm_config = MockModelParam()
        mindie_llm_config.hf_config.num_hidden_layers = 8
        engine_wrapper = EngineWrapper(
            feature_list=["prefill"],
            input_keys={"input_ids"},
            args={"is_prefill": True}
        )
        self.model._build(engine_wrapper)
        engine = engine_wrapper.engine
        engine.set_weights.assert_called_once()
    
    def test_update_feature_inputs(self):
        feature_decorator = MagicMock()
        feature_decorator.feature_name = "mock"
        feature_decorator.need_additional_engine = True
        self.model._feature_dict = {"mock": feature_decorator}
        input_metadata = {"is_prefill": True}

        features = self.model._update_feature_inputs({}, {}, {}, input_metadata)
        self.assertEqual(features, ["prefill", "mock"])

    
    class TestEngineWrapper(unittest.TestCase):
        def setUp(self) -> None:
            self.engine_wrapper = EngineWrapper(
            feature_list=["prefill"],
            input_keys={"input_ids"},
            args={"is_prefill": True}
        )

        def test_execute(self):
            engine = MagicMock()
            self.engine_wrapper.engine = engine

            self.engine_wrapper.execute({}, {}, {})
            engine.forward.assert_called_once()

            with self.assertRaises(RuntimeError):
                engine.forward.side_effect = KeyError("Test error")
                self.engine_wrapper.execute({}, {}, {})


if __name__ == "__main__":
    unittest.main()
