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

from atb_llm.models.base.flash_causal_lm_v3 import EngineWrapper
from atb_llm.models.llama.flash_causal_llama_v3 import FlashLlamaForCausalLMV3
from atb_llm.models.base.mindie_llm_config import MindIELLMConfig
from atb_llm.utils.mapping import Mapping
from tests.pythontest.atb_llm.models.base.test_mindie_llm_config import MockBaseConfig
from tests.pythontest.atb_llm.models.base.mock_class import MockTorchClasses


class TestFlashQwen2ForCausalLMV3(unittest.TestCase):
    @patch("atb_llm.models.base.flash_causal_lm_v3.load_atb_speed", MagicMock())
    @patch("atb_llm.models.llama.flash_causal_llama_v3.LlamaModel", MagicMock())
    @patch("atb_llm.models.llama.flash_causal_llama_v3.ColumnParallelLinear", MagicMock())
    def setUp(self):
        torch.classes = MockTorchClasses()
        hf_config = MockBaseConfig()
        hf_config.torch_dtype = torch.float16
        self.mindie_llm_config = MindIELLMConfig(hf_config, None, Mapping(rank=0, world_size=2))
        self.weight_loader = MagicMock()
        self.weight_loader.device = torch.device("npu")
        self.model = FlashLlamaForCausalLMV3(self.mindie_llm_config, self.weight_loader)
    
    @patch("atb_llm.models.base.flash_causal_lm_v3.get_default_net", MagicMock())
    @patch("atb_llm.models.llama.flash_causal_llama_v3.gather", MagicMock())
    @patch("atb_llm.models.llama.flash_causal_llama_v3.dist", MagicMock())
    def test_forward(self):
        engine_wrapper = EngineWrapper(
            feature_list=["prefill"],
            input_keys={"input_ids"},
            args={"is_prefill": True}
        )
        self.model._build(engine_wrapper)
        engine_wrapper = EngineWrapper(
            feature_list=["decode"],
            input_keys={"input_ids"},
            args={"is_prefill": False}
        )
        self.model._build(engine_wrapper)


if __name__ == "__main__":
    unittest.main()