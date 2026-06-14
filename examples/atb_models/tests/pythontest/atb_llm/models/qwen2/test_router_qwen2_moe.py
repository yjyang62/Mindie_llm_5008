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
import copy
from ddt import ddt
from atb_llm.models.qwen2_moe.router_qwen2_moe import Qwen2moeRouter
from atb_llm.models.qwen2_moe.configuration_qwen2_moe import Qwen2MoeConfig
from atb_llm.utils.parameter_validators import DictionaryParameterValidator

NUM_HIDDEN_LAYERS = 61

FAKE_MODEL_NAME_OR_PATH = "fake_model_name_or_path"
FAKE_CONFIG_DICT = {    
    "model_type": "qwen2_moe",
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "decoder_sparse_step": 1,
    "eos_token_id": 151645,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "initializer_range": 0.02,
    "intermediate_size": 8192,
    "max_position_embeddings": 40960,
    "max_window_layers": 94,
    "mlp_only_layers": [],
    "moe_intermediate_size": 1536,
    "num_attention_heads": 64,
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "num_hidden_layers": 94,
    "num_key_value_heads": 4,
    "rms_norm_eps": 1e-06,
    "router_aux_loss_coef": 0.001,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.51.0",
    "vocab_size": 151936
}


class MockTokenizer:
    def __init__(self, use_fast, trust_remote_code):
        self.use_fast = use_fast
        self.trust_remote_code = trust_remote_code
        self.eos_token_id = 1


def mock_safe_get_tokenizer_from_pretrained(_, **kwargs):
    use_fast = kwargs.get("use_fast")
    trust_remote_code = kwargs.get("trust_remote_code")
    return MockTokenizer(use_fast, trust_remote_code)


@ddt
class TestQwen2moeRouter(unittest.TestCase):
    def setUp(self):
        self.router = Qwen2moeRouter(
            model_name_or_path=FAKE_MODEL_NAME_OR_PATH,
            config_dict=FAKE_CONFIG_DICT
        )
        self.config = Qwen2MoeConfig(
            **FAKE_CONFIG_DICT
        )

    def test_get_llm_config_validators(self):
        llm_config_validators = self.router.get_llm_config_validators()

        self.assertIn("llm", llm_config_validators)
        llm_validator = llm_config_validators["llm"]
        self.assertIsInstance(llm_validator, DictionaryParameterValidator)
        self.assertIn("models", llm_config_validators)
        qwen_moe_config_validator = llm_config_validators["models"]["qwen_moe"]
        self.assertIsInstance(qwen_moe_config_validator, DictionaryParameterValidator)

    def test_check_config(self):
        self.router.check_config(self.config)

        error_config = copy.deepcopy(self.config)
        with self.assertRaises(ValueError):
            error_config.mlp_only_layers = [-1]
            self.router.check_config(error_config)

        error_config = copy.deepcopy(self.config)
        with self.assertRaises(ValueError):
            error_config.decoder_sparse_step = 20000
            self.router.check_config(error_config)

if __name__ == '__main__':
    unittest.main()