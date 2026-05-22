# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import copy
import unittest
from unittest.mock import patch, MagicMock
from typing import Any, Dict, Optional

from ddt import ddt, unpack

from atb_llm.models.base.config import RopeScaling
from atb_llm.models.hunyuan.config_hunyuan import HunyuanConfig
from atb_llm.models.hunyuan.router_hunyuan import HunyuanRouter

NUM_HIDDEN_LAYERS = 28

FAKE_MODEL_NAME_OR_PATH = "fake_model_name_or_path"
FAKE_CONFIG_DICT = {
    'model_type': 'hunyuan',
    'num_hidden_layers': NUM_HIDDEN_LAYERS,
    'num_key_value_heads': 8,
    'n_routed_experts': 4,
    'num_experts_per_tok': 2,
    'n_shared_experts': 0,
    'cla_share_factor': 1,
    'max_position_embeddings': 4096,
    'vocab_size': 102400,
}

rope_scaling = {
    'type': 'dynamic',
    'factor': 2.0,
    'max_position_embeddings': 8192,
}


class MockTokenizer:
    def __init__(self, use_fast, trust_remote_code):
        self.use_fast = use_fast
        self.trust_remote_code = trust_remote_code


def mock_safe_get_tokenizer_from_pretrained(_, **kwargs):
    use_fast = kwargs.get("use_fast")
    trust_remote_code = kwargs.get("trust_remote_code")
    return MockTokenizer(use_fast, trust_remote_code)
    
    
@ddt
class TestHunyuanRouter(unittest.TestCase):
    @patch("atb_llm.models.base.router.BaseRouter.check_config", return_value=None)
    @patch("atb_llm.models.hunyuan.router_hunyuan.HunyuanRouter")
    @patch("atb_llm.models.hunyuan.router_hunyuan.HunyuanRouter.check_config_hunyuan")
    @patch("atb_llm.models.hunyuan.config_hunyuan.HunyuanConfig.from_pretrained", return_value=HunyuanConfig(rope_scaling))
    def test_get_config(self, _1, _2, _3, _4):
        router = HunyuanRouter("", FAKE_CONFIG_DICT)
        config = router.get_config()
        self.assertEqual(config.model_type, "hunyuan")

    @unpack
    @patch("atb_llm.models.hunyuan.router_hunyuan.safe_get_tokenizer_from_pretrained")
    def test_get_tokenizer(self, mock_func):
        mock_func.side_effect = mock_safe_get_tokenizer_from_pretrained
        config_dict = copy.deepcopy(FAKE_CONFIG_DICT)
        router = HunyuanRouter(FAKE_MODEL_NAME_OR_PATH, config_dict)
        tokenizer = router.get_tokenizer()
        self.assertIsInstance(router, HunyuanRouter)
        self.assertEqual(router.config_dict, FAKE_CONFIG_DICT)
        self.assertEqual(router.config_dict['model_type'], 'hunyuan')
        self.assertEqual(router.config_dict['num_hidden_layers'], NUM_HIDDEN_LAYERS)
        self.assertFalse(tokenizer.use_fast)
        self.assertFalse(tokenizer.trust_remote_code)

    @unpack
    @patch("atb_llm.models.hunyuan.config_hunyuan.HunyuanConfig.from_pretrained", return_value=HunyuanConfig(rope_scaling))
    @patch("atb_llm.models.hunyuan.router_hunyuan.HunyuanRouter.check_config_hunyuan")
    @patch("atb_llm.models.hunyuan.router_hunyuan.HunyuanRouter")
    @patch("atb_llm.models.base.router.BaseRouter.check_config", return_value=None)
    def test_check_config_hunyuan(self, config, _2, _3, mock_from_pretrained):
        mock_config = HunyuanConfig(
            model_type='hunyuan',
            num_hidden_layers=NUM_HIDDEN_LAYERS,
            num_key_value_heads=8,
            n_routed_experts=4,
            num_experts_per_tok=2,
            n_shared_experts=0,
            cla_share_factor=1,
            max_position_embeddings=4096,
            vocab_size=102400,
            rope_scaling={
                'type': 'dynamic',
                'factor': 2.0,
                'max_position_embeddings': 8192,
            }
        )

        mock_from_pretrained.return_value = mock_config

        router = HunyuanRouter(model_name_or_path=FAKE_MODEL_NAME_OR_PATH, config_dict=FAKE_CONFIG_DICT)
        config = router.get_config()
        rope_scaling_type = config.rope_scaling_dict.get("type")
        
        self.assertIsInstance(config, HunyuanConfig)
        self.assertIsInstance(config.rope_scaling_dict, dict)
        self.assertIsNotNone(rope_scaling_type)

if __name__ == '__main__':
    unittest.main()