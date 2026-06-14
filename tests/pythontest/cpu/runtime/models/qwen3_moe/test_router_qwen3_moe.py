# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from unittest.mock import patch, MagicMock

import pytest

from mindie_llm.runtime.models.qwen3_moe.router_qwen3_moe import Qwen3MoeRouter


class TestQwen3MoeRouter:
    @pytest.fixture
    def qwen3_moe_router(self):
        router_instance = Qwen3MoeRouter(
            config_dict={"model_type": "qwen3_moe"},
            load_config=MagicMock()
        )
        router_instance.load_config.tokenizer_path = "/mock/tokenizer_path"
        router_instance.load_config.trust_remote_code = True
        router_instance.tokenizer = MagicMock()
        router_instance.config = MagicMock()
        router_instance.config.max_position_embeddings = None
        return router_instance

    @patch("mindie_llm.runtime.models.qwen3_moe.router_qwen3_moe.safe_get_tokenizer_from_pretrained")
    def test__get_tokenizer(self, mock_safe_tokenizer, qwen3_moe_router):
        actual_tokenizer = qwen3_moe_router._get_tokenizer()
        mock_safe_tokenizer.assert_called_once_with(
            qwen3_moe_router.load_config.tokenizer_path,
            padding_side="left",
            trust_remote_code=True
        )
        assert actual_tokenizer == mock_safe_tokenizer.return_value

    @patch("mindie_llm.runtime.models.qwen3_moe.router_qwen3_moe.Qwen3MoeInputBuilder")
    def test__get_input_builder_all_scenarios(self, mock_input_builder, qwen3_moe_router):
        # 场景1: 无模板+无max_length
        qwen3_moe_router.custom_chat_template = None
        qwen3_moe_router.config.max_position_embeddings = None
        qwen3_moe_router._get_input_builder()
        mock_input_builder.assert_called_with(qwen3_moe_router.tokenizer)

        # 场景2: 有模板
        mock_input_builder.reset_mock()
        qwen3_moe_router.custom_chat_template = "test_template"
        qwen3_moe_router._get_input_builder()
        mock_input_builder.assert_called_with(qwen3_moe_router.tokenizer, chat_template="test_template")

        # 场景3: 有max_length
        mock_input_builder.reset_mock()
        qwen3_moe_router.custom_chat_template = None
        qwen3_moe_router.config.max_position_embeddings = 8192
        qwen3_moe_router._get_input_builder()
        mock_input_builder.assert_called_with(qwen3_moe_router.tokenizer, max_length=8192)

    def test__get_tool_calls_parser(self, qwen3_moe_router):
        assert qwen3_moe_router._get_tool_calls_parser() == "qwen3"