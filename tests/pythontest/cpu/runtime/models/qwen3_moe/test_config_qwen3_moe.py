# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from mindie_llm.runtime.models.qwen3_moe.config_qwen3_moe import Qwen3MoeConfig


def test_qwen3_config_minimal():
    config = Qwen3MoeConfig.from_dict({
        "num_attention_heads": 64,
        "num_key_value_heads": 4,
        "vocab_size": 151936,
        "max_position_embeddings": 40960,
        "rms_norm_eps": 1e-6,
    })
    assert config.is_reasoning_model is True