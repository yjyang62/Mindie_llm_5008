# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from dataclasses import dataclass
from ..qwen2_moe.configuration_qwen2_moe import Qwen2MoeConfig


@dataclass
class Qwen3MoeConfig(Qwen2MoeConfig):
    norm_topk_prob: bool = True
    has_shared_expert: bool = False
    use_qk_norm: bool = True
    is_reasoning_model: bool = True
    start_reasoning_token_id = 151667
    end_reasoning_token_id = 151668
    attention_bias: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reasoning_config.start_reasoning_token_id = self.start_reasoning_token_id
        self.reasoning_config.end_reasoning_token_id = self.end_reasoning_token_id