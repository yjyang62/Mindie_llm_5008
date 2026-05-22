# Copyright (c) 2023; Baichuan Intelligent Technology. All rights reserved.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Implement part of this file based on baichuan-inc/Baichuan-7B
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from atb_llm.utils.log import logger
from ..base.input_builder import InputBuilder


class BaichuanInputBuilder(InputBuilder):
    def __init__(self, tokenizer, model_version, generation_config, **kwargs):
        self.model_version = model_version
        self.generation_config = generation_config
        super().__init__(tokenizer, **kwargs)

    def _apply_chat_template(self, conversation, **kwargs):
        total_input, round_input = [], []
        for message in conversation[::-1]:
            content_tokens = self.tokenizer.encode(message['content'])
            if message['role'] == 'user':
                round_input = [self.generation_config.user_token_id] + content_tokens + round_input
                total_input = round_input + total_input
                round_input = []
            elif message['role'] == 'assistant':
                round_input = [
                                  self.generation_config.assistant_token_id
                              ] + content_tokens + [
                                  self.generation_config.eos_token_id
                              ] + round_input
            else:
                error_msg = f"message role not supported yet: {message['role']}"
                logger.error(error_msg)
                raise ValueError(error_msg)
        total_input.append(self.generation_config.assistant_token_id)
        return total_input
