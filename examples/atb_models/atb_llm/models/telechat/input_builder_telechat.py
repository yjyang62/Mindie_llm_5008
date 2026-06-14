# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import copy
from ..base.input_builder import InputBuilder


class TelechatInputBuilder(InputBuilder):
    def __init__(self, tokenizer, model_version, config, generation_config, **kwargs):
        self.model_version = model_version
        self.config = config
        self.generation_config = generation_config
        super().__init__(tokenizer, **kwargs)

    def _apply_chat_template(self, conversation, **kwargs):
        history = copy.deepcopy(conversation)
        question = history.pop()
        q_token = self.tokenizer(question['content'])

        # get the max length we should build our inputs in
        model_max_length = self.config.seq_length
        if self.generation_config.max_new_tokens is not None:
            build_max_length = max(0, model_max_length - self.generation_config.max_new_tokens)
        else:
            build_max_length = max(0, self.generation_config.max_length)
        if build_max_length < 3:
            raise ValueError("Please change max_new_tokens in generation_config.py < seq_length in config.py or "
                             "max_length>3 in generation_config.py.")

        user_id = self.generation_config.user_token_id
        if self.generation_config.system_token_id is not None:
            system_id = self.generation_config.system_token_id
        else:
            system_id = user_id
        bot_id = self.generation_config.bot_token_id
        eos_id = self.generation_config.eos_token_id
        input_tokens = [user_id] + q_token['input_ids'][-build_max_length + 1:] + [bot_id]

        while len(history) > 0:
            message = history.pop()
            if message['role'] == self.system_role_name:
                tokens = [system_id] + self.tokenizer(message['content'])['input_ids']
            elif message['role'] == self.user_role_name:
                tokens = [user_id] + self.tokenizer(message['content'])['input_ids']
            else:
                tokens = [bot_id] + self.tokenizer(message['content'])['input_ids'] + [eos_id]
            input_tokens = tokens + input_tokens
        return input_tokens