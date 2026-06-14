# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from ..base.input_builder import InputBuilder


class Internlm2InputBuilder(InputBuilder):
    def __init__(self, tokenizer, model_version, generation_config, **kwargs):
        self.tokenizer = tokenizer
        self.model_version = model_version
        self.generation_config = generation_config
        self.meta_instruction = ("You are an AI assistant whose name is InternLM (书生·浦语).\n"
        "- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory "
        "(上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n"
        "- InternLM (书生·浦语) can understand and communicate fluently in the language chosen "
        "by the user such as English and 中文.")
        super().__init__(tokenizer, **kwargs)

    def _apply_chat_template(self, conversation, **kwargs):
        if self.tokenizer.add_bos_token:
            prompt = ""
        else:
            prompt = self.tokenizer.bos_token
        if self.meta_instruction:
            prompt += f"""<|im_start|>system\n{self.meta_instruction}<|im_end|>\n"""
        for record in conversation:
            if record['role'] == 'user':
                prompt += f"""<|im_start|>user\n{record['content']}<|im_end|>\n"""
            else:
                prompt += f"""<|im_start|>assistant\n{record['content']}<|im_end|>\n"""
        prompt += """<|im_start|>assistant\n"""
        prompt = self.tokenizer(prompt)["input_ids"]
        return prompt
