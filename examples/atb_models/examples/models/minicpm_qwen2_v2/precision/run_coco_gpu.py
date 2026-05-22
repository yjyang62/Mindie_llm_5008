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

import torch
from PIL import Image
from transformers import AutoModel

from atb_llm.models.base.model_utils import safe_from_pretrained
from atb_llm.utils import argument_utils
from atb_llm.utils.argument_utils import BooleanArgumentValidator, ArgumentAction
from atb_llm.models.base.model_utils import safe_get_tokenizer_from_pretrained
from atb_llm.utils.multimodal_utils import safe_open_image
from examples.models.coco_base_runner import CocoBaseRunner

torch.manual_seed(1234)


def parse_args():
    bool_validator = BooleanArgumentValidator()
    parser = argument_utils.ArgumentParser(description="Demo")
    parser.add_argument('--trust_remote_code', action=ArgumentAction.STORE_TRUE.value, 
                        validator=bool_validator)
    return parser.parse_args()


class CocoMinicpmLlamaRunner(CocoBaseRunner):
    def __init__(self, model_path, image_path, **kwargs):
        super().__init__(model_path, image_path)
        self.device = torch.device('cuda', 0)
        self.model = None
        self.trust_remote_code = kwargs.get('trust_remote_code', False)
        
    def prepare(self):
        model_path = self.args.model_path
        model = safe_from_pretrained(AutoModel, model_path, torch_dtype=torch.float16, 
                                     trust_remote_code=self.trust_remote_code)
        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = safe_get_tokenizer_from_pretrained(model_path, 
                                                            trust_remote_code=self.trust_remote_code)
        
    def process(self, img_path, img_name):
        image = safe_open_image(Image, img_path).convert('RGB')
        question = "Write an essay about this image, at least 256 words."
        msgs = [{'role': 'user', 'content': question}]
        res = self.model.chat(
            image=image,
            msgs=msgs,
            tokenizer=self.tokenizer,
            sampling=True,
            temperature=0.7
        )
        image.close()
        self.image_answer[img_name] = res


def main():
    args = parse_args()
    minicpmllama3v2_model_path = '/data/MiniCPM-Llama3-V-2_5/'
    minicpmllama3v2_image_path = '/data/pictures/'
    runner = CocoMinicpmLlamaRunner(minicpmllama3v2_model_path, minicpmllama3v2_image_path,
                                    trust_remote_code=args.trust_remote_code)
    runner.run()

if __name__ == "__main__":
    main()
