#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import os
import torch
from PIL import Image

from atb_llm.utils import multimodal_utils
from atb_llm.utils.log import logger
from atb_llm.models.base.model_utils import safe_from_pretrained
from examples.models.coco_base_runner import CocoBaseRunner


class CocoLLaVARunner(CocoBaseRunner):
    def __init__(self, model_path, image_path):
        super().__init__(model_path, image_path)

        llava_type = "llava"
        self.args.llava_type = llava_type
        logger.info(f"===== llava_type: {llava_type}")

        # set save_output parameters
        self.device = 'cuda:0'
        self.override_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        self.encoding = "utf-8"
        self.indent = None
        self.ensure_ascii = True
        self.processor = None
        self.model = None

    def prepare(self):
        model_path = self.args.model_path
        llava_type = self.args.llava_type
        device = self.device

        processor = None
        model = None
        if llava_type == "llava":
            from transformers import LlavaProcessor, LlavaForConditionalGeneration
            processor = safe_from_pretrained(LlavaProcessor, model_path, trust_remote_code=False)
            model = safe_from_pretrained(LlavaForConditionalGeneration, model_path,
                                         torch_dtype=torch.float16, device_map=device)
        else:
            from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
            processor = safe_from_pretrained(LlavaNextProcessor, model_path, trust_remote_code=False)
            model = safe_from_pretrained(LlavaNextForConditionalGeneration, model_path,
                                         torch_dtype=torch.float16, device_map=device)
        self.processor = processor
        self.model = model


    def process(self, img_path, img_name):

        processor = self.processor
        model = self.model
        image = multimodal_utils.safe_load_multimodal_source(Image.open, img_path)
        prompt = "USER: <image>\nDescribe this image in detail. ASSISTANT:"
        device = self.device

        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
        image.close()
        inputs = inputs.to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=30)
        response = processor.decode(outputs.cpu()[0], skip_special_tokens=True)

        self.image_answer[img_name] = response.split("ASSISTANT:")[-1]


if __name__ == "__main__":

    llava_model_path = "/data/datasets/llava-1.5-13b-hf" 
    llava_image_path = "/data/datasets/coco_data/val_images" 

    runner = CocoLLaVARunner(llava_model_path, llava_image_path)
    runner.run()