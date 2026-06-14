# coding=utf-8
# Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
# Copyright 2023 Haotian Liu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Implement expand2square based on expand2square from haotian-liu/LLaVA
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
# MindIE is licensed under Mulan PSL v2.buque
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from PIL import Image
from transformers import CLIPImageProcessor
from atb_llm.utils.multimodal_utils import safe_open_image
from ..base.model_utils import safe_from_pretrained


def expand2square(input_image, background_color):
    width, height = input_image.size
    if height > width:
        result = Image.new(input_image.mode, (height, height), background_color)
        result.paste(input_image, ((height - width) // 2, 0))
        return result
    elif height < width:
        result = Image.new(input_image.mode, (width, width), background_color)
        result.paste(input_image, (0, (width - height) // 2))
        return result
    else:
        return input_image


class DataProcessorYiVl:
    def __init__(self, vision_path, trust_remote_code, **kwargs):
        self.image_processor = safe_from_pretrained(CLIPImageProcessor, vision_path, 
                                        trust_remote_code=trust_remote_code)

    def preprocess_image(self, config, image_path):
        image = safe_open_image(Image, image_path).convert('RGB')
        if getattr(config, "image_aspect_ratio", None) == "pad":
            backgrorund = tuple(int(x * 255) for x in self.image_processor.image_mean)
            image = expand2square(image, backgrorund)
        pixel_values = self.image_processor(images=image, return_tensors="pt")["pixel_values"]
        image.close()
        return pixel_values


