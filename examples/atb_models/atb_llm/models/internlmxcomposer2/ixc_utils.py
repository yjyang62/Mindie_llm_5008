# coding=utf-8
# Copyright (c) The InternLM team and The HuggingFace Inc. team. All rights reserved.
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
# Implement padding_336 based on padding_336 from internlm/internlm-xcomposer2-4khd-7b
# Implement HD_transform based on HD_transform from internlm/internlm-xcomposer2-4khd-7b
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2. 
# You can use this software according to the terms and conditions of the Mulan PSL v2. 
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2 
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, 
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, 
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE. fu
# See the Mulan PSL v2 for more details.
import torch
import numpy as np
import torchvision
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as transforms

def padding_336(b):
    width, height = b.size
    tar = int(np.ceil(height / 336) * 336)
    top_padding = int((tar - height)/2)
    bottom_padding = tar - height - top_padding
    left_padding = 0
    right_padding = 0
    b = transforms.functional.pad(b, [left_padding, top_padding, right_padding, bottom_padding], fill=[255,255,255])

    return b

def HD_transform(img, hd_num=16):
    width, height = img.size
    trans = False
    if width < height:
        img = img.transpose(Image.TRANSPOSE)
        trans = True
        width, height = img.size
    ratio = (width/ height)
    scale = 1
    while scale*np.ceil(scale/ratio) <= hd_num:
        scale += 1
    scale -= 1
    new_w = int(scale * 336)
    new_h = int(new_w / ratio)

    img = transforms.functional.resize(img, [new_h, new_w],)
    img = padding_336(img)
    width, height = img.size
    if trans:
        img = img.transpose(Image.TRANSPOSE)

    return img