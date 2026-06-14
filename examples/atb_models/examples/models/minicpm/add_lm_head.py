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

from safetensors.torch import save_file
from safetensors import safe_open

ORIGIN_MODEL_PATH = ''
TARGET_MODEL_PATH = ''

tensors = {}
with safe_open(ORIGIN_MODEL_PATH, framework="pt", device='cpu') as f:
    for k in f.keys():
        tensors[k] = f.get_tensor(k)
    tensors['lm_head'] = f.get_tensor('model.embed_tokens.weight').clone()
    save_file(tensors, TARGET_MODEL_PATH, metadata={'format': 'pt'})