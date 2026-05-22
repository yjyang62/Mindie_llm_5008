#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import argparse
from optimum.onnxruntime import ORTModelForFeatureExtraction
from atb_llm.models.base.model_utils import safe_from_pretrained

parser = argparse.ArgumentParser(description="Export a model from transformers to ONNX format.")
parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint to convert.")

args = parser.parse_args()

model_checkpoint = args.model_path

ort_model = safe_from_pretrained(ORTModelForFeatureExtraction, model_checkpoint, export=True, from_transformers=True)

# Save the ONNX model
ort_model.save_pretrained(model_checkpoint)