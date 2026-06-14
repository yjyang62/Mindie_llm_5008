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

import argparse
import json
import logging
import os

import torch
import torch.nn.functional as F

from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig, AntiOutlier
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
from atb_llm.utils import file_utils
from examples.convert.convert_utils import copy_tokenizer_files
from atb_llm.models.base.model_utils import safe_get_tokenizer_from_pretrained
from atb_llm.models.base.model_utils import safe_get_model_from_pretrained
from atb_llm.models.base.model_utils import safe_get_config_from_pretrained


def parse_args():
    parser = argparse.ArgumentParser(description="Creating quant weights ")
    parser.add_argument("--model_path", type=str, help="The path to model float weights")
    parser.add_argument("--save_path", type=str, help="The path to save quant weights")
    parser.add_argument("--anti_prompt", type=str, default="examples/models/llama3/anti_prompt_pdmix.json", 
                        help="The prompts for anti outlier")
    parser.add_argument("--calib_prompt", type=str, default="examples/models/llama3/calib_prompt_pdmix.json",
                        help="The prompts for anti outlier")
    parser.add_argument("--auto_layer", action='store_true', help="If true, auto select rollback layer")
    parser.add_argument("--trust_remote_code", action='store_true', help="Whether to trust local executable files")
    return parser.parse_args()


def get_anti_dataset(tokenizer, calib_list, device="npu"):
    calib_dataset = []
    max_len = 0
    for calib_data in calib_list:
        inputs = tokenizer(calib_data, return_tensors='pt')
        calib_dataset.append(
            inputs.data['input_ids'].to(device))
        max_len = max(max_len, inputs.data['input_ids'].size(1))
    for i, calib_data in enumerate(calib_dataset):
        calib_dataset[i] = F.pad(calib_data, (0, max_len - calib_data.size(1)), value=0)
    return torch.cat(calib_dataset)


def get_calib_dataset(tokenizer, calib_list, device="npu"):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer(calib_data, return_tensors='pt', add_special_tokens=False)
        calib_dataset.append([
            inputs.data['input_ids'].to(device),
            inputs.data['attention_mask'].to(device)
        ])
    return calib_dataset


def modify_config(model_dir, dest_dir, torch_dtype, quantize_type):
    model_dir = file_utils.standardize_path(model_dir, check_link=False)
    file_utils.check_path_permission(model_dir)
    src_config_filepath = os.path.join(model_dir, 'config.json')
    with file_utils.safe_open(src_config_filepath, 'r', encoding='utf-8') as fr:
        data = json.load(fr)
    data['torch_dtype'] = str(torch_dtype).split(".")[1]
    data['quantize'] = quantize_type
    quantization_config = {
        'kv_quant_type': "C8",
        'pdmix': True
    }
    data['quantization_config'] = quantization_config
    dest_dir = file_utils.standardize_path(dest_dir, check_link=False)
    file_utils.check_path_permission(dest_dir)
    dest_config_filepath = os.path.join(dest_dir, 'config.json')
    with file_utils.safe_open(dest_config_filepath, 'w', encoding='utf-8', is_exist_ok=False) as fw:
        json.dump(data, fw, indent=4)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    args = parse_args()
    config = safe_get_config_from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    tokenizer = safe_get_tokenizer_from_pretrained(args.model_path,
                                                   trust_remote_code=args.trust_remote_code)
    model = safe_get_model_from_pretrained(args.model_path,
                                           trust_remote_code=args.trust_remote_code,
                                           config=config,
                                           torch_dtype='auto',
                                           device_map='auto')
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token

    with file_utils.safe_open(args.anti_prompt, 'r', encoding='utf-8') as f:
        anti_prompt = json.load(f)
    anti_data = []
    for prompt in anti_prompt:
        tmp = get_anti_dataset(tokenizer, prompt)
        anti_data.append(tmp)
    anti_dataset = []
    for data in anti_data:
        anti_dataset.append([data])
    with file_utils.safe_open(args.calib_prompt, 'r', encoding='utf-8') as f:
        calib_prompt = json.load(f)
    calib_dataset = []
    for prompt in calib_prompt:
        tmp = get_calib_dataset(tokenizer, prompt)
        calib_dataset += (tmp)

    keys = ['.o_proj']
    anti_disable_names = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            for key in keys:
                if key in name:
                    anti_disable_names.append(name)
    
    anti_config = AntiOutlierConfig(anti_method='m6', dev_type='npu', dev_id=model.device.index,
                                    disable_anti_names=anti_disable_names, flex_config={})
    anti_outlier = AntiOutlier(model, calib_data=anti_dataset, cfg=anti_config)
    anti_outlier.process()

    if args.auto_layer:
        disable_names = [f"model.layers.{layer}.mlp.down_proj" for layer in range(0, config.num_hidden_layers)]
        from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.layer_select import LayerSelector
        ls = LayerSelector(model, layer_names=disable_names, range_method='quantile')
        ls.run(anti_dataset)
        disable_names = ls.select_layers_by_disable_level(40)
    quant_config = QuantConfig(
        w_bit=8,
        a_bit=8,
        disable_names=disable_names,
        dev_type='npu',
        dev_id=model.device.index,
        act_method=1,
        pr=1.0,
        w_sym=True,
        mm_tensor=False,
        is_dynamic=False,
        use_kvcache_quant=True,
    )

    calibrator = Calibrator(model, quant_config, calib_data=calib_dataset, disable_level='L0')
    calibrator.run()
    calibrator.save(args.save_path, save_type=['ascendV1'])
    modify_config(args.model_path, args.save_path, config.torch_dtype, quantize_type="w8a8")
    copy_tokenizer_files(args.model_path, args.save_path)