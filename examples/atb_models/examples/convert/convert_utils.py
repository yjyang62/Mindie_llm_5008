# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import json
import os.path
import shutil
from atb_llm.utils import file_utils

MAX_TOKENIZER_FILE_SIZE = 1024 * 1024 * 1024


def copy_tokenizer_files(model_dir, dest_dir):
    model_dir = file_utils.standardize_path(model_dir, check_link=False)
    file_utils.check_path_permission(model_dir)
    if os.path.exists(dest_dir):
        dest_dir = file_utils.standardize_path(dest_dir, check_link=False)
        file_utils.check_path_permission(dest_dir)
    else:
        os.makedirs(dest_dir, exist_ok=True)
        dest_dir = file_utils.standardize_path(dest_dir, check_link=False)
    for filename in file_utils.safe_listdir(model_dir):
        need_move = False
        file_names = ['tokenizer', 'tokenization', 'special_token_map', 'generation', 'configuration']
        for f in file_names:
            if f in filename:
                need_move = True
                break
        if need_move:
            src_filepath = os.path.join(model_dir, filename)
            src_filepath = file_utils.standardize_path(src_filepath, check_link=False)
            file_utils.check_file_safety(src_filepath, 'r', max_file_size=MAX_TOKENIZER_FILE_SIZE)
            dest_filepath = os.path.join(dest_dir, filename)
            dest_filepath = file_utils.standardize_path(dest_filepath, check_link=False)
            file_utils.check_file_safety(dest_filepath, 'w', max_file_size=MAX_TOKENIZER_FILE_SIZE)
            shutil.copyfile(src_filepath, dest_filepath)


def modify_config(model_dir, dest_dir, torch_dtype, quantize_type, is_exist_ok=False):
    model_dir = file_utils.standardize_path(model_dir, check_link=False)
    file_utils.check_path_permission(model_dir)
    src_config_filepath = os.path.join(model_dir, 'config.json')
    with file_utils.safe_open(src_config_filepath, 'r', encoding='utf-8') as fr:
        data = json.load(fr)
    data['torch_dtype'] = str(torch_dtype).split(".")[1]
    dest_dir = file_utils.standardize_path(dest_dir, check_link=False)
    file_utils.check_path_permission(dest_dir)
    dest_config_filepath = os.path.join(dest_dir, 'config.json')
    with file_utils.safe_open(dest_config_filepath, 'w', encoding='utf-8', is_exist_ok=is_exist_ok) as fw:
        json.dump(data, fw, indent=4)