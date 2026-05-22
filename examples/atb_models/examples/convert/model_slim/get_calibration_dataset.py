# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import json

from atb_llm.utils import file_utils


def load_jsonl(dataset_path, key_name='inputs_pretokenized'):
    dataset = []
    if dataset_path == "./atb_llm/models/qwen2/humaneval_x.jsonl":
        key_name = 'prompt'
    with file_utils.safe_open(dataset_path, 'r', encoding='utf-8') as file:
        for line in file_utils.safe_readlines(file):
            data = json.loads(line)
            text = data.get(key_name, line)
            dataset.append(text)
    return dataset