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
import ast
from dataclasses import dataclass, asdict
from typing import Union, List, Dict
import torch
import yaml
from tabulate import tabulate
from atb_llm.utils.log.logging import logger
from atb_llm.utils.file_utils import safe_open


@dataclass
class TaskConfig:
    task_type: str
    task_name: str
    hf_dataset_path: str
    om_dataset_path: str
    local_dataset_path: str
    prompt: str
    choices: List
    shots: int
    requested_max_input_length: int
    requested_max_output_length: int
    need_logits: bool
    need_truncate_input: bool
    metric: Dict[str, Union[str, float]]
    metric_type: str
    metadata_version: str
    humaneval_x_datasets_selector: List[str]
    subject_mapping: Dict

    def __repr__(self):
        config_table = [[k, v] for k, v in asdict(self).items()]
        return tabulate(config_table, headers=["Field", "Value"], tablefmt="grid", maxcolwidths=[None, 100])


class LogitsDumpConfig:
    bad_case_logits_dump: bool
    logits_dump_token_max_length: int
    bad_case_list: List[int]

    bad_case_logits_dump = ast.literal_eval(os.getenv("BAD_CASE_LOGITS_DUMP", "False"))
    try:
        logits_dump_token_max_length = int(os.getenv("LOGITS_DUMP_TOKEN_MAX_LENGTH", "0"))
    except Exception as e:
        raise ValueError("The token num for logits dump must be integer and greater or equal to 0") from e
    if bad_case_logits_dump and logits_dump_token_max_length <= 0:
        raise ValueError("Please check env: LOGITS_DUMP_TOKEN_MAX_LENGTH, must greater than 0")
    bad_case_list = ast.literal_eval(os.getenv('BAD_CASE_LIST', '[]'))
    if not isinstance(bad_case_list, list) or len(bad_case_list) >= 100000:
        raise ValueError("Please check env: BAD_CASE_LIST, must be a list with less than 100000 elements")


class Task():
    def __init__(self, task_config) -> None:
        self.task_config: TaskConfig = task_config
        self.tokenizer = None
        self.local_dataset_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.metric_type = task_config.metric.get('metric_type', 'pass_k')
        self.k_value = task_config.metric.get('k', 1.0)

    @staticmethod
    def parse_config(config_path):
        with safe_open(config_path, 'r', encoding='utf-8') as file:
            config_dict = yaml.safe_load(file)
            task_config = TaskConfig(
                task_type=config_dict.get('task_type', ""),
                task_name=config_dict.get('task_name', ""),
                hf_dataset_path=config_dict.get('hf_dataset_path', ""),
                om_dataset_path=config_dict.get('om_dataset_path', ""),
                local_dataset_path=config_dict.get('local_dataset_path', ""),
                prompt=config_dict.get('prompt', ""),
                choices=config_dict.get('choices', []),
                shots=config_dict.get('shots', 0),
                requested_max_input_length=config_dict.get('requested_max_input_length', 256),
                requested_max_output_length=config_dict.get('requested_max_output_length', 256),
                need_logits=config_dict.get('need_logits', False),
                need_truncate_input=config_dict.get('need_truncate_input', False),
                metric=config_dict.get('metric', {}),
                metric_type=config_dict.get('metric', {}).get('metric_type', ""),
                metadata_version=config_dict.get('metadata', {}).get('version', "1.0"),
                humaneval_x_datasets_selector=config_dict.get('humaneval_x_datasets_selector', []),
                subject_mapping=config_dict.get('subject_mapping', {})
            )
        
        if not os.path.exists(task_config.local_dataset_path):
            task_config.local_dataset_path = os.path.join("..", task_config.local_dataset_path)
        logger.info(f"Task config:\n{task_config}")
        return task_config
    
    @staticmethod
    def get_npu_runner_extra_args():
        return {}

    def run(self):
        with torch.no_grad():
            self.inference()

    def inference(self):
        pass

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def prepare_data(self, metric):
        raise NotImplementedError("Subclasses should implement prepare_data.")

    def build_queries(self, sub_dataset_idx, batched_data, model_config):
        raise NotImplementedError("Subclasses should implement build_queries.")

    def result_judge(self, metric, generate_token_lists, logits, sub_dataset_idx, batched_data):
        raise NotImplementedError("Subclasses should implement result_judge.")
