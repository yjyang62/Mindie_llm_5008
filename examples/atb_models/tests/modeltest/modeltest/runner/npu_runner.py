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

from functools import wraps
import os
import datetime
import torch
import pytz
from modeltest.model.npu_model import NPUModel
from modeltest.api.runner import Runner
from modeltest.api.task import LogitsDumpConfig
from atb_llm.utils.file_utils import safe_open


class NPURunner(Runner):
    def __init__(self, *args) -> None:
        super().__init__('NPU', *args)
        self.set_environ()
        self.model_runner = NPUModel(self.runner_config.batch_size, self.model_config, self.task.task_config)
        self.model = self.model_runner.model
        self.tokenizer = self.model.tokenizer
        self.now_str = datetime.datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d_%H-%M-%S.%f")
        self.post_init()

    @staticmethod
    def enable_logits_save(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            os.environ['ATB_LLM_LOGITS_SAVE_ENABLE'] = "1"
            os.environ['ATB_LLM_LOGITS_SAVE_FOLDER'] = self.metric.data_dir
            rtv = func(self, *args, **kwargs)
            os.environ['ATB_LLM_LOGITS_SAVE_ENABLE'] = "0"
            return rtv
        return wrapper

    @staticmethod
    def enable_token_ids_save(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            os.environ['ATB_LLM_TOKEN_IDS_SAVE_ENABLE'] = "1"
            os.environ['ATB_LLM_TOKEN_IDS_SAVE_FOLDER'] = self.metric.data_dir
            rtv = func(self, *args, **kwargs)
            os.environ['ATB_LLM_TOKEN_IDS_SAVE_ENABLE'] = "0"
            return rtv
        return wrapper
    
    def set_environ(self):
        os.environ['ATB_LAYER_INTERNAL_TENSOR_REUSE'] = "1"
        os.environ['ATB_OPERATION_EXECUTE_ASYNC'] = "1"
        os.environ['ATB_CONVERT_NCHW_TO_ND'] = "1"
        os.environ['TASK_QUEUE_ENABLE'] = "1"
        os.environ['ATB_WORKSPACE_MEM_ALLOC_GLOBAL'] = "1"
        os.environ['ATB_CONTEXT_WORKSPACE_SIZE'] = "0"
        os.environ['ATB_LAUNCH_KERNEL_WITH_TILING'] = "1"
        os.environ['PYTORCH_NPU_ALLOC_CONF'] = "expandable_segments:True"
        for env_name, env_value in self.model_config.env.items():
            os.environ[env_name] = env_value
    
    def get_rank(self):
        return self.model.rank
    
    def save_queries_and_token_ids_impl(self, queries, result_tuple):
        if self.model_config.mm_model:
            with safe_open(os.path.join(self.metric.debug_dir, f"outputs_{self.now_str}.txt"), 'a') as f:
                f.write(str(result_tuple.generate_text))
                f.write(str(queries))
                f.write(str(result_tuple.e2e_time) + '\n')
        else:
            for i, query in enumerate(queries):
                self.metric.csv_debug.get("key", []).append(len(self.metric.csv_debug.get("key", [])))
                self.metric.csv_debug.get("queries", []).append(query)
                input_token_ids = torch.load(os.path.join(self.metric.data_dir, f'input_ids_{i}.pth'))
                self.metric.csv_debug.get("input_token_ids", []).append(input_token_ids.tolist())
                with safe_open(os.path.join(self.metric.data_dir, f"output_ids_{i}.txt"), 'r') as f:
                    output_token_ids = list(map(int, f.read().split()))
                self.metric.csv_debug.get("output_token_ids", []).append(output_token_ids)   
    
    def save_logits_impl(self, task_name, bad_case_idx, scores):
        logits_dump_token_max_length = LogitsDumpConfig.logits_dump_token_max_length
        bad_case_list = LogitsDumpConfig.bad_case_list
        if self.task.task_config.task_name == 'humanevalx':
            flatten_bad_cases = []
            for bad_cases in bad_case_list:
                for item in bad_cases:
                    flatten_bad_cases.append(item)
            bad_case_list = flatten_bad_cases
        for token_id in range(logits_dump_token_max_length):
            logits = torch.load(
                os.path.join(self.metric.data_dir, f'logits_{token_id}.pth')
            )
            if '/' in task_name:
                task_name = task_name.replace("/", "_")
            logits_batch_path = os.path.join(
                self.metric.logits_dump_dir, f'logits_{task_name}_{bad_case_list[bad_case_idx]}_{token_id}.pth'
            )
            torch.save(logits, logits_batch_path)

    def get_logits_impl(self, _):
        return torch.load(os.path.join(self.metric.data_dir, 'logits_0.pth'))

    @enable_token_ids_save
    @enable_logits_save
    def run_inference(self, queries):
        if self.model_runner is not None and not isinstance(self.model_runner, NPUModel):
            raise TypeError("Expected a model of NPUModel.")
        extra_args = self.task.get_npu_runner_extra_args()
        return self.model_runner.inference(
            queries,
            self.runner_config.batch_size,
            self.task.task_config.requested_max_output_length,
            False,
            **extra_args)
