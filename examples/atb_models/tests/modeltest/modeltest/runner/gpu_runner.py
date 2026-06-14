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
import importlib
import torch
from modeltest.model.gpu_model import GPUModel
from modeltest.api.runner import Runner
from modeltest.api.task import LogitsDumpConfig


class GPURunner(Runner):
    def __init__(self, *args) -> None:
        super().__init__('GPU', *args)
        self.backend = self.model_config.requested_gpu_framework
        self.model_runner = self.get_model_runner_cls()(
            self.runner_config.tp,
            self.model_config,
            self.task.task_config)
        self.model = self.model_runner.model
        self.tokenizer = self.model_runner.tokenizer
        self.post_init()
    
    def get_rank(self):
        return torch.cuda.current_device()
    
    def get_model_runner_cls(self):
        module = importlib.import_module(f"modeltest.model.{self.backend.lower()}_model")
        return getattr(module, f"{self.backend}Model")
    
    def run_inference(self, queries):
        if self.model_runner is not None and not isinstance(self.model_runner, GPUModel):
            raise TypeError("Expected a model of GPUModel.")
        return self.model_runner.inference(
            queries, self.task.task_config.requested_max_output_length)
    
    def save_queries_and_token_ids_impl(self, queries, result_tuple):
        for _, query in enumerate(queries):
            self.metric.csv_debug.get("key", []).append(len(self.metric.csv_debug.get("key", [])))
            self.metric.csv_debug.get("queries", []).append(query)
        self.metric.csv_debug.get("input_token_ids", []).extend(result_tuple.input_id)
        self.metric.csv_debug.get("output_token_ids", []).extend(result_tuple.generate_id)
    
    def save_logits_impl(self, task_name, bad_case_idx, scores):
        logits_dump_token_max_length = LogitsDumpConfig.logits_dump_token_max_length
        bad_case_list = LogitsDumpConfig.bad_case_list
        if self.task.task_config.task_name == 'boolq':
            scores = scores[:, -1, :] if scores.dim() == 3 else scores
            logits_batch_path = os.path.join(self.metric.logits_dump_dir, 
                                             f'logits_{task_name}_{bad_case_list[bad_case_idx]}_0.pth')
            torch.save(scores.cpu(), logits_batch_path)
        else:
            if self.task.task_config.task_name == 'humanevalx':
                flatten_bad_cases = []
                for bad_cases in bad_case_list:
                    for item in bad_cases:
                        flatten_bad_cases.append(item)
                bad_case_list = flatten_bad_cases
            for token_id, logits in enumerate(scores):
                if token_id == logits_dump_token_max_length:
                    break
                if '/' in task_name:
                    task_name = task_name.replace("/", "_")
                logits_batch_path = os.path.join(
                    self.metric.logits_dump_dir, f'logits_{task_name}_{bad_case_list[bad_case_idx]}_{token_id}.pth'
                )
                torch.save(logits.cpu(), logits_batch_path)

    def get_logits_impl(self, result_tuple):
        return result_tuple.logits