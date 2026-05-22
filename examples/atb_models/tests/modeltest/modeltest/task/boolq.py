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

import json
import os
from tqdm import tqdm
from modeltest.metric.acc import AccMetric
from modeltest.api.task import LogitsDumpConfig
from atb_llm.utils.file_utils import safe_open
import torch.nn.functional as F
from .precision_task import PrecisionTask


class BoolQPrecisionTask(PrecisionTask):
    def __init__(self, task_config) -> None:
        super().__init__(task_config)
    
    def prepare_data(self, metric):
        boolq_datasets = []
        for sub_dataset_name in tqdm(self.task_config.subject_mapping.keys()):
            entry = os.path.join(
                self.local_dataset_folder,
                self.task_config.local_dataset_path,
                sub_dataset_name)
            if isinstance(metric, AccMetric):
                metric.correct_num_list.append(0)
            dataset = []
            with safe_open(entry, encoding='utf-8') as f:
                for line in f:
                    line_json = json.loads(line)
                    dataset.append(line_json)
            boolq_datasets.append(dataset)

        if LogitsDumpConfig.bad_case_logits_dump:
            boolq_datasets = super().build_bad_case_datasets(boolq_datasets)
        return boolq_datasets

    def build_queries(self, _, batched_data, model_config):
        titles, questions, passages = zip(*[(item['title'], item['question'], item['passage']) for
                                            item in batched_data])
        formatted_queries = [
            self.task_config.prompt.format(title=title, text=question, passage=passage)
            for title, question, passage in zip(titles, questions, passages)
        ]
        return formatted_queries
    
    def result_judge(self, metric, _, logits, sub_dataset_idx, batched_data):
        logits = logits[:, -1, :] if logits.dim() == 3 else logits
        logits_softmax = F.log_softmax(logits.float(), dim=-1)[:, self.__get_choice_tokens()]
        for idx, item in enumerate(batched_data):
            choice = (logits_softmax[idx, 0] > logits_softmax[idx, 1]).cpu()
            acc = choice == item['answer']
            metric.csv_debug.get("golden_result", []).append(item['answer'])
            metric.csv_debug.get("test_result", []).append(choice.item())
            metric.csv_debug.get("pass", []).append(acc.item())
            if acc:
                metric.correct_num += 1
                metric.correct_num_list[sub_dataset_idx] += 1
    
    def __get_choice_tokens(self):
        sample_yes = "How can we learning machine learning: yes"
        sample_no = "How can we learning machine learning: no"
        # extract yes & no
        choice_tokens = [
            self.tokenizer(
                [sample_yes], 
                return_tensors="pt", 
                max_length=2048, 
                add_special_tokens=False
            ).input_ids[0, -1].item(), 
            self.tokenizer(
                [sample_no], 
                return_tensors="pt", 
                max_length=2048, 
                add_special_tokens=False
            ).input_ids[0, -1].item()
        ]
        return choice_tokens