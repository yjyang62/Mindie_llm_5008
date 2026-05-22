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
import re
import json
import math
from decimal import Decimal, InvalidOperation
from tqdm import tqdm
from modeltest.metric.acc import AccMetric
from modeltest.api.task import LogitsDumpConfig
from atb_llm.utils.log.logging import logger
from atb_llm.utils.file_utils import safe_open
from .precision_task import PrecisionTask


class GSM8KPrecisionTask(PrecisionTask):
    def __init__(self, task_config) -> None:
        super().__init__(task_config)

    def prepare_data(self, metric):
        gsm8k_datasets = []
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
            gsm8k_datasets.append(dataset)
        
        if LogitsDumpConfig.bad_case_logits_dump:
            gsm8k_datasets = super().build_bad_case_datasets(gsm8k_datasets)
        return gsm8k_datasets
    
    def build_queries(self, _, batched_data, model_config):
        queries = [item['question'] for item in batched_data]
        return queries
    
    def result_judge(self, metric, generate_token_lists, logits, sub_dataset_idx, batched_data):
        answers = generate_token_lists
        answer_results = [answer.lstrip() if answer else "-1" for answer in answers]

        for idx, item in enumerate(batched_data):
            completion = answer_results[idx]
            answer = item['answer']
            acc = self.is_correct(completion, answer)
            metric.csv_debug.get("golden_result").append(answer)
            metric.csv_debug.get("test_result").append(completion)
            metric.csv_debug.get("pass").append(acc)
            
            if acc:
                metric.correct_num += 1
                metric.correct_num_list[sub_dataset_idx] += 1
    
    def is_correct(self, completion, answer):
        gold = self.extract_answer(answer)
        if gold is None:
            return False

        def number_equal(answer, pred):
            if pred is None:
                return False
            try:
                answer_dec = Decimal(answer)
                pred_dec = Decimal(pred)
                return math.isclose(answer_dec, pred_dec, rel_tol=0, abs_tol=Decimal('1e-4'))
            except (InvalidOperation, ValueError, TypeError, SyntaxError) as e:
                logger.error("Error evaluating expression: %s", str(e))
                return False
            except OverflowError as e:
                logger.error("OverflowError: %s", str(e))
                return False

        return number_equal(gold, self.extract_answer(completion))
    
    def extract_answer(self, s):
        pattern = r"([+-])?(?=([0-9]|\.[0-9]))(0|([1-9](\d{0,2}(,\d{3})*)|\d*))?(\.\d*)?(?=\D|$)"
        _pat_last_digit = re.compile(pattern)
        match = list(_pat_last_digit.finditer(s))
        if match:
            last_digit = match[-1].group().replace(",", "").replace("+", "").strip()
        else:
            last_digit = None
        return last_digit