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

from tabulate import tabulate
from atb_llm.utils.file_utils import safe_open
from modeltest.api.metric import Metric
from atb_llm.utils.log.logging import logger


class AccMetric(Metric):
    def __init__(self, *args) -> None:
        super().__init__("acc", *args)
        self.correct_num = 0
        self.correct_num_list = []
    
    def print_metric(self):
        data = [
            [
                self.task_config.task_name,
                self.task_config.shots,
                self.case_num,
                self.correct_num,
                "acc",
                self.correct_num / self.case_num
            ]
        ]
        for idx, sub_dataset in enumerate(self.task_config.subject_mapping.keys()):
            if self.case_num_list[idx] == 0:
                continue
            data.append([self.task_config.subject_mapping[sub_dataset]["name"],
                         self.task_config.shots,
                         self.case_num_list[idx],
                         self.correct_num_list[idx],
                         "acc", 
                         self.correct_num_list[idx] / self.case_num_list[idx]])
        headers = ["Tasks", "N-Shot", "Total", "Pass", "Metric", "Value"]
        markdown_table = tabulate(data, headers, tablefmt="grid")
        logger.info(f"Metric Table:\n{markdown_table}")
        with safe_open(self.result_file_path, mode='w', encoding='utf-8') as file:
            file.write(f"runner config:\n{self.runner_config}\n")
            file.write(f"model config:\n{self.model_config}\n")
            file.write(f"task config:\n{self.task_config}\n")
            file.write(f"metric table:\n{markdown_table}\n")
        super().print_metric()