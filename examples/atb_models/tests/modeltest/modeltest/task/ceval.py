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

from .ceval_mmlu import CEvalMMLUFewShotsPrecisionTask, CEvalMMLUZeroShotPrecisionTask


class CEvalFewShotsPrecisionTask(CEvalMMLUFewShotsPrecisionTask):
    def __init__(self, task_config) -> None:
        super().__init__(task_config)

    def get_test_set(self):
        return 'val'
    
    def get_row_col(self):
        return 1


class CEvalZeroShotPrecisionTask(CEvalMMLUZeroShotPrecisionTask):
    def __init__(self, task_config) -> None:
        super().__init__(task_config)
    
    def format_example(self, name, batch_data, idx):
        question = batch_data[idx][0]
        option_a = batch_data[idx][1]
        option_b = batch_data[idx][2]
        option_c = batch_data[idx][3]
        option_d = batch_data[idx][4]
        prompt = self.task_config.prompt.format(name["name_ch"], 
                                                question, option_a, option_b, option_c, option_d)
        return prompt
    
    def get_test_set(self):
        return 'val'
    
    def get_row_col(self):
        return 1