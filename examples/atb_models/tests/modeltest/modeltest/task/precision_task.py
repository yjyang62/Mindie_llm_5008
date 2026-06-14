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
from modeltest.api.task import Task, LogitsDumpConfig


class PrecisionTask(Task):
    def __init__(self, task_config) -> None:
        super().__init__(task_config)
        self.set_environ()
    
    @staticmethod
    def get_batched_data(data, batch_size):
        """
        将数据集分割成多个批次。

        :param data: 数据集，一个包含字典的列表
        :param batch_size: 每个批次的大小
        :return: 生成器，每次迭代返回一个批次的数据
        """
        batch = []
        for idx, item in enumerate(data):
            batch.append(item)
            if len(batch) == batch_size:
                yield idx, batch
                batch = []
        if batch:  # 如果最后一个批次不足batch_size，也返回
            yield idx, batch
    
    def build_bad_case_datasets(self, datasets):
        """
        根据bad case的编号筛选出对应的数据。

        :datasets: 数据集所有数据，包含list的list
        :param bad_case_list: 错例的编号
        :return: 返回一个子数据集
        """
        bad_case_list = LogitsDumpConfig.bad_case_list
        if len(bad_case_list) == 0 or not isinstance(bad_case_list[0], int):
            raise IndexError("BAD_CASE_LIST must has some int values")
        current_idx = 0
        bad_case_datasets = []
        for dataset in datasets:
            bad_case_dataset = []
            for data in dataset:
                if current_idx in bad_case_list:
                    bad_case_dataset.append(data)
                current_idx += 1
            bad_case_datasets.append(bad_case_dataset)
        return bad_case_datasets

    def set_environ(self):
        os.environ["LCCL_DETERMINISTIC"] = "1"
        os.environ["HCCL_DETERMINISTIC"] = "true"
        os.environ["ATB_MATMUL_SHUFFLE_K_ENABLE"] = "0"
        os.environ["MODELTEST_DATASET_SPECIFIED"] = self.task_config.task_name

    def inference(self):
        pass