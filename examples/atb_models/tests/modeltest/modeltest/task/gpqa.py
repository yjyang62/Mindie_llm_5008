# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import re
import os
import pandas as pd
from modeltest.metric.acc import AccMetric
from atb_llm.utils.log.logging import logger
from .precision_task import PrecisionTask


class GPQAPrecisionTask(PrecisionTask):
    def __init__(self, task_config) -> None:
        super().__init__(task_config)
        self.labels = []
        self.model_config = None

    @staticmethod
    def gpqa_postprocess(text: str, options: str, cushion=True) -> str:
        patterns = [
        f'答案是?\s*([{options}])',
        f'答案是?\s*：\s*([{options}])',
        f'答案是?\s*:\s*([{options}])',
        f'答案选项应?该?是\s*([{options}])',
        f'答案选项应?该?为\s*([{options}])',
        f'答案应该?是\s*([{options}])',
        f'答案应该?选\s*([{options}])',
        f'答案选项为?\s*：\s*([{options}])',
        f'答案选项为?\s+\(?\*?\*?([{options}])\*?\*?\)?',
        f'答案选项是?\s*:\s*([{options}])',
        f'答案为\s*([{options}])',
        f'答案选\s*([{options}])',
        f'选择?\s*([{options}])',
        f'故选?\s*([{options}])'
        f'只有选?项?\s?([{options}])\s?是?对',
        f'只有选?项?\s?([{options}])\s?是?错',
        f'只有选?项?\s?([{options}])\s?不?正确',
        f'只有选?项?\s?([{options}])\s?错误',
        f'说法不?对选?项?的?是\s?([{options}])',
        f'说法不?正确选?项?的?是\s?([{options}])',
        f'说法错误选?项?的?是\s?([{options}])',
        f'([{options}])\s?是正确的',
        f'([{options}])\s?是正确答案',
        f'选项\s?([{options}])\s?正确',
        f'所以答\s?([{options}])',
        f'所以\s?([{options}][.。$]?$)',
        f'所有\s?([{options}][.。$]?$)',
        f'[\s，：:,]([{options}])[。，,\.]?$',
        f'[\s，,：:][故即]([{options}])[。\.]?$',
        f'[\s，,：:]因此([{options}])[。\.]?$',
        f'[是为。]\s?([{options}])[。\.]?$',
        f'因此\s?([{options}])[。\.]?$',
        f'显然\s?([{options}])[。\.]?$',
        '答案是\s?(\S+)(?:。|$)',
        '答案应该是\s?(\S+)(?:。|$)',
        '答案为\s?(\S+)(?:。|$)',
        f'(?i)ANSWER\s*:\s*([{options}])',
        f'[Tt]he answer is:?\s+\(?([{options}])\)?',
        f'[Tt]he answer is:?\s+\(?\*?\*?([{options}])\*?\*?\)?',
        f'[Tt]he answer is option:?\s+\(?([{options}])\)?',
        f'[Tt]he correct answer is:?\s+\(?([{options}])\)?',
        f'[Tt]he correct answer is option:?\s+\(?([{options}])\)?',
        f'[Tt]he correct answer is:?.*?boxed{{([{options}])}}',
        f'[Tt]he correct option is:?.*?boxed{{([{options}])}}',
        f'[Tt]he correct answer option is:?.*?boxed{{([{options}])}}',
        f'[Tt]he answer to the question is:?\s+\(?([{options}])\)?',
        f'^选项\s?([{options}])',
        f'^([{options}])\s?选?项',
        f'(\s|^)[{options}][\s。，,：:\.$]',
        '1.\s?(.*?)$',
        f'1.\s?([{options}])[.。$]?$',
        ]
        cushion_patterns = [
            f'([{options}]):',
            f'([{options}])',
        ]

        if cushion:
            patterns.extend(cushion_patterns)
        for pattern in patterns:
            text = text.strip()
            match = re.search(pattern, text, re.DOTALL)
            if match:
                if match.group(1) is not None and match.group(1) != '':
                    outputs = match.group(1)
                else:
                    outputs = match.group(0)
                for i in options:
                    if i in outputs:
                        return i
        return ''

    def prepare_data(self, metric):
        for _ in self.task_config.subject_mapping:
            if isinstance(metric, AccMetric):
                metric.correct_num_list.append(0)
        val_list = self.load_dataset_by_task_name()
        return val_list

    def build_queries(self, sub_dataset_idx, batched_data, model_config):
        self.model_config = model_config
        q_num = len(batched_data)
        prompt = [self.format_example(batched_data, j) for j in range(q_num)]
        self.labels = [j[-1] for j in batched_data]
        prompts = [prpt.encode().decode(encoding="utf8") for prpt in prompt]
        return prompts

    def format_example(self, batch_data, idx):
        question = batch_data[idx][0]
        option_a = batch_data[idx][1]
        option_b = batch_data[idx][2]
        option_c = batch_data[idx][3]
        option_d = batch_data[idx][4]
        prompt = self.task_config.prompt.format(question, option_a, option_b, option_c, option_d).strip()
        return prompt

    def result_judge(self, metric, generate_token_lists, logits, sub_dataset_idx, batched_data):
        for idx, generate_token_list in enumerate(generate_token_lists):
            logger.debug('Question[%d]: %s', len(batched_data) + idx,
                self.build_queries(sub_dataset_idx, batched_data, self.model_config))
            logger.debug('Answer[%d]: %s', len(batched_data) + idx, generate_token_list)

        answer_results = [self.gpqa_postprocess(generate_token_list, "ABCD")
                        for generate_token_list in generate_token_lists]
        is_correct = ["Correct" if answer_result == label else "Wrong"
                    for answer_result, label in zip(answer_results, self.labels)]
        for idx, is_pass in enumerate(is_correct):
            metric.csv_debug.get("golden_result").append(self.labels[idx])
            metric.csv_debug.get("test_result").append(answer_results[idx])
            metric.csv_debug.get("pass").append(is_pass)
            if is_pass != "Correct":
                logger.debug(">>>推理结果 is : %s", answer_results[idx])
                logger.debug(">>>真实结果 is : %s", self.labels[idx])
            else:
                metric.correct_num += 1
                metric.correct_num_list[sub_dataset_idx] += 1

    def shuffle_data(self, data):
        if len(data) > 0 and isinstance(data[0], list) and len(data[0]) == 5:
            for i, item in enumerate(data):
                if i % 4 == 0:
                    data[i].append("A")
                elif i % 4 == 1:
                    data[i] = [item[0], item[2], item[3], item[4], item[1], "D"]
                elif i % 4 == 2:
                    data[i] = [item[0], item[3], item[4], item[1], item[2], "C"]
                else:
                    data[i] = [item[0], item[4], item[1], item[2], item[3], "B"]
            return data
        else:
            self.logger.error("Please check GPQA dataset!")
            return None

    def load_dataset_by_task_name(self):
        val_list = []
        row_begin_idx, col_begin_idx, col_end_idx = 1, 7, 12
        for subject_name in self.task_config.subject_mapping:
            origin_val_df = pd.read_csv(os.path.join(self.task_config.local_dataset_path, 
                                                     self.task_config.subject_mapping[subject_name]['name']), 
                                                     header=None)
            val_df = origin_val_df.iloc[row_begin_idx:, col_begin_idx:col_end_idx]
            val_data = val_df.values.tolist()
            val_data = self.shuffle_data(val_data)
            val_list.append(val_data)
        return val_list