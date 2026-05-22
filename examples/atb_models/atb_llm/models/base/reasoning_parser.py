# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from typing import Tuple, List

from atb_llm.utils.log import logger


class ReasoningParser:
    def __init__(self, config):
        self.start_reasoning_token_id = config.reasoning_config.start_reasoning_token_id
        self.end_reasoning_token_id = config.reasoning_config.end_reasoning_token_id

    def is_reasoning_end(self, all_token_ids: list):
        return self.end_reasoning_token_id in all_token_ids


class CommonReasoningParser(ReasoningParser):
    """
    Common implementation.
    compatibility without start <think>
    compatibility limited by length with unfinished
    Implementation logic: </think>: reasoning content on the left and content on the right
    """

    def __init__(self, config):
        super().__init__(config)

    def stream_process_reasoning(self, all_token_ids: list, current_index: int) -> Tuple[List[int], List[int]]:
        # compatibility without start <think>
        valid_text_start_index = 1 if self.start_reasoning_token_id == all_token_ids[0] else 0
        if len(all_token_ids) == 1 and valid_text_start_index == 1:
            return [], []
        delta_reasoning_content_token_ids = []
        delta_content_token_ids = []
        reasoning_end_token_index = len(all_token_ids) if self.end_reasoning_token_id not in all_token_ids \
            else all_token_ids.index(self.end_reasoning_token_id)
        # get reasoning delta
        if current_index < reasoning_end_token_index:
            delta_reasoning_content_token_ids = all_token_ids[current_index:reasoning_end_token_index]
        # get content delta
        if reasoning_end_token_index < len(all_token_ids) - 1:
            delta_content_token_ids = all_token_ids[max(1 + reasoning_end_token_index, current_index):]
        return delta_reasoning_content_token_ids, delta_content_token_ids

    def single_process_reasoning(self, all_token_ids: list) -> Tuple[List[int], List[int]]:
        reasoning_content_token_ids = []
        content_token_ids = []
        if not all_token_ids:
            return reasoning_content_token_ids, content_token_ids
        if self.end_reasoning_token_id is None:
            logger.error("ERROR: now in reasoning parser without given end_reasoning_token id.")
            return reasoning_content_token_ids, all_token_ids
        # compatibility without start <think>
        reasoning_content_start_index = 1 if self.start_reasoning_token_id == all_token_ids[0] else 0
        if self.end_reasoning_token_id not in all_token_ids:
            # compatibility limited by length with unfinished
            return all_token_ids[reasoning_content_start_index:], []
        # common scene
        reasoning_content_end_index = all_token_ids.index(self.end_reasoning_token_id)
        reasoning_content_token_ids = all_token_ids[reasoning_content_start_index:reasoning_content_end_index]
        content_token_ids = [] if reasoning_content_end_index == len(all_token_ids) - 1 \
            else all_token_ids[reasoning_content_end_index + 1:]
        return reasoning_content_token_ids, content_token_ids

    def count_reasoning_tokens(self, all_token_ids):
        try:
            return all_token_ids.index(self.end_reasoning_token_id)
        except ValueError:
            return 0