# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from typing import List, Union

from atb_llm.utils.env import ENV
from atb_llm.utils.log import logger, print_log
from atb_llm.utils.argument_utils import MAX_KEY_LENGTH


class Postprocessor:
    """A stop checker only used in run_pa.py"""
    def __init__(self, tokenizer, generation_config):
        self.max_new_tokens: int = generation_config.max_new_tokens
        self.pad_token_id: int = generation_config.pad_token_id
        self.eos_token_id: Union[int, List[Union[int, List[int]]]] = generation_config.eos_token_id
        if not self.eos_token_id:
            self.eos_token_id = tokenizer.eos_token_id

    def stopping_criteria(self, output_ids):
        """Check if the output ids encounter eos token id."""
        ret = False
        rank = ENV.rank
        if isinstance(self.eos_token_id, int):
            ret = output_ids[-1] == self.eos_token_id
        elif isinstance(self.eos_token_id, list):
            is_end_list = []
            for eos in self.eos_token_id:
                if isinstance(eos, int):
                    is_end_list.append(output_ids[-1] == eos)
                elif isinstance(eos, list):
                    is_end_list.append(len(output_ids) >= len(eos) and output_ids[-len(eos):] == eos)
                else:
                    print_log(rank, logger.warning, f"unsupport type of eos_token_id: "
                              f"{self.eos_token_id}.\nPlease check the type of your eos_token_id. It must be "
                              f"Union[int, List[Union[int, List[int]]]].")
            ret = any(is_end_list)
        else:
            print_log(rank, logger.warning, f"unsupport type of eos_token_id: "
                      f"{self.eos_token_id}.\nPlease check the type of your eos_token_id. It must be "
                      f"Union[int, List[Union[int, List[int]]]].")
        
        if ENV.modeltest_dataset_specified:
            ENV.update()
            if isinstance(ENV.modeltest_dataset_specified, str):
                if len(ENV.modeltest_dataset_specified) > MAX_KEY_LENGTH:
                    raise ValueError("The length of environment variable `MODELTEST_DATASET_SPECIFIED` "
                        f"should be no larger than {MAX_KEY_LENGTH}.")
                split_parts = ENV.modeltest_dataset_specified.split('_')
                if len(split_parts) >= 3 and "HumanEval" in split_parts[0]:
                    from tests.modeltest.modeltest.task.humanevalx import is_code_generation_finished
                    text = self.tokenizer.decode(output_ids, skip_special_tokens=False)
                    ret = is_code_generation_finished(
                        text,
                        language_type=split_parts[2],
                        dataset=split_parts[0],
                    )
        return ret