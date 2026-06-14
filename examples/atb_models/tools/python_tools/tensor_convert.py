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

import sys
import os
import torch
from atb_llm.utils.log import logger
from atb_llm.utils import file_utils

sys.path.append(os.path.dirname(__file__))
from tensor_file import read_tensor  # NOQA: E402


def main():
    tensor = read_tensor(sys.argv[1])
    logger.info(f"tensor:{tensor}")
    tensor_save_path = file_utils.standardize_path(sys.argv[2])
    file_utils.check_file_safety(tensor_save_path, 'w', is_check_file_size=False)
    torch.save(tensor, tensor_save_path)


if __name__ == "__main__":
    main()
