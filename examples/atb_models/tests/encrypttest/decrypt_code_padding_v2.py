#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

from tests.encrypttest.decrypt_code_padding import process_file
from tests.encrypttest.crypted_weights_handler import CryptedWeightsFileHandler
from mindie_llm.runtime.utils.loader import default_model_loader


operations = [
    {
        'type': 'check_modified',
        'target_block': """from tests.encrypttest.crypted_weights_handler import CryptedWeightsFileHandler as WeightsFileHandler"""
    },
    {
        'type': 'insert',
        'target_block': """from mindie_llm.runtime.utils.loader.weight_utils import WeightsFileHandler""",
        'code': """from tests.encrypttest.crypted_weights_handler import CryptedWeightsFileHandler as WeightsFileHandler"""
    },
]


if __name__ == "__main__":
    target_file = default_model_loader.__file__
    if not os.path.exists(target_file):
        print(f"Error: file {target_file} does not exists")
        sys.exit(1)
    print(f"Start processing the file: {target_file}")
    success = process_file(target_file, operations)
    if success:
        print("All operations have been completed.!")
    else:
        print("No modifications were executed")