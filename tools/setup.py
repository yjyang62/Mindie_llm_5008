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

import argparse
import os
import sys
from setuptools import setup
os.environ['SOURCE_DATE_EPOCH'] = '0'

parser = argparse.ArgumentParser(description="LLM Manager Setup Parameters")
parser.add_argument("--setup_cmd", type=str, default="bdist_wheel")
parser.add_argument("--version", type=str, default="1.0.RC1")

args = parser.parse_args()
sys.argv = [sys.argv[0], args.setup_cmd]
infer_engine_version = args.version

setup(
    name='llm_manager_python_api_demo',
    version=infer_engine_version,
    description='llm manager python api demo',
    package_dir={'llm_manager_python_api_demo': 'llm_manager_python_api_demo'},
    package_data={
        '': ['llm_manager_python.so']
    },
    include_package_data=True,
    python_requires=">=3.10",
)