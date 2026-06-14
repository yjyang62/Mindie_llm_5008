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

import argparse
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')


def is_torch_use_cxx11():
    """
    torch cxx 11 判断
    : return:
    """
    import torch
    return torch.compiled_with_cxx11_abi()


def get_cxx11_flag(framework):
    """
    根据不同框架判断是否使用cxx11
    : param framework
    """
    cxx11_abi_flag_1 = "ON"
    cxx11_abi_flag_0 = "OFF"
    use_cxx11_flag = True
    if framework == "torch":
        use_cxx11_flag = is_torch_use_cxx11()
    return cxx11_abi_flag_1 if use_cxx11_flag else cxx11_abi_flag_0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--framework", default='torch', help='framework to be used')
    args = parser.parse_args()
    logging.info(get_cxx11_flag(args.framework))
