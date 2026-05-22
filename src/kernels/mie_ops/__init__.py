# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import os
import pkgutil
import warnings
import torch

__all__ = list(module for _, module, _ in pkgutil.iter_modules([os.path.dirname(__file__)]))

# 导入so 和 python
so_directory = os.path.dirname(os.path.abspath(__file__))

os.environ['ASCEND_CUSTOM_OPP_PATH'] = so_directory + '/opp/vendors/customize/' + ':' + os.environ.get('ASCEND_CUSTOM_OPP_PATH', '')
os.environ['LD_LIBRARY_PATH'] = so_directory + '/opp/vendors/customize/op_api/lib/' + ':' + os.environ.get('LD_LIBRARY_PATH', '')
# 遍历目录，加载所有 .so 文件
for filename in os.listdir(so_directory):
    if filename.endswith(".so"):
        filepath = os.path.join(so_directory, filename)
        torch.ops.load_library(filepath)
