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
import random
import torch


class OperationFuzz:
    def __init__(self, operation_name):
        torch.npu.set_device(0)
        atb_speed_home_path = os.environ.get("ATB_SPEED_HOME_PATH")
        if atb_speed_home_path is None:
            raise RuntimeError("env ATB_SPEED_HOME_PATH not exist, source set_env.sh")
        lib_path = os.path.join(atb_speed_home_path, "lib/libatb_speed_torch.so")
        torch.classes.load_library(lib_path)
        self.op = torch.classes.OperationTorch.OperationTorch(operation_name)

    def set_name(self, name):
        self.op.set_name(name)

    def set_param(self, params):
        self.op.set_param(params)

    def execute(self):
        input_tensors = []
        input_tensors.append(torch.randint(1, 100, (random.randint(10, 1024),)).npu()) # intensor1
        input_tensors.append(torch.randint(1, 100, (random.randint(10, 1024),)).npu()) # intensor2

        self.op.execute(input_tensors)
    
    def execute_out(self):
        input_tensors = []
        input_tensors.append(torch.randint(1, 100, (random.randint(10, 1024),)).npu()) # intensor1
        input_tensors.append(torch.randint(1, 100, (random.randint(10, 1024),)).npu()) # intensor2
        output_tensors = []
        output_tensors.append(torch.randint(1, 100, (random.randint(10, 1024),)).npu()) # outtensor1
        output_tensors.append(torch.randint(1, 100, (random.randint(10, 1024),)).npu()) # outtensor2

        self.op.execute_out(input_tensors, output_tensors)
    
    def execute_with_param(self, param):
        input_tensors = []
        input_tensors.append(torch.randint(1, 100, (random.randint(10, 1024),)).npu()) # intensor1
        input_tensors.append(torch.randint(1, 100, (random.randint(10, 1024),)).npu()) # intensor2

        self.op.execute_with_param(input_tensors, param)
    
    def execute_out_with_param(self, param, num_input_tensors=2, num_output_tensors=2):
        input_tensors = []
        output_tensors = []
        for _ in range(num_input_tensors):
            input_tensors.append(torch.randint(1, 100, (random.randint(10, 1024),)).npu())
        for _ in range(num_output_tensors):
            output_tensors.append(torch.randint(1, 100, (random.randint(10, 1024),)).npu())
        self.op.execute_out_with_param(input_tensors, output_tensors, param)