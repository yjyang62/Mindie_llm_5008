# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import os
from base import model_test


class ErnieModelTest(model_test.ModelTest):
    def __init__(self, *args) -> None:
        model_name = "ernie"
        updated_args = args[:3] + (model_name,) + args[4:]
        super().__init__(*updated_args)
    
    @staticmethod
    def get_chip_num():
        return 4

    @staticmethod
    def get_dataset_list():
        return ["BoolQ", "CEval", "MMLU", "GSM8K"]
    
    def prepare_environ(self):
        os.environ['ATB_LAYER_INTERNAL_TENSOR_REUSE'] = "1"
        os.environ['ATB_OPERATION_EXECUTE_ASYNC'] = "1"
        os.environ['ATB_CONVERT_NCHW_TO_ND'] = "1"
        os.environ['TASK_QUEUE_ENABLE'] = "1"
        os.environ['ATB_CONTEXT_WORKSPACE_SIZE'] = "0"
        os.environ['ATB_LAUNCH_KERNEL_WITH_TILING'] = "1"
        os.environ['ATB_LLM_ENABLE_AUTO_TRANSPOSE'] = "0"
    
    def get_supported_model_type(self):
        return ["ernie4_5_moe"]


def main():
    ErnieModelTest.create_instance()

if __name__ == "__main__":
    main()