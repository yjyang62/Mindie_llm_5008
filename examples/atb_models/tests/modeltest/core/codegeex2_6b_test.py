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

from base import model_test


class CodegeeX26BModelTest(model_test.ModelTest):
    def __init__(self, *args) -> None:
        super().__init__(*args)
    
    @staticmethod
    def get_dataset_list():
        return ["HumanEval"]
    
    def prepare_environ(self):
        # memory
        os.environ['ATB_LAYER_INTERNAL_TENSOR_REUSE'] = "1"
        os.environ['ATB_WORKSPACE_MEM_ALLOC_GLOBAL'] = "1"
        # performance
        os.environ['ATB_OPERATION_EXECUTE_ASYNC'] = "1"
        os.environ['ATB_CONVERT_NCHW_TO_ND'] = "1"
        os.environ['TASK_QUEUE_ENABLE'] = "1"

    def get_supported_model_type(self):
        return ["chatglm"]


def main():
    CodegeeX26BModelTest.create_instance()

if __name__ == "__main__":
    main()