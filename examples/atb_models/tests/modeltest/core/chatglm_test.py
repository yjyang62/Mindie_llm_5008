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
import json

from base import model_test
from atb_llm.utils.file_utils import safe_open


class ChatGLMModelTest(model_test.ModelTest):
    def __init__(self, *args) -> None:
        weight_dir = args[10]
        model_name = "chatglm"
        config_path = os.path.join(weight_dir, "config.json")
        name_or_path = "_name_or_path"
        with safe_open(config_path, 'r') as f:
            self.config_data = json.load(f)
            if name_or_path not in self.config_data.keys():
                model_name = "glm-4"
            else:
                model_name = self.config_data[name_or_path].split('/')[-1]

        updated_args = args[:3] + (model_name,) + args[4:]
        super().__init__(*updated_args)


    @staticmethod   
    def get_dataset_list():
        return ["BoolQ", "CEval"]

    def prepare_environ(self):
        os.environ['INF_NAN_MODE_ENABLE'] = "0"
        # memory
        os.environ['ATB_LAYER_INTERNAL_TENSOR_REUSE'] = "1"
        os.environ['ATB_WORKSPACE_MEM_ALLOC_GLOBAL'] = "1"
        # performance
        os.environ['ATB_OPERATION_EXECUTE_ASYNC'] = "1"
        os.environ['ATB_CONVERT_NCHW_TO_ND'] = "1"
        os.environ['TASK_QUEUE_ENABLE'] = "1"

    def get_supported_model_type(self):
        if self.config_data["model_type"] == "glm":
            return ["glm"]
        return ["chatglm"]


def main():
    ChatGLMModelTest.create_instance()

if __name__ == "__main__":
    main()