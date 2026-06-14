# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from base import model_test


class GlmModelTest(model_test.ModelTest):
    def __init__(self, *args) -> None:
        model_name = "glm4_moe"
        updated_args = args[:3] + (model_name,) + args[4:]
        super().__init__(*updated_args)

    @staticmethod
    def get_chip_num():
        return 8

    @staticmethod
    def get_dataset_list():
        return ["BoolQ", "CEval", "GSM8K"]

    def prepare_environ(self):
        pass

    def get_supported_model_type(self):
        return ["glm4_moe"]


def main():
    GlmModelTest.create_instance()

if __name__ == "__main__":
    main()