# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from base import model_test


class BloomModelTest(model_test.ModelTest):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.tokenizer_params = None

    @staticmethod
    def get_dataset_list():
        return ["CEval"]
    
    @staticmethod
    def get_chip_num():
        return 8

    def set_fa_tokenizer_params(self):
        self.tokenizer_params = {
            'revision': None,
            'trust_remote_code': self.trust_remote_code
        }

    def get_supported_model_type(self):
        return ["bloom"]


def main():
    BloomModelTest.create_instance()

if __name__ == "__main__":
    main()