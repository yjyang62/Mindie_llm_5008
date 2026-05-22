# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import unittest
from ddt import ddt
from atb_llm.models.internlm3.config_internlm3 import Internlm3Config

FAKE_CONFIG_DICT = {
    'model_type': 'internlm3',
    'num_hidden_layers': 2,
    'max_position_embeddings': 2048,
    'vocab_size': 65536
}


@ddt
class TestInternlm3Config(unittest.TestCase):
    def setUp(self):
        self.internlm3_config = Internlm3Config(**FAKE_CONFIG_DICT)

    def test_init(self):
        self.assertEqual(self.internlm3_config.model_type, 'internlm3')


if __name__ == '__main__':
    unittest.main()