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
from ddt import ddt, data, unpack
from atb_llm.models import router_model_type


FAKE_MODEL_NAME_OR_PATH = "fake_model_name_or_path_MiniCPM-V-2_6"
FAKE_CONFIG_DICT = {
    "_name_or_path": "yi-vl",
    "vision_config": None,
    "aligner_config": None
}


@ddt
class TestInit(unittest.TestCase):
    @data(
        ("kclgpt", "codeshell"),
        ("internvl_chat", "internvl"),
        ("llava_next_video", "llava_next"),
        ("bunny-qwen2", "bunny"),
        ("bunny-minicpm", "bunny"),
        ("deepseek_v2", "deepseekv2"),
        ("deepseek_v3", "deepseekv2"),
        ("vita-qwen2", "vita"),
        ("qwen2_5_vl", "qwen2_vl"),
        ("ernie4_5_moe", "ernie_moe"),
        ("llava", "yivl"),
        ("chatglm", "glm4v"),
        ("minicpmv", "minicpm_qwen2_v2"),
        ("multi_modality", "janus")
    )
    @unpack
    def test_router_model_type(self, original_model_type, expect_model_type):
        converted_model_type = router_model_type(
            original_model_type, FAKE_CONFIG_DICT, "model_type", FAKE_MODEL_NAME_OR_PATH)
        self.assertEqual(converted_model_type, expect_model_type)