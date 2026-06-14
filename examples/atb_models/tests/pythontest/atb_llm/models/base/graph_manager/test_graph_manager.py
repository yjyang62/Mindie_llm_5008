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
from unittest.mock import patch

from atb_llm.models.base.graph_manager.graph_manager import ATBGraphManager
from atb_llm.models.base.graph_manager.graph_wrapper import ATBGraphWrapper


class MockATBGraphWrapper(ATBGraphWrapper):
    def __init__(self, feature_name):
        super().__init__()
        self.feature_name = feature_name
        self.feature_params = {}
        self.engine = None

    def set_param(self, model_type, params):
        pass

    def set_weight(self, weights):
        pass

    def set_kv_cache(self, k_caches, v_caches):
        pass
    
    def execute(self, inputs, runtime_params):
        return 1

    def activate(self, context, runtime_params, **kwargs) -> bool:
        return True


class TestATBGraphManager(unittest.TestCase):
    def setUp(self):
        self.graph_manager = ATBGraphManager()
        self.model_type = "test_class"   

    @patch("atb_llm.models.base.graph_manager.graph_wrapper.ATBGraphWrapper.set_param")
    @patch("atb_llm.models.base.graph_manager.graph_wrapper.ATBGraphWrapper.set_weight")
    @patch("atb_llm.models.base.graph_manager.graph_wrapper.ATBGraphWrapper.set_kv_cache")
    @patch("atb_llm.models.base.graph_manager.graph_wrapper.ATBGraphWrapper.execute")
    def test_set_param(self, mock_execute, _, __, ___):
        mock_execute.return_value = 10
        self.graph_manager.register_graph(MockATBGraphWrapper("flashcomm"))
        param = {}
        specified_param = {"decode": {}}
        self.graph_manager.set_param(self.model_type, param, specified_param)
        self.assertEqual(len(self.graph_manager._graph_list), 3)

        weights = 1
        specified_weights = {"decode": 2}
        self.graph_manager.set_weight(weights, specified_weights)

        self.graph_manager.set_kv_cache(3, 4)

        out = self.graph_manager.select_and_execute(None, 5, {}, is_prefill=True)
        self.assertEqual(out, 10)