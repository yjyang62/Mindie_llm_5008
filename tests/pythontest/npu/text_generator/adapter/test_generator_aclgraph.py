# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import unittest
from unittest.mock import MagicMock, patch

from mindie_llm.text_generator.adapter.generator_aclgraph import GeneratorAclGraph


class TestGeneratorAclGraphRecovery(unittest.TestCase):
    @patch("mindie_llm.text_generator.adapter.generator_aclgraph.torch.distributed.barrier")
    @patch("mindie_llm.text_generator.adapter.generator_aclgraph.torch.distributed.get_world_size", return_value=2)
    @patch("mindie_llm.text_generator.adapter.generator_aclgraph.torch.distributed.is_initialized", return_value=True)
    @patch("mindie_llm.text_generator.adapter.generator_aclgraph.reset_distributed_comm_state_after_reinit")
    @patch("mindie_llm.text_generator.adapter.generator_aclgraph.torch_npu.distributed.reinit_process_group")
    @patch("mindie_llm.text_generator.adapter.generator_aclgraph.torch_npu.npu.restart_device")
    def test_reinit_rebuilds_hccl_links_and_syncs_ranks(
        self,
        mock_restart_device,
        mock_reinit_process_group,
        mock_reset_comm_state,
        _mock_is_initialized,
        _mock_get_world_size,
        mock_barrier,
    ):
        backend = object.__new__(GeneratorAclGraph)
        backend.npu_device_id = 0
        backend.model_wrapper = MagicMock()

        backend._execute_cmd_reinit_npu()

        mock_restart_device.assert_called_once_with(0)
        mock_reinit_process_group.assert_called_once_with(rebuild_link=True)
        mock_reset_comm_state.assert_called_once_with(backend.model_wrapper.model)
        mock_barrier.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
