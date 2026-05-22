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
from unittest.mock import Mock, patch
import queue

from mindie_llm.connector.common.model_execute_data_pb2 import ExecuteRequest, ExecuteType
from mindie_llm.connector.request_router.request_router import RequestRouter


class TestRequestRouter(unittest.TestCase):
    def setUp(self):
        self.router = RequestRouter()
        self.router.router_impl = Mock()

    def test_init(self):
        self.assertIsInstance(self.router.inference_queue, queue.Queue)
        self.assertIsInstance(self.router.transfer_queue, queue.Queue)
        self.assertIsInstance(self.router.link_queue, queue.Queue)
        self.assertTrue(self.router.inference_related_thread.is_alive())
        self.assertTrue(self.router.trans_related_thread.is_alive())
        self.assertIsNone(self.router.enable_dp_distributed)

    @patch("mindie_llm.connector.request_router.request_router.BaseConfig")
    @patch("mindie_llm.connector.request_router.request_router.send_model_execute_response")
    @patch("mindie_llm.connector.request_router.request_router.RouterImpl")
    def test_initialize_standard_mode(self, mock_router_impl_cls, mock_send, mock_base_config):
        mock_config = Mock()
        mock_config.items.return_value = [
            ("infer_mode", "standard"),
            ("cpu_mem", "2048"),
            ("model_path", "/path/to/standard/model"),
            ("device", "npu"),
            ("distributed_enable", True)
        ]

        mock_base_config_instance = Mock()
        mock_model_config = dict(mock_config.items.return_value)
        mock_base_config_instance.model_config = mock_model_config
        mock_base_config.return_value = mock_base_config_instance

        mock_router_impl_instance = Mock()
        mock_router_impl_instance.initialize.return_value = {"status": "ok"}
        mock_router_impl_cls.return_value = mock_router_impl_instance

        self.router.initialize(mock_config)

        mock_base_config.assert_called_once_with(mock_model_config)
        mock_router_impl_instance.initialize.assert_called_once_with(mock_base_config_instance)
        mock_send.assert_called_once()
        self.assertEqual(self.router.enable_dp_distributed, mock_base_config_instance.distributed_enable)

    @patch("mindie_llm.connector.request_router.request_router.DmiConfig")
    @patch("mindie_llm.connector.request_router.request_router.send_model_execute_response")
    @patch("mindie_llm.connector.request_router.request_router.RouterImpl")
    def test_initialize_dmi_mode(self, mock_router_impl_cls, mock_send, mock_dmi_config):
        mock_config = Mock()
        mock_config.items.return_value = [
            ("infer_mode", "dmi"),
            ("cpu_mem", "1024"),
            ("model_path", "/path/to/model"),
            ("device", "npu"),
            ("distributed_enable", False)
        ]

        mock_dmi_config_instance = Mock()
        mock_model_config = dict(mock_config.items.return_value)
        mock_dmi_config_instance.model_config = mock_model_config
        mock_dmi_config.return_value = mock_dmi_config_instance

        mock_router_impl_instance = Mock()
        mock_router_impl_instance.initialize.return_value = {"status": "ok"}
        mock_router_impl_cls.return_value = mock_router_impl_instance

        self.router.initialize(mock_config)

        mock_dmi_config.assert_called_once_with(mock_model_config)
        mock_router_impl_cls.assert_called_once()
        mock_router_impl_instance.initialize.assert_called_once_with(mock_dmi_config_instance)
        mock_send.assert_called_once()
        self.assertEqual(self.router.enable_dp_distributed, mock_dmi_config_instance.distributed_enable)

    @patch("mindie_llm.connector.request_router.request_router.logger")
    def test_initialize_invalid_mode(self, mock_logger):
        mock_config = Mock()
        mock_config.items.return_value = [("infer_mode", "invalid")]

        with self.assertRaises(UnboundLocalError):
            self.router.initialize(mock_config)
        mock_logger.error.assert_called_once()

    def test_accept_inference_request(self):
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = ExecuteType.MODEL_INFER

        self.router.accept(mock_request)
        self.assertEqual(self.router.inference_queue.qsize(), 1)
        self.assertEqual(self.router.transfer_queue.qsize(), 0)
        self.assertEqual(self.router.link_queue.qsize(), 0)

    def test_accept_transfer_request(self):
        kv_request = Mock(spec=ExecuteRequest)
        kv_request.execute_type = ExecuteType.KV_TRANSFER
        self.router.accept(kv_request)

        self.assertEqual(self.router.transfer_queue.qsize(), 1)

    def test_accept_link_request(self):
        link_request = Mock(spec=ExecuteRequest)
        link_request.execute_type = ExecuteType.PD_LINK
        self.router.accept(link_request)

        self.assertEqual(self.router.link_queue.qsize(), 1)
        self.assertEqual(self.router.transfer_queue.qsize(), 0)

    def test_accept_command_request(self):
        cmd_request = Mock(spec=ExecuteRequest)
        # LORA_OPERATION 应该进入 command_queue 分支
        cmd_request.execute_type = ExecuteType.LORA_OPERATION
        self.router.accept(cmd_request)

        self.assertEqual(self.router.command_queue.qsize(), 1)

    @patch.object(RequestRouter, "do_inference")
    def test_accept_other_request(self, _):
        router = RequestRouter()
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = ExecuteType.MODEL_INIT
        router.accept(mock_request)

        self.assertEqual(router.inference_queue.qsize(), 1)

    @patch("mindie_llm.utils.status.os._exit")
    def test_core_thread_exit_on_exception(self, mock_exit):
        # 核心线程在目标函数抛异常时应调用 os._exit(1)
        from mindie_llm.utils.status import CoreThread

        def bad_target():
            raise RuntimeError("boom")

        t = CoreThread(target=bad_target, name="core")
        t.start()
        t.join(timeout=1)
        mock_exit.assert_called_once_with(1)

    @patch("mindie_llm.utils.status.CoreThread")
    def test_do_inference_model_infer(self, _):
        router = RequestRouter()
        router.router_impl = Mock()
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = ExecuteType.MODEL_INFER

        with patch.object(router.inference_queue, 'get', side_effect=[mock_request, StopIteration]):
            with self.assertRaises(StopIteration):
                router.do_inference()

        router.router_impl.execute.assert_called_once_with(mock_request)

    @patch("mindie_llm.utils.status.CoreThread")
    def test_do_inference_model_infer_second(self, _):
        router = RequestRouter()
        router.router_impl = Mock()
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = ExecuteType.MODEL_INFER_SECOND

        with patch.object(router.inference_queue, 'get', side_effect=[mock_request, StopIteration]):
            with self.assertRaises(StopIteration):
                router.do_inference()

        router.router_impl.execute.assert_called_once_with(mock_request)

    @patch("mindie_llm.utils.status.CoreThread")
    @patch("mindie_llm.connector.request_router.request_router.logger")
    def test_do_inference_model_finalize(self, mock_logger, _):
        router = RequestRouter()
        router.router_impl = Mock()
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = ExecuteType.MODEL_FINALIZE

        with patch.object(router.inference_queue, 'get', side_effect=[mock_request, StopIteration]):
            with self.assertRaises(StopIteration):
                router.do_inference()

        router.router_impl.finalize.assert_called_once()
        mock_logger.info.assert_called_with("[python thread: infer] model finalized.")

    @patch("mindie_llm.utils.status.CoreThread")
    def test_do_inference_text_generator_cleanup(self, _):
        router = RequestRouter()
        router.router_impl = Mock()
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = ExecuteType.TEXT_GENERATOR_CLEANUP

        with patch.object(router.inference_queue, 'get', side_effect=[mock_request, StopIteration]):
            with self.assertRaises(StopIteration):
                router.do_inference()

        router.router_impl.seq_ctrl.assert_called_once_with(mock_request)

    @patch("mindie_llm.utils.status.CoreThread")
    @patch("mindie_llm.connector.request_router.request_router.logger")
    def test_do_inference_unknown_type(self, mock_logger, _):
        router = RequestRouter()
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = 9999  # 未知类型

        with patch.object(router.inference_queue, 'get', side_effect=[mock_request]):
            try:
                router.do_inference()
            except StopIteration:
                pass
            mock_logger.error.assert_called()


if __name__ == "__main__":
    unittest.main()