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
from unittest.mock import patch, MagicMock
import sys
from mindie_llm.connector.request_listener.request_listener import RequestListener


class TestRequestListener(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.modules['_libatb_torch'] = MagicMock()

    @classmethod
    def tearDownClass(cls):
        del sys.modules['_libatb_torch']

    def setUp(self):
        RequestListener._instance = None
        self.mock_config = MagicMock()
        self.mock_config.shm_name_prefix = "test_prefix"
        self.mock_config.local_rank = 0
        self.mock_config.npu_num_per_dp = 1

    def test_singleton_pattern(self):
        instance1 = RequestListener.get_instance(self.mock_config)
        instance2 = RequestListener.get_instance(self.mock_config)

        self.assertIs(instance1, instance2)
        self.assertEqual(instance1.config, self.mock_config)

    def test_init_method(self):
        listener = RequestListener(self.mock_config)

        self.assertIsNone(listener.communication)
        self.assertEqual(listener.config, self.mock_config)

    @patch("mindie_llm.connector.request_listener.shared_mem_communication.SharedMemCommunication")
    def test_start_method(self, mock_shared_mem_comm):
        mock_comm_instance = MagicMock()
        mock_shared_mem_comm.get_instance.return_value = mock_comm_instance

        listener = RequestListener.get_instance(self.mock_config)
        result = listener.start()

        self.assertTrue(result)
        mock_shared_mem_comm.get_instance.assert_called_once_with(self.mock_config)
        mock_comm_instance.start.assert_called_once()
        self.assertEqual(listener.communication, mock_comm_instance)

    @patch("mindie_llm.connector.request_listener.shared_mem_communication.SharedMemCommunication")
    def test_stop_method(self, mock_shared_mem_comm):
        mock_comm_instance = MagicMock()
        mock_shared_mem_comm.get_instance.return_value = mock_comm_instance

        listener = RequestListener.get_instance(self.mock_config)
        listener.start()
        result = listener.stop()

        self.assertTrue(result)
        mock_comm_instance.stop.assert_called_once()

    def test_stop_before_start(self):
        listener = RequestListener.get_instance(self.mock_config)

        with self.assertRaises(AttributeError):
            listener.stop()


if __name__ == "__main__":
    unittest.main()