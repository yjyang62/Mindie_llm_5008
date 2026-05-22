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
import numpy as np

from mindie_llm.model_wrapper.utils.config import BaseConfig, DmiConfig, LinkMapParams


class TestBaseConfig(unittest.TestCase):
    def setUp(self):
        self.base_config = {
            "local_rank": "0",
            "rank": "0",
            "world_size": "2",
            "npu_device_id": "0",
            "npu_device_ids": "0,1",
            "cpu_mem": "1024",
            "npu_mem": "2048",
            "max_seq_len": "512",
            "max_iter_times": "100",
            "max_prefill_tokens": "1024",
            "block_size": "32",
            "model_id": "./model",
            "distributed_enable": "true",
            "globalWorldSize": "2"
        }

    def test_base_config_init(self):
        config = BaseConfig(self.base_config)
        self.assertIsNotNone(config.global_world_size)
        self.assertEqual(config.global_world_size, 2)
        self.assertEqual(config.world_size, 2)
        self.assertTrue(config.distributed_enable)

    def test_missing_required_field(self):
        invalid_config = self.base_config.copy()
        invalid_config.pop("model_id")
        with self.assertRaises(ValueError):
            BaseConfig(invalid_config)

    def test_parse_list(self):
        config = BaseConfig(self.base_config)
        self.assertEqual(config.parse_list("npu_device_ids", to_int=True), [0, 1])


class TestDmiConfig(unittest.TestCase):
    def setUp(self):
        self.dmi_config = {
            "local_rank": "0",
            "rank": "0",
            "world_size": "2",
            "npu_device_id": "0",
            "npu_device_ids": "0,1",
            "cpu_mem": "1024",
            "npu_mem": "2048",
            "max_seq_len": "512",
            "max_iter_times": "100",
            "max_prefill_tokens": "1024",
            "block_size": "32",
            "model_id": "./model",
            "distributed_enable": "true",
            "globalWorldSize": "2",
            "role": "prefill",
            "local_logic_device_id": "0,1",
            "local_physical_device_id": "10,11",
            "local_device_ip": "192.168.0.1,192.168.0.2",
            "local_host_ip": "10.0.0.1,10.0.0.2",
            "tp": "1",
            "local_instance_id": "instance_0"
        }

    def test_dmi_config_init_success(self):
        config = DmiConfig(self.dmi_config)
        self.assertIsNotNone(config.global_world_size)
        self.assertEqual(config.global_world_size, 2)
        self.assertIsNotNone(config.model_config["world_size"])
        self.assertEqual(int(config.model_config["world_size"]), 2)
        self.assertEqual(config.role, "prefill")

    def test_invalid_role_raises_error(self):
        invalid_config = self.dmi_config.copy()
        invalid_config["role"] = "invalid_role"
        with self.assertRaises(ValueError) as ctx:
            DmiConfig(invalid_config)
        self.assertIn("The pd_role should be prefill or decoder in DMI mode.", str(ctx.exception))

    def test_generate_link_map_matches_actual_logic(self):
        params = LinkMapParams(role="prefill", tp_p=4, tp_d=2)
        prefill_map = DmiConfig.generate_link_map(params)
        self.assertEqual(prefill_map, {0: [0], 2: [1]})
        params = LinkMapParams(role="decoder", tp_p=4, tp_d=2)
        decoder_map = DmiConfig.generate_link_map(params)
        self.assertEqual(decoder_map, {0: [0], 1: [2]})

        with self.assertRaises(ValueError):
            params = LinkMapParams(role="prefill", tp_p=0, tp_d=2)
            DmiConfig.generate_link_map(params)
        with self.assertRaises(ValueError):
            params = LinkMapParams(role="decoder", tp_p=4, tp_d=0)
            DmiConfig.generate_link_map(params)

    def test_generate_link_map_for_sp(self):
        params = LinkMapParams(role="prefill", tp_p=8, tp_d=1, sp_p=8, sp_d=1, cp_p=2, cp_d=1)
        prefill_sp_map = DmiConfig.generate_link_map(params)
        self.assertEqual(prefill_sp_map, {0: [0], 1: [0], 2: [0], 3: [0], 4: [0], 5: [0], 6: [0], 7: [0],
                                          8: [0], 9: [0], 10: [0], 11: [0], 12: [0], 13: [0], 14: [0], 15: [0]})
        params = LinkMapParams(role="decoder", tp_p=8, tp_d=1, sp_p=8, sp_d=1, cp_p=2, cp_d=1)
        decoder_sp_map = DmiConfig.generate_link_map(params)
        self.assertEqual(decoder_sp_map, {0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]})

    def test_clear_remote_info(self):
        config = DmiConfig(self.dmi_config)
        config.remote_sp_size = 5
        config.remote_link_host_ip = {"node1": "192.168.0.100"}
        config.clear_remote_info()
        self.assertEqual(config.remote_sp_size, 0)
        self.assertEqual(config.remote_link_host_ip, {})

    def test_set_pd_role(self):
        config = DmiConfig(self.dmi_config)
        config.set_pd_role(1)
        self.assertEqual(config.role, "prefill")
        config.set_pd_role(2)
        self.assertEqual(config.role, "decoder")
        config.set_pd_role(3)
        self.assertEqual(config.role, "flex")
        config.set_pd_role(4)
        self.assertEqual(config.role, "unknown")

    @patch("mindie_llm.model_wrapper.utils.config.logger")
    def test_set_pd_link_info(self, mock_logger):
        config = DmiConfig(self.dmi_config)
        mock_requests = [
            np.array([[1, 0, 0, 0, 0, 0, 0]], dtype=np.int64),
            np.array([[[0, 0, 0, 0, 0, 0, 0]]], dtype=np.int64),
            np.array([[0, 1, 1]], dtype=np.int64)
        ]
        config.set_pd_link_info(mock_requests)
        mock_logger.info.assert_any_call("[Config]\t>>> start to set PD link/unlink info according to the request.")
        self.assertEqual(config.remote_sp_size, 1)

    def test_dp_size_logic(self):
        dp_config = self.dmi_config.copy()
        dp_config["dp"] = "2"
        with patch("mindie_llm.model_wrapper.utils.config.generate_dp_inst_id",
                   return_value=["dp_inst_0", "dp_inst_1"]):
            config = DmiConfig(dp_config)
            self.assertEqual(config.model_config["local_instance_id"], "dp_inst_0")

    def test_set_pd_link_info_with_unlink(self):
        full_config = self.dmi_config.copy()
        full_config.update({
            "sp": "1",
            "local_super_device_id": "20,21",
            "tp": "1",
            "dp": "2",
            "local_instance_id": "1000"
        })
        with patch("mindie_llm.model_wrapper.utils.config.logger"), \
                patch("mindie_llm.model_wrapper.utils.config.generate_dp_inst_id",
                      return_value=["10000", "10001"]):
            config = DmiConfig(full_config)
            mock_requests = [
                np.array([[1, 0, 1, 1, 1, 0, 0]], dtype=np.int64),
                np.array([
                    [[192, 168, 0, 100, -1, -1, -1, -1, 1000], [192, 168, 0, 101, -1, -1, -1, -1, 1001]],
                    [[192, 168, 0, 200, -1, -1, -1, -1, 2000], [192, 168, 0, 201, -1, -1, -1, -1, 2001]]
                ], dtype=np.int64),
                np.array([[0, 1, 1]], dtype=np.int64)
            ]
            config.set_pd_link_info(mock_requests)

            self.assertIn(0, config.remote_unlink_cluster_id)
            self.assertIn(2000, config.remote_unlink_cluster_id[0])
            self.assertIn(0, config.remote_unlink_device_ips)
            self.assertEqual(
                config.remote_unlink_device_ips[0],
                ["192.168.0.201"]
            )


if __name__ == "__main__":
    unittest.main()