#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import os
import unittest
from unittest.mock import patch
from mindie_llm.utils.env import EnvVar

WORLD_SIZE = "WORLD_SIZE"


class TestEnvVar(unittest.TestCase):
    def setUp(self):
        os.environ.clear()
        os.environ.update({
            "RESERVED_MEMORY_GB": "4",
            "ASCEND_RT_VISIBLE_DEVICES": "0,1",
            "BIND_CPU": "1",
            "NPU_MEMORY_FRACTION": "0.8",
            "MINDIE_LLM_BENCHMARK_ENABLE": "1",
            "MINDIE_LLM_HOME_PATH" : "/home", 
            "MINDIE_LLM_BENCHMARK_FILEPATH": "/tmp/benchmark.jsonl",
            "MINDIE_LOG_LEVEL": "DEBUG",
            "MINDIE_LLM_USE_MB_SWAPPER": "1",
            "POST_PROCESSING_SPEED_MODE_TYPE": "1",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "2"
        })

    def test_default_initialization(self):
        env_var = EnvVar()
        self.assertEqual(env_var.reserved_memory_gb, 4)
        self.assertEqual(env_var.visible_devices, [0, 1])
        self.assertTrue(env_var.bind_cpu)
        self.assertAlmostEqual(env_var.memory_fraction, 0.8)
        self.assertTrue(env_var.benchmark_enable)
        self.assertEqual(env_var.benchmark_filepath, "/tmp/benchmark.jsonl")
        self.assertTrue(env_var.use_mb_swapper)
        self.assertEqual(env_var.speed_mode_type, 1)
        self.assertEqual(env_var.rank, 0)
        self.assertEqual(env_var.local_rank, 0)
        self.assertEqual(env_var.world_size, 2)

    def test_invalid_reserved_memory_gb(self):
        with patch.dict(os.environ, {"RESERVED_MEMORY_GB": "-1"}):
            with self.assertRaises(ValueError):
                EnvVar()

        with patch.dict(os.environ, {"RESERVED_MEMORY_GB": "64"}):
            with self.assertRaises(ValueError):
                EnvVar()

    def test_invalid_visible_devices(self):
        with patch.dict(os.environ, {"ASCEND_RT_VISIBLE_DEVICES": "0,a,2"}):
            with self.assertRaises(ValueError):
                EnvVar()

    def test_invalid_memory_fraction(self):
        with patch.dict(os.environ, {"NPU_MEMORY_FRACTION": "1.1"}):
            with self.assertRaises(ValueError):
                EnvVar()

        with patch.dict(os.environ, {"NPU_MEMORY_FRACTION": "0"}):
            with self.assertRaises(ValueError):
                EnvVar()

    def test_invalid_world_size_and_rank(self):
        with patch.dict(os.environ, {WORLD_SIZE: "-1"}):
            with self.assertRaises(ValueError):
                EnvVar()

        with patch.dict(os.environ, {WORLD_SIZE: "2", "RANK": "2"}):
            with self.assertRaises(ValueError):
                EnvVar()

        with patch.dict(os.environ, {WORLD_SIZE: "2", "LOCAL_RANK": "2"}):
            with self.assertRaises(ValueError):
                EnvVar()

    def test_invalid_benchmark_filepath(self):
        with patch.dict(os.environ, {"MINDIE_LLM_BENCHMARK_FILEPATH": "relative/path/benchmark.jsonl"}):
            with self.assertRaises(ValueError):
                EnvVar()

        with patch.dict(os.environ, {"MINDIE_LLM_BENCHMARK_FILEPATH": "/tmp/"}):
            with self.assertRaises(ValueError):
                EnvVar()

if __name__ == "__main__":
    unittest.main()
