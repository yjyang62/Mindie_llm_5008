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

import json
import os
import sys
import tempfile
import unittest
from unittest import mock
import torch
from mindie_llm.text_generator.mempool.base import MemPool
from mindie_llm.text_generator.mempool.mooncake_mempool import BASE_PORT


SCHEDULER = "scheduler"
WORKER = "worker"
MOONCAKE_BACKEND = "mooncake"
BASE_CONFIG = {
    "local_hostname": "hostA",
    "metadata_server": "meta:1234",
    "global_segment_size": 123,
    "local_buffer_size": 456,
    "protocol": "ascend",
    "device_name": "npu",
    "master_server_address": "master:5678",
    "use_ascend_direct": False
}


def _write_tmp_config(payload: dict) -> str:
    f = tempfile.NamedTemporaryFile("w+", delete=False)
    with f:
        json.dump(payload, f)
        f.flush()
    return f.name


class TestUnifiedCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg_json = BASE_CONFIG
        cls.cfg_path = _write_tmp_config(cls.cfg_json)
        cls._store_key = "mooncake.store"
        cls._prev_store_mod = sys.modules.get(cls._store_key)
        store_mod = mock.MagicMock()
        store_instance = mock.MagicMock()
        store_instance.setup.return_value = 0
        store_instance.is_exist.return_value = 1
        store_instance.batch_get_into_ascend.return_value = [0]
        store_instance.batch_put_from_ascend.return_value = [123]
        store_instance.close.return_value = None
        store_mod.MooncakeDistributedStore.return_value = store_instance
        sys.modules[cls._store_key] = store_mod
        cls.store_mod = store_mod
        cls.store_instance = store_instance
        cls.pool = MemPool.create_pool(MOONCAKE_BACKEND, cls.cfg_path, role=WORKER)
    
    @classmethod
    def tearDownClass(cls):
        if cls._prev_store_mod is not None:
            sys.modules[cls._store_key] = cls._prev_store_mod
        else:
            sys.modules.pop(cls._store_key, None)
        try:
            os.remove(cls.cfg_path)
        except OSError:
            pass

    def test_init_success(self):
        device_id = 0
        expected_suffix = self.cfg_json["local_hostname"] + ":" + str(BASE_PORT + device_id) + \
                ":" + "npu_" + str(device_id)
        args, _ = self.store_instance.setup.call_args
        self.assertTrue(str(args[0]).endswith(expected_suffix))
        self.assertEqual(args[1], self.cfg_json["metadata_server"])
        self.assertIn(args[2], [self.cfg_json["global_segment_size"], 0])
        self.assertIn(args[3], [self.cfg_json["local_buffer_size"], 0])
        self.assertIn(args[4], ["ascend", "ascend_no_transport"])
        self.assertEqual(args[5], self.cfg_json["device_name"])
        self.assertEqual(args[6], self.cfg_json["master_server_address"])

    def test_init_fail(self):
        with self.subTest("protocol not ascend"):
            bad_cfg_path = _write_tmp_config({**self.cfg_json, "protocol": "tcp"})
            try:
                with self.assertRaisesRegex(ValueError, r"protocol.*ascend"):
                    MemPool.create_pool(MOONCAKE_BACKEND, bad_cfg_path)
            finally:
                os.remove(bad_cfg_path)
        with self.subTest("store module missing"):
            prev = sys.modules.pop(self._store_key, None)
            try:
                with self.assertRaises(ImportError):
                    MemPool.create_pool(MOONCAKE_BACKEND, self.cfg_path)
            finally:
                if prev is not None:
                    sys.modules[self._store_key] = prev
        with self.subTest("store.setup nonzero"):
            self.store_instance.setup.return_value = -1
            with self.assertRaises(RuntimeError):
                MemPool.create_pool(MOONCAKE_BACKEND, self.cfg_path)
            self.store_instance.setup.return_value = 0

    def test_exists(self):
        self.store_instance.is_exist.return_value = 1
        self.assertTrue(self.pool.exists("abc"))
        self.store_instance.is_exist.return_value = 0
        self.assertFalse(self.pool.exists("def"))

    def test_exists_fail(self):
        for bad in [123, ["k"], {"k": 1}, None]:
            with self.subTest(bad=bad):
                self.store_instance.is_exist.reset_mock()
                self.assertFalse(self.pool.exists(bad))
                self.pool.store.is_exist.assert_not_called()

    def test_put_success(self):
        self.store_instance.is_exist.return_value = 0
        self.store_instance.batch_put_from_ascend.return_value = (0,)
        single_tensor1 = torch.rand(32, 128, 16)
        single_tensor2 = torch.rand(4, 128, 16)
        key_cases = [
            "k_single_str",
            ["k1_list", "k2_list"] 
        ]
        tensor_cases = [
            [single_tensor1, single_tensor2],
            [[single_tensor1, single_tensor1], [single_tensor2, single_tensor2]]
        ]
        with self.subTest("single key"):
            except_res = [True]
            ret = self.pool.put(key_cases[0], tensor_cases[0])
            self.assertEqual(ret, except_res)
        with self.subTest("multi keys"):
            except_res = [True] * len(key_cases[1])
            ret = self.pool.put(key_cases[1], tensor_cases[1])
            self.assertEqual(ret, except_res)
        with self.subTest("key has been existed"):
            self.store_instance.is_exist.return_value = 0
            self.assertEqual(self.pool.put("bad", single_tensor1), [True])

    def test_put_fail(self):
        single_tensor = torch.rand(2, 3)
        with self.subTest("len mismatch"):
            self.assertEqual(self.pool.put(["k1", "k2"], [single_tensor]), [False])
        with self.subTest("bad key type"):
            self.assertEqual(self.pool.put(123, single_tensor), [False])
        with self.subTest("backend error"):
            self.store_instance.is_exist.return_value = 0
            self.store_instance.batch_put_from_ascend.return_value = (-800,)
            self.assertEqual(self.pool.put("bad", single_tensor), [False])

    def test_get_success(self):
        with self.subTest("single key get"):
            single_tensor = torch.empty(2, 3)
            expect_size = single_tensor.numel() * single_tensor.element_size()
            self.store_instance.batch_get_into_ascend.return_value = expect_size
            self.assertEqual(self.pool.put("k", single_tensor), [True])
        with self.subTest("multi keys get"):
            tensor1 = torch.empty(2, 3)
            tensor2 = torch.empty(2, 3)
            expect_size = tensor1.numel() * tensor1.element_size() + tensor2.numel() * tensor2.element_size()
            self.store_instance.batch_get_into_ascend.return_value = expect_size
            self.assertEqual(self.pool.put(["k1", "k2"], [[tensor1, tensor2], [tensor1, tensor2]]), [True, True])

    def test_get_fail(self):
        single_tensor = torch.empty(2, 3)
        expect_size = single_tensor.numel() * single_tensor.element_size()
        with self.subTest("len mismatch"):
            self.assertEqual(self.pool.get(["k1", "k2"], [single_tensor]), [False])
        with self.subTest("bad key type"):
            self.assertEqual(self.pool.get(123, single_tensor), [False])
        with self.subTest("key dose not exist"):
            self.store_instance.is_exist.return_value = 0
            self.store_instance.batch_get_into_ascend.return_value = (expect_size,)
            self.assertEqual(self.pool.get("bad", single_tensor), [False])
        with self.subTest("backend error"):
            self.store_instance.is_exist.return_value = 1
            self.store_instance.batch_get_into_ascend.return_value = (-800,)
            self.assertEqual(self.pool.get("bad", single_tensor), [False])

    def test_put_get_use_ascend_direct(self):
        engine_key = "mooncake.engine"
        engine_mod = mock.MagicMock()
        tg_instance = mock.MagicMock()
        tg_instance.initialize.return_value = 0
        tg_instance.get_rpc_port.return_value = 9999
        tg_instance.get_engine.return_value = "fake_engine"
        engine_mod.TransferEngine.return_value = tg_instance
        sys.modules[engine_key] = engine_mod

        store_key = "mooncake.store"
        store_mod = mock.MagicMock()
        store_instance = mock.MagicMock()
        store_instance.setup.return_value = 0
        store_mod.MooncakeDistributedStore.return_value = store_instance
        sys.modules[store_key] = store_mod

        cfg = {**BASE_CONFIG, "use_ascend_direct": True}
        cfg_path = _write_tmp_config(cfg)

        t1 = torch.empty(4, 8, 16)
        t2 = torch.empty(4, 8, 16)
        kv_caches = [[t1, t2], [t1, t2]]
        keys = ["k1", "k2"]
        
        expected_t1 = t1.numel() * t1.element_size()
        expected_t2 = t2.numel() * t2.element_size()

        pool = MemPool.create_pool(MOONCAKE_BACKEND, cfg_path, role=WORKER, kv_caches=kv_caches)

        store_instance.batch_put_from_multi_buffers = mock.MagicMock(return_value=[0] * len(keys))
        store_instance.batch_get_into_multi_buffers = mock.MagicMock(return_value=[expected_t1, expected_t2])
        store_instance.batch_is_exist = mock.MagicMock(return_value=[1] * len(keys))

        ret = pool.batch_exist(keys)
        self.assertEqual(ret, [True] * len(keys))

        tensors = [[t1], [t2]]
        expected = [True, True]
        ret = pool.put(keys, tensors)
        self.assertEqual(ret, expected)
        store_instance.batch_put_from_multi_buffers.assert_called_once()
        
        tensors = [[t1], [t2]]
        ret = pool.get(keys, tensors)
        self.assertEqual(ret, [True, True])
        store_instance.batch_get_into_multi_buffers.assert_called_once()