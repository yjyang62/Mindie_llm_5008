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
import tempfile
import unittest
from unittest import mock
from mindie_llm.text_generator.mempool.base import MemPool


def _write_tmp_config(payload: dict) -> str:
    f = tempfile.NamedTemporaryFile("w+", delete=False)
    with f:
        json.dump(payload, f)
        f.flush()
    return f.name


class CreatePoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg_json = {
            "local_hostname": "hostA",
            "metadata_server": "meta:1234",
            "global_segment_size": 123,
            "local_buffer_size": 456,
            "protocol": "ascend",
            "device_name": "npu",
            "master_server_address": "master:5678"
        }
        cls.cfg_path = _write_tmp_config(cls.cfg_json)
    
    @classmethod
    def tearDownClass(cls):
        try:
            os.remove(cls.cfg_path)
        except OSError:
            pass

    def test_import_mooncake_mempool_error(self):
        backend = "mooncake"

        def fake_import(name, globals_=None, locals_=None, fromlist=(), level=0):
            if level == 1 and name == "mooncake_mempool" and fromlist == ("MooncakeMempool",):
                raise ImportError("boom")
            return origin_import(name, globals_, locals_, fromlist, level)
        
        origin_import = __import__
        with mock.patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaises(ImportError):
                MemPool.create_pool(backend, self.cfg_path)

    def test_import_unifiedcache_mempool_error(self):
        backend = "unifiedcache"

        def fake_import(name, globals_=None, locals_=None, fromlist=(), level=0):
            if level == 1 and name == "unifiedcache_mempool" and fromlist == ("UnifiedCacheMempool",):
                raise ImportError("boom")
            return origin_import(name, globals_, locals_, fromlist, level)
        
        origin_import = __import__
        with mock.patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaises(ImportError):
                MemPool.create_pool(backend, self.cfg_path)

    def test_import_unknown_mempool_error(self):
        backend = "unknown"
        with self.assertRaises(ValueError):
            MemPool.create_pool(backend, self.cfg_path)