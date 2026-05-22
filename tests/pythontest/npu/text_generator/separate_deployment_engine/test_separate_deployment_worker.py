# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import time
import unittest
from unittest.mock import MagicMock, patch, call

from llm_datadist import DataType, RegisterMemStatus, LLMStatusCode, LLMException
from mindie_llm.text_generator.utils.separate_deployment_engine import (
    SeparateDeploymentWorker, SeparateDeploymentEngine
)
from mindie_llm.utils.log.logging import logger
from mindie_llm.utils.log.error_code import ErrorCode
from mindie_llm.utils.status import MindieLlmStatusCode


class TestSeparateDeploymentWorker(unittest.TestCase):
    def setUp(self):

        self.test_name = self._testMethodName  # 获取当前用例名称
        self.class_name = self.__class__.__name__  # 获取当前类名
        logger.info("=" * 80)
        logger.info(f"开始执行测试用例：{self.class_name}.{self.test_name}")
        logger.info("=" * 80)

        # Patch 掉 LLMDataDist 的构造函数，注意路径需要与实际代码中导入时保持一致
        patcher = patch("mindie_llm.text_generator.utils.separate_deployment_engine.LLMDataDist")
        self.addCleanup(patcher.stop)
        self.mock_llm_datadist = patcher.start()

        # 构造一个模拟的 LLMDataDist 实例
        self.mock_llm_data_dist_instance = MagicMock()
        # 模拟 link 返回一个通信 id
        self.mock_llm_data_dist_instance.link.return_value = 123
        # 模拟 unlink 返回 True
        self.mock_llm_data_dist_instance.unlink.return_value = MindieLlmStatusCode.SUCCESS
        # 模拟查询内存注册状态（默认返回 OK）
        self.mock_llm_data_dist_instance.query_register_mem_status.return_value = RegisterMemStatus.OK
        # 模拟 pull_kv 返回成功
        self.mock_llm_data_dist_instance.pull_kv.return_value = MindieLlmStatusCode.SUCCESS
        # 模拟 swap_in/swap_out 与 finalize 方法
        self.mock_llm_data_dist_instance.swap_in.return_value = None
        self.mock_llm_data_dist_instance.swap_out.return_value = None
        self.mock_llm_data_dist_instance.finalize.return_value = None

        # 模拟 cache_manager 对象
        self.mock_cache_manager = MagicMock()
        self.mock_llm_data_dist_instance.cache_manager = self.mock_cache_manager

        # 让 LLMDataDist 构造函数返回我们的模拟实例
        self.mock_llm_datadist.return_value = self.mock_llm_data_dist_instance

        # 创建 SeparateDeploymentWorker 实例（参数可随意设置，只要符合业务要求）
        self.worker = SeparateDeploymentWorker(
            role='decoder',
            local_logic_device_id=0,
            local_physical_device_id=1,
            local_cluster_id=0,
            local_device_ip="192.168.1.1",
            local_host_ip="127.0.0.1",
            local_super_pod_id=0,
            local_super_device_id=8716291
        )
        # 保证 worker 内部使用的是我们的 mock 实例
        self.worker.separate_deployment_engine = self.mock_llm_data_dist_instance

    def test_init_invalid_role(self):
        with self.assertRaises(Exception) as context:
            _ = SeparateDeploymentWorker(
                role='invalid',
                local_logic_device_id=0,
                local_physical_device_id=1,
                local_cluster_id=0,
                local_device_ip="192.168.1.1",
                local_host_ip="127.0.0.1",
                local_super_pod_id=0,
                local_super_device_id=8716291,
                kv_link_timeout=0
            )
        self.assertIn("SeparateDeploymentEngine: not support role.", str(context.exception))

    def test_build(self):
        """测试 build 方法生成正确的 cache 描述信息"""
        self.worker.build(model_id=0, num_tensors=1, num_blocks=10, blockshape=(2, 2), dtype="float16")
        cache_desc = self.worker.cache_desc_map[0]
        # 检查设备端缓存描述 shape 是否为 (10, 2, 2)
        self.assertEqual(cache_desc.shape, (10, 2, 2))
        # max_block_nums 应该保存为 10
        self.assertEqual(self.worker.max_block_nums_map[0], 10)
        # 检查数据类型是否映射正确
        self.assertEqual(cache_desc.data_type, DataType.DT_FLOAT16)

    def test_set_npu_cache(self):
        """测试 set_npu_cache 方法，确保调用了 register_blocks_cache 和 set_npu_cache 方法"""
        dummy_npu_cache = "dummy_npu_cache"
        self.worker.cache_desc_map[0] = MagicMock()
        self.mock_llm_data_dist_instance.register_blocks_cache.return_value = dummy_npu_cache
        self.worker.build(model_id=0, num_tensors=1, num_blocks=10, blockshape=(2, 2), dtype="float16")
        self.worker.set_npu_cache(model_id=0, npu_addrs=[4, 5, 6])
        self.mock_llm_data_dist_instance.register_blocks_cache.assert_called_once()
        self.mock_llm_data_dist_instance.set_npu_cache.assert_called_once_with(0, dummy_npu_cache)

    def test_pull_blocks_invalid_model(self):
        """测试 pull_blocks，当 remote_model_instance_id 不存在时返回错误码"""
        ret = self.worker.pull_blocks(remote_model_instance_id=999, src_block_table=[0, 1], dst_block_table=[0, 1])
        self.assertEqual(ret, ErrorCode.TEXT_GENERATOR_PD_MODEL_INSTANCE_ID_ERROR)

    def test_pull_blocks_block_id_out_of_range(self):
        """测试 pull_blocks，当 dst_block_table 中 block id 超出范围时返回错误码"""
        self.worker.cache_desc_map[0] = MagicMock()
        self.worker.cluster_comm_map[0] = [111]  # 模拟已建立连接
        self.worker.max_block_nums_map[0] = 10
        self.worker.build(model_id=0, num_tensors=1, num_blocks=10, blockshape=(2, 2), dtype="float16")
        ret = self.worker.pull_blocks(remote_model_instance_id=0, src_block_table=[0, 1], dst_block_table=[0, 11])
        self.assertEqual(ret, ErrorCode.TEXT_GENERATOR_PD_BLOCK_ID_OUT_OF_RANGE)

    def test_pull_blocks_success(self):
        """测试 pull_blocks 正常调用情况"""
        self.worker.cache_desc_map[0] = MagicMock()
        self.worker.cluster_comm_map[0] = [111]
        self.worker.max_block_nums_map[0] = 10
        self.worker.build(model_id=0, num_tensors=1, num_blocks=10, blockshape=(2, 2), dtype="float16")
        self.mock_llm_data_dist_instance.pull_kv.return_value = MindieLlmStatusCode.SUCCESS
        ret = self.worker.pull_blocks(remote_model_instance_id=0, src_block_table=[0, 1], dst_block_table=[0, 1])
        self.assertEqual(ret, MindieLlmStatusCode.SUCCESS)

    def test_pull_blocks_pull_kv_fail(self):
        """测试 pull_blocks pull_kv失败的情况"""
        self.worker.cache_desc_map[0] = MagicMock()
        self.worker.cluster_comm_map[0] = [111]
        self.worker.max_block_nums_map[0] = 10
        self.worker.build(model_id=0, num_tensors=1, num_blocks=10, blockshape=(2, 2), dtype="float16")
        self.mock_llm_data_dist_instance.pull_kv.return_value = 1
        ret = self.worker.pull_blocks(remote_model_instance_id=0, src_block_table=[0, 1], dst_block_table=[0, 1])
        self.assertEqual(ret, 1)

    def test_link_success(self):
        """测试 link 方法，当 query_register_mem_status 返回 OK 时应成功建立连接"""
        # 模拟 link 返回成功
        self.mock_llm_data_dist_instance.link.return_value = {
            "status": MindieLlmStatusCode.SUCCESS,
            "comm_id": 200
        }
        self.mock_llm_data_dist_instance.query_register_mem_status.return_value = RegisterMemStatus.OK

        # 构造 link 参数
        link_params = {
            'remote_cluster_ids': {0: [1]},
            'remote_physical_device_ids': {0: [2]},
            'remote_device_ips': {0: ['192.168.1.0']},
            'host_ips': {0: ['192.168.1.100']},
            'remote_super_device_ids': {0: [8650754]},
            'remote_super_pod_ids': {0: [0]}
        }

        ret = self.worker.link(**link_params)
        self.assertEqual(ret, [])
        # 检查 cluster_comm_map 中已记录该连接信息
        self.assertIn(1, self.worker.cluster_comm_map)
        self.assertEqual(self.worker.cluster_comm_map[1], 200)

    def test_link_success_with_different_host_ip(self):
        """测试 link 方法，当远程主机 IP 与本地不同时的连接情况"""
        self.mock_llm_data_dist_instance.link.return_value = {
            "status": MindieLlmStatusCode.SUCCESS,
            "comm_id": 200
        }
        self.mock_llm_data_dist_instance.query_register_mem_status.return_value = RegisterMemStatus.OK

        # 构造 link 参数，使用不同的 host_ip
        link_params = {
            'remote_cluster_ids': {0: [1]},
            'remote_physical_device_ids': {0: [2]},
            'remote_device_ips': {0: ['192.168.1.2']},
            'host_ips': {0: ['127.0.0.2']},  # 远程主机 IP 与本地的不同
            'remote_super_device_ids': {0: [8650754]},
            'remote_super_pod_ids': {0: [0]}
        }
        
        ret = self.worker.link(**link_params)
        # 验证返回值应为 SUCCESS
        self.assertEqual(ret, [])
        # 检查连接信息是否正确记录
        self.assertIn(1, self.worker.cluster_comm_map)
        self.assertEqual(self.worker.cluster_comm_map[1], 200)

    def test_link_summary_multiple_success(self):
        """测试多条链路全部成功时 [Link_Summary] 日志输出（INFO级）"""
        # 1. Mock 所有链路建立成功（按调用顺序返回不同 comm_id，模拟多条链路）
        link_responses = [
            {"status": MindieLlmStatusCode.SUCCESS, "comm_id": 200},  # 第1条链路（192.168.1.2）成功
            {"status": MindieLlmStatusCode.SUCCESS, "comm_id": 201},  # 第2条链路（192.168.1.3）成功
            {"status": MindieLlmStatusCode.SUCCESS, "comm_id": 202}   # 第3条链路（192.168.1.4）成功
        ]
        self.mock_llm_data_dist_instance.link.side_effect = link_responses
        self.mock_llm_data_dist_instance.query_register_mem_status.return_value = RegisterMemStatus.OK

        # 2. 构造3条链路参数（模拟多条成功链路场景）
        link_params = {
            'remote_cluster_ids': {0: [1, 2, 3]},
            'remote_physical_device_ids': {0: [2, 3, 4]},
            'remote_device_ips': {0: ['192.168.1.2', '192.168.1.3', '192.168.1.4']},
            'host_ips': {0: ['192.168.1.100', '192.168.1.101', '192.168.1.102']},
            'remote_super_device_ids': {0: [8650754, 8650755, 8650756]},
            'remote_super_pod_ids': {0: [0, 0, 0]}
        }

        # 3. 执行 link 方法
        ret = self.worker.link(**link_params)

        # 4. 核心断言：失败链路为空（所有链路成功）
        self.assertEqual(ret, [])
        
        # 5. 断言 cluster_comm_map 中记录了所有成功链路的 comm_id（验证链路真的建立成功）
        self.assertEqual(self.worker.cluster_comm_map.get(1), 200)  # 第1条链路 comm_id
        self.assertEqual(self.worker.cluster_comm_map.get(2), 201)  # 第2条链路 comm_id
        self.assertEqual(self.worker.cluster_comm_map.get(3), 202)  # 第3条链路 comm_id
        

    def test_link_failed(self):
        """测试 link 方法失败的情况"""
        # 模拟 link 返回失败
        self.mock_llm_data_dist_instance.link.return_value = {
            "status": ErrorCode.TEXT_GENERATOR_PD_LINK_ERROR,
            "comm_id": None
        }

        link_params = {
            'remote_cluster_ids': {0: [1]},
            'remote_physical_device_ids': {0: [2]},
            'remote_device_ips': {0: ['192.168.1.2']},
            'host_ips': {0: ['192.168.1.100']},
            'remote_super_device_ids': {0: [8650754]},
            'remote_super_pod_ids': {0: [0]}
        }

        ret = self.worker.link(**link_params)
        # 验证返回的是失败链路列表
        self.assertIsInstance(ret, list)
        self.assertEqual(len(ret), 1)
        self.assertEqual(ret[0][0], '192.168.1.2')
        self.assertEqual(ret[0][1], ErrorCode.TEXT_GENERATOR_PD_LINK_ERROR)
        # 验证连接信息未被记录
        self.assertNotIn(1, self.worker.cluster_comm_map)

    def test_link_timeout(self):
        """测试 link 方法因查询注册内存状态超时而失败的情况"""
        # 1. 模拟底层 link 调用成功，返回一个 comm_id
        self.mock_llm_data_dist_instance.link.return_value = {
            "status": MindieLlmStatusCode.SUCCESS,
            "comm_id": 200  # 一个有效的 comm_id
        }

        # 2. 模拟 query_register_mem_status 持续返回 QUERYING 状态，
        #    这将导致 worker.link 内部的轮询逻辑因超时而失败。
        self.mock_llm_data_dist_instance.query_register_mem_status.return_value = None
        self.worker.link_time_out = 0.1

        link_params = {
            'remote_cluster_ids': {0: [1, 2]},
            'remote_physical_device_ids': {0: [2, 3]},
            'remote_device_ips': {0: ['192.168.1.2', '192.168.1.3']},
            'host_ips': {0: ['192.168.1.100', '192.168.1.101']},
            'remote_super_device_ids': {0: [8650754, 8650755]},
            'remote_super_pod_ids': {0: [0, 0]}
        }

        ret = self.worker.link(**link_params)
        
        # 验证返回的是失败链路列表
        self.assertIsInstance(ret, list)
        self.assertEqual(len(ret), 2)
        self.assertEqual(ret[0][0], '192.168.1.2') # 发生超时的IP
        self.assertEqual(ret[0][1], ErrorCode.TEXT_GENERATOR_PD_LINK_OUT_OF_TIME) # 期望的超时错误码
        self.assertEqual(ret[1][0], '192.168.1.3') # 发生超时的IP
        self.assertEqual(ret[1][1], ErrorCode.TEXT_GENERATOR_PD_LINK_OUT_OF_TIME) # 期望的超时错误码
        
        # 验证连接信息未被记录在 cluster_comm_map 中
        self.assertNotIn(1, self.worker.cluster_comm_map)
        
        # 验证 query_register_mem_status 方法被调用过 (可能多次)
        self.mock_llm_data_dist_instance.query_register_mem_status.assert_called()

    def test_link_summary_all_success(self):
        """测试所有链路成功时 [Link_Summary] 日志输出（INFO级）"""
        # 1. Mock 链路建立成功
        self.mock_llm_data_dist_instance.link.return_value = {
            "status": MindieLlmStatusCode.SUCCESS,
            "comm_id": 200
        }
        self.mock_llm_data_dist_instance.query_register_mem_status.return_value = RegisterMemStatus.OK

        # 2. 构造链路参数（1条链路）
        link_params = {
            'remote_cluster_ids': {0: [1]},
            'remote_physical_device_ids': {0: [2]},
            'remote_device_ips': {0: ['192.168.1.2']},
            'host_ips': {0: ['192.168.1.100']},
            'remote_super_device_ids': {0: [8650754]},
            'remote_super_pod_ids': {0: [0]}
        }

        # 3. 执行 link 方法
        ret = self.worker.link(**link_params)

        # 4. 断言结果
        self.assertEqual(ret, [])  # 失败链路为空

    def test_link_summary_partial_failed(self):
        """测试部分链路失败时 [Link_Summary] 日志输出（ERROR级）"""
        # 1. Mock 部分链路失败（第一条失败，第二条成功）
        link_responses = [
            {"status": ErrorCode.TEXT_GENERATOR_PD_LINK_ERROR, "comm_id": None},  # 第一条链路（192.168.1.2）失败
            {"status": MindieLlmStatusCode.SUCCESS, "comm_id": 201}               # 第二条链路（192.168.1.3）成功
        ]

        self.mock_llm_data_dist_instance.link.side_effect = link_responses
        self.mock_llm_data_dist_instance.query_register_mem_status.return_value = RegisterMemStatus.OK

        # 2. 构造2条链路参数
        link_params = {
            'remote_cluster_ids': {0: [1, 2]},
            'remote_physical_device_ids': {0: [2, 3]},
            'remote_device_ips': {0: ['192.168.1.2', '192.168.1.3']},
            'host_ips': {0: ['192.168.1.100', '192.168.1.101']},
            'remote_super_device_ids': {0: [8650754, 8650755]},
            'remote_super_pod_ids': {0: [0, 0]}
        }

        # 3. 执行 link 方法
        ret = self.worker.link(**link_params)

        # 4. 断言结果
        self.assertEqual(len(ret), 1)  # 1条失败链路
        self.assertEqual(ret[0][0], '192.168.1.2')
        self.assertEqual(ret[0][1], ErrorCode.TEXT_GENERATOR_PD_LINK_ERROR)

    def test_link_summary_global_timeout(self):
        """测试全局超时导致失败时 [Link_Summary] 日志输出"""
        # 1. Mock 链路建立成功，但查询状态时触发全局超时
        self.mock_llm_data_dist_instance.link.return_value = {
            "status": MindieLlmStatusCode.SUCCESS,
            "comm_id": 200
        }
        self.mock_llm_data_dist_instance.query_register_mem_status.return_value = None  # 模拟查询中
        self.worker.link_time_out = 0.1  # 缩短超时时间
        self.worker.link_start_time = time.time() - 0.2  # 篡改开始时间，触发超时

        # 2. 构造链路参数
        link_params = {
            'remote_cluster_ids': {0: [1]},
            'remote_physical_device_ids': {0: [2]},
            'remote_device_ips': {0: ['192.168.1.2']},
            'host_ips': {0: ['192.168.1.100']},
            'remote_super_device_ids': {0: [8650754]},
            'remote_super_pod_ids': {0: [0]}
        }

        # 3. 执行 link 方法
        ret = self.worker.link(**link_params)

        # 4. 断言结果
        self.assertEqual(len(ret), 1)
        self.assertEqual(ret[0][1], ErrorCode.TEXT_GENERATOR_PD_LINK_OUT_OF_TIME)

    def test_link_summary_multiple_failed(self):
        """测试多链路失败时 [Link_Summary] 日志逐条输出"""
        # 1. Mock 所有链路失败
        self.mock_llm_data_dist_instance.link.return_value = {
            "status": ErrorCode.TEXT_GENERATOR_PD_LINK_ERROR,
            "comm_id": None
        }

        # 2. 构造3条链路参数
        link_params = {
            'remote_cluster_ids': {0: [1, 2, 3]},
            'remote_physical_device_ids': {0: [2, 3, 4]},
            'remote_device_ips': {0: ['192.168.1.10', '192.168.1.11', '192.168.1.12']},
            'host_ips': {0: ['192.168.1.100', '192.168.1.101', '192.168.1.102']},
            'remote_super_device_ids': {0: [8650754, 8650755, 8650756]},
            'remote_super_pod_ids': {0: [0, 0, 0]}
        }

        # 3. 执行 link 方法
        ret = self.worker.link(**link_params)

        # 4. 断言结果
        self.assertEqual(len(ret), 3)




    def test_link_summary_logs_multiple_failures(self):
        """测试多个链路失败时的 [Link_Summary] 日志输出"""
        # 模拟多个链路失败的情况
        self.mock_llm_data_dist_instance.link.return_value = {
            "status": ErrorCode.TEXT_GENERATOR_PD_LINK_ERROR,
            "comm_id": None
        }

        # 构造多个远程设备参数
        link_params = {
            'remote_cluster_ids': {0: [1, 2, 3]},
            'remote_physical_device_ids': {0: [2, 3, 4]},
            'remote_device_ips': {0: ['192.168.1.10', '192.168.1.11', '192.168.1.12']},
            'host_ips': {0: ['192.168.1.100', '192.168.1.101', '192.168.1.102']},
            'remote_super_device_ids': {0: [8650754, 8650755, 8650756]},
            'remote_super_pod_ids': {0: [0, 0, 0]}
        }

        ret = self.worker.link(**link_params)
        
        # 验证返回的是失败链路列表，包含3个失败的链路
        self.assertIsInstance(ret, list)
        self.assertEqual(len(ret), 3)
        
        # 验证每个失败的IP都正确记录
        failed_ips = [link[0] for link in ret]
        expected_ips = ['192.168.1.10', '192.168.1.11', '192.168.1.12']
        for ip in expected_ips:
            self.assertIn(ip, failed_ips)
        

    def test_unlink(self):
        """测试 unlink 方法，确保调用了引擎的 unlink 方法，并清除 cluster_comm_map 中对应记录"""
        self.worker.cluster_comm_map[1] = [400, 401]
        ret = self.worker.unlink(remote_cluster_id=1)
        self.assertEqual(ret, MindieLlmStatusCode.SUCCESS)
        expected_calls = [call([400, 401])]
        self.mock_llm_data_dist_instance.unlink.assert_has_calls(expected_calls, any_order=True)
        self.assertNotIn(1, self.worker.cluster_comm_map)

    def test_unlink_nonexistent(self):
        """测试 unlink 方法，当指定的 remote_cluster_id 不存在时返回错误码"""
        ret = self.worker.unlink(remote_cluster_id=999)
        self.assertEqual(ret, ErrorCode.TEXT_GENERATOR_PD_UNLINK_ERROR)

    def test_unlink_all_exception(self):
        """测试 unlink_all 方法, 当unlink失败返回错误码时抛出异常"""
        with self.assertRaises(Exception) as context:
            self.worker.cluster_comm_map[1] = [500]
            self.mock_llm_data_dist_instance.unlink.return_value = 1
            self.worker.unlink_all()
        self.assertIn("SeparateDeploymentEngine: unlink_all fail", str(context.exception))

    def test_finalize(self):
        """测试 finalize 方法，确保先断开所有连接，再调用 engine.finalize 进行资源清理"""
        self.worker.cluster_comm_map[1] = [500]
        self.worker.finalize()
        self.mock_llm_data_dist_instance.finalize.assert_called_once()
        self.assertEqual(self.worker.cluster_comm_map, {})


class TestSeparateDeploymentEngine(unittest.TestCase):

    @patch('mindie_llm.text_generator.utils.separate_deployment_engine.LLMDataDist')
    @patch('mindie_llm.text_generator.utils.separate_deployment_engine.LLMDataDistConfig')
    def setUp(self, mock_llm_data_dist_config, mock_llm_data_dist):

        self.test_name = self._testMethodName  # 获取当前用例名称
        self.class_name = self.__class__.__name__  # 获取当前类名
        logger.info("=" * 80)
        logger.info(f"开始执行测试用例：{self.class_name}.{self.test_name}")
        logger.info("=" * 80)

        self.mock_llm_data_dist = mock_llm_data_dist
        self.mock_llm_data_dist_config = mock_llm_data_dist_config

        self.engine = SeparateDeploymentEngine(
            role='decoder',
            local_cluster_id=0,
            local_logic_device_id=0,
            kv_trans_timeout=1,
            kv_rdma_sl=-1,
            kv_rdma_tc=-1
        )
    
    @patch('mindie_llm.text_generator.utils.separate_deployment_engine.LLMDataDist')
    @patch('mindie_llm.text_generator.utils.separate_deployment_engine.LLMDataDistConfig')
    def test_init_prefill(self, mock_llm_data_dist_config, mock_llm_data_dist):
        engine = SeparateDeploymentEngine(
            role='prefill',
            local_cluster_id=0,
            local_logic_device_id=0,
            kv_trans_timeout=0,
            kv_rdma_sl=-1,
            kv_rdma_tc=-1
        )

        self.assertEqual(engine.role, 'prefill')
    
    @patch('mindie_llm.text_generator.utils.separate_deployment_engine.LLMDataDist')
    @patch('mindie_llm.text_generator.utils.separate_deployment_engine.LLMDataDistConfig')
    def test_init_decoder(self, mock_llm_data_dist_config, mock_llm_data_dist):
        engine = SeparateDeploymentEngine(
            role='decoder',
            local_cluster_id=0,
            local_logic_device_id=0,
            kv_trans_timeout=1,
            kv_rdma_sl=7,
            kv_rdma_tc=255
        )

        self.assertEqual(engine.role, 'decoder')
    
    def test_init_invalid_role(self):
        with self.assertRaises(Exception) as context:
            _ = SeparateDeploymentEngine(
                role='invalid',
                local_cluster_id=0,
                local_logic_device_id=0,
                kv_trans_timeout=1,
                kv_rdma_sl=-1,
                kv_rdma_tc=-1
            )


    def test_init_invalid_kv_rdma_sl(self):
        with self.assertRaises(Exception) as context:
            _ = SeparateDeploymentEngine(
                role='prefill',
                local_cluster_id=0,
                local_logic_device_id=0,
                kv_trans_timeout=1,
                kv_rdma_sl=8,
                kv_rdma_tc=255
            )

        self.assertIn("SeparateDeploymentEngine: kv_rdma_sl only support: 0-7.", str(context.exception))
    
    def test_init_invalid_kv_rdma_tc(self):
        with self.assertRaises(Exception) as context:
            _ = SeparateDeploymentEngine(
                role='decoder',
                local_cluster_id=0,
                local_logic_device_id=0,
                kv_trans_timeout=1,
                kv_rdma_sl=7,
                kv_rdma_tc=256
            )

        self.assertIn("SeparateDeploymentEngine: kv_rdma_tc only support: 0-255.", str(context.exception))
    
    def test_link(self):
        cluster_rank_info = {0: 0}
        rank_table = \
            '{"server_count": "1", ' \
            '"server_list": [{"device": [{"device_ip": "192.168.1.1"}, {"device_ip": "192.168.1.2"}]}]}'
        self.engine.separate_deployment_engine.link.return_value = 12345
        result = self.engine.link(cluster_rank_info, rank_table)
        self.assertEqual(result, {'status': MindieLlmStatusCode.SUCCESS, 'comm_id': 12345})
    
    def test_link_server_count_2(self):
        cluster_rank_info = {0: 0}
        rank_table = \
            '{"server_count": "2", ' \
            '"server_list": [{"device": [{"device_ip": "192.168.1.2"}]}, {"device": [{"device_ip": "192.168.1.1"}]}]}'
        self.engine.separate_deployment_engine.link.return_value = 12345
        result = self.engine.link(cluster_rank_info, rank_table)
        self.assertEqual(result, {'status': MindieLlmStatusCode.SUCCESS, 'comm_id': 12345})
    
    def test_link_already_linked(self):
        cluster_rank_info = {0: 0}
        rank_table = \
            '{"server_count": "1", ' \
            '"server_list": [{"device": [{"device_ip": "192.168.1.1"}, {"device_ip": "192.168.1.2"}]}]}'
        self.engine.separate_deployment_engine.link.return_value = LLMStatusCode.LLM_ALREADY_LINK
        result = self.engine.link(cluster_rank_info, rank_table)
        self.assertEqual(result, {'status': MindieLlmStatusCode.TEXT_GENERATOR_PD_ALREADY_LINK, 'comm_id': None})
    
    def test_link_exception(self):
        cluster_rank_info = {0: 0}
        rank_table = \
            '{"server_count": "1", ' \
            '"server_list": [{"device": [{"device_ip": "192.168.1.1"}, {"device_ip": "192.168.1.2"}]}]}'
        self.engine.separate_deployment_engine.link.side_effect = LLMException("Link failed")
        result = self.engine.link(cluster_rank_info, rank_table)
        self.assertEqual(result, {'status': ErrorCode.TEXT_GENERATOR_PD_LINK_ERROR, 'comm_id': None})
    
    def test_unlink(self):
        self.engine.separate_deployment_engine.unlink.return_value = None
        result = self.engine.unlink(12345)
        self.assertEqual(result, MindieLlmStatusCode.SUCCESS)
    
    def test_unlink_exception(self):
        self.engine.separate_deployment_engine.unlink.side_effect = LLMException("Unlink failed")
        result = self.engine.unlink(12345)
        self.assertEqual(result, ErrorCode.TEXT_GENERATOR_PD_UNLINK_ERROR)
    
    def test_set_npu_cache(self):
        model_id = 1
        npu_cache = [1, 2, 3]
        self.engine.set_npu_cache(model_id, npu_cache)
        self.assertEqual(self.engine.npu_cache_map[model_id], npu_cache)
    
    def test_pull_kv(self):
        model_id = 1
        src_block_table = [1, 2, 3]
        dst_block_table = [4, 5, 6]
        remote_cluster_id = 0
        self.engine.npu_cache_map[model_id] = [1, 2, 3]
        self.engine.separate_deployment_engine.cache_manager.pull_blocks.return_value = None
        result = self.engine.pull_kv(model_id, src_block_table, dst_block_table, remote_cluster_id)
        self.assertEqual(result, MindieLlmStatusCode.SUCCESS)
    
    def test_pull_kv_exception(self):
        model_id = 1
        src_block_table = [1, 2, 3]
        dst_block_table = [4, 5, 6]
        remote_cluster_id = 0
        self.engine.npu_cache_map[model_id] = [1, 2, 3]
        self.engine.separate_deployment_engine.cache_manager.pull_blocks.side_effect = LLMException("Pull kv failed")
        result = self.engine.pull_kv(model_id, src_block_table, dst_block_table, remote_cluster_id)
        self.assertEqual(result, ErrorCode.TEXT_GENERATOR_PD_PULL_KV_ERROR)
    
    def test_register_blocks_cache(self):
        cache_desc = 'cache_desc'
        npu_addrs = [1, 2, 3]
        cache_key = 'cache_key'
        self.engine.separate_deployment_engine.cache_manager.register_blocks_cache.return_value = None
        result = self.engine.register_blocks_cache(cache_desc, npu_addrs, cache_key)
        self.assertIsNone(result)
    
    def test_query_register_mem_status(self):
        comm_id = 12345
        self.engine.separate_deployment_engine.query_register_mem_status.return_value = True
        result = self.engine.query_register_mem_status(comm_id)
        self.assertTrue(result)
    
    def test_finalize(self):
        self.engine.separate_deployment_engine.finalize.return_value = None
        self.engine.finalize()


if __name__ == "__main__":
    unittest.main()