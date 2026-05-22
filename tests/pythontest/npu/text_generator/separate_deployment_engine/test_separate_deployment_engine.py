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
import multiprocessing
from functools import reduce
import operator
import torch
import torch_npu
from mindie_llm.utils.log.logging import logger
from mindie_llm.text_generator.utils.separate_deployment_engine import SeparateDeploymentWorker
from mindie_llm.utils.status import MindieLlmStatusCode

# 角色定义
DECODER1 = 'decoder1'
DECODER2 = 'decoder2'
PREFILL1 = 'prefill1'
PREFILL2 = 'prefill2'
ROLE = 'role'
STATUS = 'status'
BUILD_DONE = 'build_done'
ERROR = 'error'
MESSAGE = 'message'
PULL_DOWN = 'pull_done'
LINK_SUCCESS = 'link_success'
DTYPE_FLOAT16 = "float16"
DTYPE_BFLOAT16 = "bfloat16"
DTYPE_FLOAT32 = "float"
DTYPE_INT8 = 'int8'

# 卡的配置
DEVICES_CONFIG = {
    DECODER1: {
        'local_logic_device_id': 0,
        'local_physical_device_id': 0,
        'local_cluster_id': 0,
        'local_device_ip': '124.0.2.41',
    },
    DECODER2: {
        'local_logic_device_id': 1,
        'local_physical_device_id': 1,
        'local_cluster_id': 1,
        'local_device_ip': '124.0.2.42',
    },
    PREFILL1: {
        'local_logic_device_id': 2,
        'local_physical_device_id': 2,
        'local_cluster_id': 2,
        'local_device_ip': '124.0.2.43',
    },
    PREFILL2: {
        'local_logic_device_id': 3,
        'local_physical_device_id': 3,
        'local_cluster_id': 3,
        'local_device_ip': '124.0.2.44',
    }
}

# 远程连接配置
REMOTE_CONFIG = {
    DECODER1: {
        'remote_cluster_ids': {0: [2, 3]},
        'remote_physical_device_ids': {0: [2, 3]},
        'remote_device_ips': {0: ['124.0.2.43', '124.0.2.44']},
        'host_ips': {0: ['1', '1']},
    },
    DECODER2: {
        'remote_cluster_ids': {0: [2, 3]},
        'remote_physical_device_ids': {0: [2, 3]},
        'remote_device_ips': {0: ['124.0.2.43', '124.0.2.44']},
        'host_ips': {0: ['1', '1']},
    },
    PREFILL1: {
        'remote_cluster_ids': {0: [0, 1]},
        'remote_physical_device_ids': {0: [0, 1]},
        'remote_device_ips': {0: ['124.0.2.41', '124.0.2.42']},
        'host_ips': {0: ['1', '1']},
    },
    PREFILL2: {
        'remote_cluster_ids': {0: [0, 1]},
        'remote_physical_device_ids': {0: [0, 1]},
        'remote_device_ips': {0: ['124.0.2.41', '124.0.2.42']},
        'host_ips': {0: ['1', '1']},
    }
}


def create_aligned_tensor(target_shape, dtype):
    element_size = torch.tensor([], dtype=dtype).element_size()
    desired_num_elements = reduce(operator.mul, target_shape, 1)
    tensor_size = desired_num_elements * element_size
    additional_memory = 2 * 1024 * 1024
    total_size = tensor_size + additional_memory

    kv_tensor = torch.rand(size=(total_size // element_size, ), dtype=dtype, device='npu')
    tensor_ptr = kv_tensor.data_ptr()
    alignment = 2 * 1024 * 1024
    offset = tensor_ptr % alignment
    if offset != 0:
        cut_size = alignment - offset
    else:
        cut_size = 0
    # 切片对齐 tensor
    cut_elements = cut_size // element_size
    aligned_tensor = kv_tensor[cut_elements: cut_elements + desired_num_elements]
    aligned_tensor = aligned_tensor.contiguous()
    aligned_tensor = aligned_tensor.view(*target_shape)
    return aligned_tensor


def test_separate_deployment_engine(role: str, proc_id: int, link_barrier: multiprocessing.Barrier, pull_barrier: multiprocessing.Barrier):
    try:
        # 构造配置键
        config_key = f"{role}{proc_id+1}"  # 如'decoder1', 'decoder2'
        
        # 获取设备配置
        device_config = DEVICES_CONFIG[config_key]
        remote_config = REMOTE_CONFIG[config_key]
        
        # 初始化 SeparateDeploymentWorker
        worker = SeparateDeploymentWorker(
            role=role,
            local_logic_device_id=device_config['local_logic_device_id'],
            local_physical_device_id=device_config['local_physical_device_id'],
            local_cluster_id=device_config['local_cluster_id'],
            local_device_ip=device_config['local_device_ip'],
            local_host_ip="1"
        )
        
        # 建立和配置缓存
        num_layers = 1
        num_blocks = 1198
        k_block_shape = (128, 1, 512)
        v_block_shape = (128, 1, 64)
        k_head_size = 512
        v_head_size = 64
        dtype = torch.float16
        npu_cache = []
        k_blocks_addrs = []
        v_blocks_addrs = []
        
        # 使用API进行build操作
        worker.build(model_id=0, num_tensors=num_layers, num_blocks=num_layers,
                    blockshape=k_block_shape, dtype='float16')
        worker.build(model_id=1, num_tensors=num_layers, num_blocks=num_layers,
                    blockshape=v_block_shape, dtype='float16')
        
        # 创建和设置缓存
        for _ in range(num_layers):
            key_blocks = create_aligned_tensor((num_blocks, *k_block_shape), dtype)
            k_blocks_addrs.append(key_blocks.data_ptr())
            
            value_blocks = create_aligned_tensor((num_blocks, *v_block_shape), dtype)
            v_blocks_addrs.append(value_blocks.data_ptr())
            
            npu_cache.append((key_blocks, value_blocks))
        
        logger.info(f"{config_key}: k_head_size{k_head_size}, v_head_size{v_head_size}, k_block_shape{k_block_shape}, v_block_shape{v_block_shape}")
        worker.set_npu_cache(model_id=0, npu_addrs=k_blocks_addrs)
        worker.set_npu_cache(model_id=1, npu_addrs=v_blocks_addrs)
        
        # 2. 建立连接 - 批量建链
        logger.info(f"{config_key}: Starting link operation.")
        
        # 调用批量建链接口
        link_result = worker.link(
            remote_cluster_ids=remote_config['remote_cluster_ids'],
            remote_physical_device_ids=remote_config['remote_physical_device_ids'],
            remote_device_ips=remote_config['remote_device_ips'],
            host_ips=remote_config['host_ips'],
        )

        # 检查建链结果
        if link_result:  # 返回的失败链路列表不为空
            logger.error(f"{config_key}: Link operation failed with error: {link_result}")
            raise Exception(f"{config_key} link failed with error {link_result}")
        
        remote_clusters = remote_config['remote_cluster_ids'][0]
        logger.info(f"==> {config_key} linked successfully to remote clusters {remote_clusters}")
        
        # 使用 Barrier 等待所有进程完成链接
        logger.info(f"{config_key}: Waiting for all processes to complete link.")
        try:
            link_barrier.wait(timeout=60)  # 60秒超时
            logger.info(f"{config_key}: All processes have completed linking.")
        except multiprocessing.TimeoutError:
            logger.error(f"{config_key}: Timeout waiting for other processes to link")
            raise Exception(f"{config_key}: Timeout waiting for other processes to link") from None
        
        # 执行 pull 操作 (只有 decoder 角色执行)
        if role == "decoder":
            # 远程 prefill 的 cluster_id
            prefill_cluster_ids = [cluster_id for cluster_id in remote_clusters 
                                  if cluster_id in [DEVICES_CONFIG['prefill1']['local_cluster_id'], 
                                                   DEVICES_CONFIG['prefill2']['local_cluster_id']]]
            
            logger.info(f"{config_key}: Starting pull_blocks operation from prefill clusters: {prefill_cluster_ids}")
            
            for remote_cluster_id in prefill_cluster_ids:
                src_block_table = [0]
                dst_block_table = [0]
                
                pull_result = worker.pull_blocks(
                    remote_model_instance_id=remote_cluster_id,
                    src_block_table=src_block_table,
                    dst_block_table=dst_block_table
                )
                
                if pull_result != MindieLlmStatusCode.SUCCESS:
                    logger.error(f"{config_key}: pull_blocks from cluster {remote_cluster_id}"
                                 f"failed with error code: {pull_result}")
                    raise Exception(f"{config_key}: pull_blocks from cluster {remote_cluster_id}"
                                     f"failed with error code: {pull_result}")
                
                logger.info(f"==> {config_key} pull_blocks success from cluster {remote_cluster_id}")
            
            # 通知 prefill 进程 pull 完成
            try:
                pull_barrier.wait(timeout=60)  # 60秒超时
                logger.info(f"{config_key}: Pull operation completed and prefill processes notified.")
            except multiprocessing.TimeoutError:
                logger.error(f"{config_key}: Timeout waiting for prefill processes")
                raise Exception(f"{config_key}: Timeout waiting for prefill processes") from None
        
        # 等待 pull 完成信号
        if role == "prefill":
            logger.info(f"{config_key}: Waiting for pull operation to complete.")
            try:
                pull_barrier.wait(timeout=60)  # 60秒超时
                logger.info(f"{config_key}: Pull operation completed.")
            except multiprocessing.TimeoutError:
                logger.error(f"{config_key}: Timeout waiting for pull operation")
                raise Exception(f"{config_key}: Timeout waiting for pull operation") from None
        
        # 4. 断开连接
        logger.info(f"{config_key}: Unlinking from all remote clusters.")
        for remote_cluster_id in remote_clusters:
            unlink_result = worker.unlink(remote_cluster_id)
            if unlink_result != MindieLlmStatusCode.SUCCESS:
                logger.error(f"{config_key}: Unlink from cluster {remote_cluster_id} failed with error code: {unlink_result}")
                raise Exception(f"{config_key}: Unlink from cluster {remote_cluster_id} failed with error code: {unlink_result}")
            else:
                logger.info(f"==> {config_key} unlink success from cluster {remote_cluster_id}")
        
    except Exception as e:
        logger.exception(f"Exception in {config_key} process: {e}")
    finally:
        # 5. 清理工作
        logger.info(f"{config_key}: Finalizing and cleaning up.")
        worker.finalize()


if __name__ == '__main__':
    # 创建两个 Barrier，一个用于链接同步，一个用于 pull 操作同步
    link_barrier = multiprocessing.Barrier(4)  # 4个进程都需要等待链接完成
    pull_barrier = multiprocessing.Barrier(4)  # 4个进程都需要等待 pull 完成

    # 启动4个进程: 2个decoder和2个prefill
    p_decoder1 = multiprocessing.Process(target=test_separate_deployment_engine, args=('decoder', 0, link_barrier, pull_barrier))
    p_decoder2 = multiprocessing.Process(target=test_separate_deployment_engine, args=('decoder', 1, link_barrier, pull_barrier))
    p_prefill1 = multiprocessing.Process(target=test_separate_deployment_engine, args=('prefill', 0, link_barrier, pull_barrier))
    p_prefill2 = multiprocessing.Process(target=test_separate_deployment_engine, args=('prefill', 1, link_barrier, pull_barrier))
    
    # 顺序启动进程，每个进程之间稍微延迟以避免冲突
    p_decoder1.start()
    time.sleep(1)
    p_decoder2.start()
    time.sleep(1)
    p_prefill1.start()
    time.sleep(1)
    p_prefill2.start()
    
    # 等待所有进程完成
    p_decoder1.join()
    p_decoder2.join()
    p_prefill1.join()
    p_prefill2.join()
    
    logger.info("所有进程已完成，测试结束。")