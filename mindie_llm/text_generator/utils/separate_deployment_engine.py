# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import json
import time

from typing import List, Dict
from dataclasses import dataclass
from enum import Enum
from llm_datadist import LLMRole, BlocksCacheKey, LLMStatusCode
from llm_datadist import CacheDesc, DataType, LLMException, Placement
from llm_datadist import LLMDataDist, RegisterMemStatus
from llm_datadist import LLMConfig as LLMDataDistConfig
from mindie_llm.utils.log.logging import logger
from mindie_llm.utils.log.error_code import ErrorCode
from mindie_llm.utils.status import MindieLlmStatusCode
from mindie_llm.model_wrapper.utils.common_util import get_ip_obj


STR_DTYPE_TO_DTYPE = {
    "float16": DataType.DT_FLOAT16,
    "bfloat16": DataType.DT_BF16,
    "float": DataType.DT_FLOAT,
    "int8": DataType.DT_INT8,
}


class DmiModeNodeRole(str, Enum):
    '''
    PD分离特性中的节点角色管理机制。
    在PD分离架构中,不同类型的计算节点被赋予特定角色以优化处理效率:
    - Flex(弹性)节点:具备动态任务处理能力,可根据系统负载同时处理Prefill和Decode请求
    - 角色细分：
    * FlexP   - 专用于PD分离Prefill阶段请求处理
    * FlexD   - 专用于PD分离Decode阶段请求处理  
    * FlexPnD - 支持Prefill和Decode在同一个节点上请求处理
    '''
    """节点工作模式角色枚举"""
    PREFILL = 'prefill'    # 专用于请求预处理阶段（Prefill）的节点角色
    DECODER = 'decoder'      # 专用于序列解码阶段（Decode）的节点角色
    FLEX = 'flex'          # 弹性节点角色，可动态处理Prefill和Decode混合请求


@dataclass
class LinkParams:
    remote_cluster_ids: Dict[int, List[int]]
    remote_physical_device_ids: Dict[int, List[int]]
    remote_device_ips: Dict[int, List[str]]
    host_ips: Dict[int, List[str]]
    remote_super_device_ids: Dict[int, List[int]] | None = None
    remote_super_pod_ids: Dict[int, List[int]] | None = None


class RankInfo:
    def __init__(self, remote_cluster_id, remote_physical_device_id, 
                 remote_device_ip, remote_host_ip, local_host_ip, 
                 local_cluster_id, local_device_ip, local_physical_device_id,
                 local_super_device_id=None, remote_super_device_id=None,
                 local_super_pod_id=None, remote_super_pod_id=None):  # 保留链接索引参数
        self.local_cluster_id = local_cluster_id
        self.remote_cluster_id = remote_cluster_id
        self.remote_physical_device_id = str(remote_physical_device_id)
        self.remote_device_ip = remote_device_ip
        self.remote_server_id = remote_host_ip
        self.local_server_id = local_host_ip
        self.local_device_ip = local_device_ip
        self.local_physical_device_id = str(local_physical_device_id)
        rank_ids = self.assign_rank_id(remote_device_ip, local_device_ip)
        self.local_rank_id = rank_ids[local_device_ip]
        self.remote_rank_id = rank_ids[remote_device_ip]
        self.local_super_device_id = local_super_device_id
        self.remote_super_device_id = remote_super_device_id
        self.local_super_pod_id = local_super_pod_id
        self.remote_super_pod_id = remote_super_pod_id
    
    @staticmethod
    def assign_rank_id(remote_device_ip: str, local_device_ip: str) -> Dict[str, int]:
        # 比较 ip 大小分配 rank 地址，小卡用"0"，大卡用"1"
        remote_device_ip_obj = get_ip_obj(remote_device_ip, "remote_device_ip")
        local_device_ip_obj = get_ip_obj(local_device_ip, "local_device_ip")

        if remote_device_ip_obj < local_device_ip_obj:
            return {remote_device_ip: 0, local_device_ip: 1}
        else:
            return {remote_device_ip: 1, local_device_ip: 0}
        
    @staticmethod
    def create_device(device_id, device_ip, rank_id, super_device_id=None) -> Dict:
        device_info = {
            "device_id": device_id,
            "device_ip": device_ip,
            "rank_id": str(rank_id)
        }
        if super_device_id is not None:
            device_info["super_device_id"] = str(super_device_id)
        return device_info

    def create_super_pod_list(self): 
        super_pod_list = []
        server_id_key = "server_id"
        server_list_key = "server_list"
        super_pod_id_key = "super_pod_id"
        server_list = [{server_id_key: str(self.local_server_id)}]
        if self.local_super_pod_id == self.remote_super_pod_id:
            if self.local_server_id != self.remote_server_id:
                server_list.append({server_id_key: str(self.remote_server_id)})
                
            super_pod_list = [
                {
                    super_pod_id_key: str(self.local_super_pod_id),
                    server_list_key: server_list
                }
            ]
        else:
            super_pod_list = [
                {
                    super_pod_id_key: str(self.local_super_pod_id),
                    server_list_key: [{server_id_key: str(self.local_server_id)}]
                },
                {
                    super_pod_id_key: str(self.remote_super_pod_id),
                    server_list_key: [{server_id_key: str(self.remote_server_id)}]
                }
            ]
        return super_pod_list

    def get_rank_table(self) -> str:
        rank_table_status = "completed"
        rank_table_version = "1.0"
        device = "device"
        server_id = "server_id"
        server_list: List[Dict] = []
        
        if self.remote_server_id == self.local_server_id:
            # 当远程服务器ID等于本地服务器ID时，只需要一个服务器条目，但包含两个设备
            server = {
                device: [
                    self.create_device(self.local_physical_device_id, self.local_device_ip, 
                                       self.local_rank_id, self.local_super_device_id),
                    self.create_device(self.remote_physical_device_id, self.remote_device_ip, 
                                       self.remote_rank_id, self.remote_super_device_id)
                ],
                server_id: str(self.local_server_id)
            }
            server_list.append(server)
            server_count = "1"
        else:
            # 当远程服务器ID不等于本地服务器ID时，有两个服务器条目，每个服务器包含一个设备
            # A3限制必须对端和本端的rank table的server list一样
            if self.local_rank_id < self.remote_rank_id:
                server_list.append({
                    device: [
                        self.create_device(self.local_physical_device_id, self.local_device_ip,
                                           self.local_rank_id, self.local_super_device_id)
                    ],
                    server_id: str(self.local_server_id)
                })
                server_list.append({
                    device: [
                        self.create_device(self.remote_physical_device_id, self.remote_device_ip,
                                           self.remote_rank_id, self.remote_super_device_id)
                    ],
                    server_id: str(self.remote_server_id)
                })
            else:
                server_list.append({
                    device: [
                        self.create_device(self.remote_physical_device_id, self.remote_device_ip,
                                           self.remote_rank_id, self.remote_super_device_id)
                    ],
                    server_id: str(self.remote_server_id)
                })
                server_list.append({
                    device: [
                        self.create_device(self.local_physical_device_id, self.local_device_ip,
                                           self.local_rank_id, self.local_super_device_id)
                    ],
                    server_id: str(self.local_server_id)
                })
            server_count = "2"
        rank_table_dict = {
            "server_count": server_count,
            "server_list": server_list,
            "status": rank_table_status,
            "version": rank_table_version
        }
        if self.local_super_device_id is not None:
            rank_table_version = "1.2"
            rank_table_dict["version"] = rank_table_version
            super_pod_list = self.create_super_pod_list()
            rank_table_dict["super_pod_list"] = super_pod_list
            
        return json.dumps(rank_table_dict)
    
    def get_cluster_rank_info(self) -> dict:
        if self.local_rank_id < self.remote_rank_id:
            return {int(self.local_cluster_id): self.local_rank_id, int(self.remote_cluster_id): self.remote_rank_id}
        else:
            return {int(self.remote_cluster_id): self.remote_rank_id, int(self.local_cluster_id): self.local_rank_id}


class SeparateDeploymentEngine:
    def __init__(self, role=DmiModeNodeRole.DECODER, local_cluster_id=0, local_logic_device_id=0, kv_trans_timeout=1,
                 kv_rdma_sl=-1, kv_rdma_tc=-1, kv_link_timeout=1080):
        engine_role = None
        if role == DmiModeNodeRole.PREFILL:
            engine_role = LLMRole.PROMPT
        elif role == DmiModeNodeRole.DECODER:
            engine_role = LLMRole.DECODER
        elif role == DmiModeNodeRole.FLEX: # 底层LLMDataDist接口给出的枚举是MIX，上层定义的微调实例名称为flex，在此处进行转换
            engine_role = LLMRole.MIX

        self.role = role
        
        self.separate_deployment_engine = LLMDataDist(engine_role, local_cluster_id)

        llm_config = LLMDataDistConfig()
        llm_config.device_id = local_logic_device_id
        llm_config.enable_cache_manager = True
        llm_config.enable_remote_cache_accessible = True
        llm_config.link_total_time = int(max(120, kv_link_timeout/8))
        llm_config.link_retry_count = 10
        #配置pull kv 超时时间，默认为1秒，sync_kv_timeout单位为ms。
        if kv_trans_timeout <= 0:
            kv_trans_timeout = 1
        llm_config.sync_kv_timeout = kv_trans_timeout * 1000
        logger.info(f"kv_trans_timeout: {llm_config.sync_kv_timeout}."
                    f"link_total_time: {llm_config.link_total_time}."
                    f"link_retry_count: {llm_config.link_retry_count}.")
        llm_options = llm_config.generate_options()
        if kv_rdma_sl != -1 and kv_rdma_tc != -1:
            if kv_rdma_sl < 0 or kv_rdma_sl > 7:
                raise Exception("SeparateDeploymentEngine: kv_rdma_sl only support: 0-7.")
            if kv_rdma_tc < 0 or kv_rdma_tc > 255:
                raise Exception("SeparateDeploymentEngine: kv_rdma_tc only support: 0-255.")
            llm_options["llm.RdmaServiceLevel"] = str(kv_rdma_sl)
            llm_options["llm.RdmaTrafficClass"] = str(kv_rdma_tc)
            logger.info(f"kv_rdma_sl: {llm_options['llm.RdmaServiceLevel']}, "
                        f"kv_rdma_tc: {llm_options['llm.RdmaTrafficClass']}.")
        self.separate_deployment_engine.init(llm_options)
        self.npu_tensors = []
        self.npu_cache_map = {}

    def link(self, cluster_rank_info: Dict[int, int], rank_table: str):
        rank_table_dict = json.loads(rank_table)
        server_count = rank_table_dict.get("server_count")
        server_list = rank_table_dict.get("server_list", [])
        device = "device"
        device_ip = "device_ip"
        status = "status"
        comm_id = "comm_id"
        if server_count == "1":
            server = server_list[0]
            devices = server.get(device, [])
            device_ip1 = devices[0].get(device_ip)
            device_ip2 = devices[1].get(device_ip)
        elif server_count == "2":
            server1 = server_list[0]
            server2 = server_list[1]
            devices1 = server1.get(device, [])
            devices2 = server2.get(device, [])
            device_ip1 = devices1[0].get(device_ip)
            device_ip2 = devices2[0].get(device_ip)

        if device_ip1 > device_ip2:
            link_name = f"link{device_ip1}:{device_ip2}"
        else:
            link_name = f"link{device_ip2}:{device_ip1}"

        logger.info(f"Link params cluster_rank_info: {cluster_rank_info}, rank_table: {rank_table}, "
                    f"link_name: {link_name}.")
        
        try:
            result = self.separate_deployment_engine.link(link_name, cluster_rank_info, rank_table)
            # 检查返回值是否为已建链错误码
            if result == LLMStatusCode.LLM_ALREADY_LINK:
                return {
                    status: MindieLlmStatusCode.TEXT_GENERATOR_PD_ALREADY_LINK,
                    comm_id: None
                }
            # 正常返回通信ID
            return {
                status: MindieLlmStatusCode.SUCCESS,
                comm_id: result
            }
        except LLMException as e:
            logger.error(f"Link failed, error code is {e.status_code}, rank_table is {rank_table}, "
                         f"cluster_rank_info is {cluster_rank_info}.")
            return {
                status: ErrorCode.TEXT_GENERATOR_PD_LINK_ERROR,
                comm_id: None
            }
            
    def unlink(self, comm_id: int):
        try:
            self.separate_deployment_engine.unlink(comm_id)
            return MindieLlmStatusCode.SUCCESS
        except LLMException:
            logger.error(f"Unlink failed, error code is {ErrorCode.TEXT_GENERATOR_PD_UNLINK_ERROR}.")
            return ErrorCode.TEXT_GENERATOR_PD_UNLINK_ERROR
    
    def set_npu_cache(self, model_id, npu_cache):
        self.npu_cache_map[model_id] = npu_cache

    def pull_kv(self, model_id: int, src_block_table: List[int], dst_block_table: List[int], remote_cluster_id: int):
        remote_cache_key = BlocksCacheKey(cluster_id=remote_cluster_id, model_id=model_id)
        try:
            self.separate_deployment_engine.cache_manager.pull_blocks(remote_cache_key, self.npu_cache_map[model_id], 
                                                                      src_block_table, dst_block_table)
            return MindieLlmStatusCode.SUCCESS
        except LLMException as e:
            logger.error(f"Pull kv from remote_cluster_id: {remote_cluster_id} failed, "
                         f"CANN status_code is: {e.status_code}.")
            return ErrorCode.TEXT_GENERATOR_PD_PULL_KV_ERROR

    def register_blocks_cache(self, cache_desc, npu_addrs, cache_key):
        cache_manager = self.separate_deployment_engine.cache_manager
        return cache_manager.register_blocks_cache(cache_desc, npu_addrs, cache_key)

    def query_register_mem_status(self, comm_id: int):
        return self.separate_deployment_engine.query_register_mem_status(comm_id)

    def finalize(self):
        self.separate_deployment_engine.finalize()



class SeparateDeploymentWorker:
    def __init__(self, role: str, local_logic_device_id: int,
                 local_physical_device_id: int, 
                 local_cluster_id: int, local_device_ip: str,
                 local_host_ip: str, local_super_pod_id: int | None = None,
                 local_super_device_id: int | None = None,
                 kv_trans_timeout: int = 1,
                 kv_link_timeout: int = 1080,
                 kv_rdma_sl: int = -1,
                 kv_rdma_tc: int = -1
                 ):
        self.role = role
        self.local_logic_device_id = local_logic_device_id
        self.local_physical_device_id = local_physical_device_id
        self.local_cluster_id = local_cluster_id
        self.local_device_ip = local_device_ip
        self.local_host_ip = local_host_ip
        self.local_super_device_id = local_super_device_id
        self.local_super_pod_id = local_super_pod_id
        self.cluster_comm_map = {}
        self.cluster_device_ip_map = {}  # 新增：cluster_id到device_ip的映射
        self.link_time_out = kv_link_timeout
        if self.link_time_out <= 0:
            self.link_time_out = 1080
        logger.info(f"kv_link_timeout: {self.link_time_out}.")
        self.link_start_time = 0
        self.cache_desc_map = {}       
        self.max_block_nums_map = {}
        if role in [DmiModeNodeRole.DECODER, DmiModeNodeRole.PREFILL, DmiModeNodeRole.FLEX]:
            self.separate_deployment_engine = SeparateDeploymentEngine(role=role, 
                                                                       local_cluster_id=local_cluster_id,
                                                                       local_logic_device_id=local_logic_device_id,
                                                                       kv_trans_timeout=kv_trans_timeout,
                                                                       kv_rdma_sl=kv_rdma_sl,
                                                                       kv_rdma_tc=kv_rdma_tc,
                                                                       kv_link_timeout=kv_link_timeout
                                                                       )
        else:
            raise Exception("SeparateDeploymentEngine: not support role.")
        
    @staticmethod
    def _collect_remaining_failed_links(instance_ids, 
                                        instance_idx, 
                                        link_idx, 
                                        params, 
                                        failed_links,
                                        window=None):
        """
        收集所有未处理的链接并标记为超时失败
        
        参数:
            instance_ids: 实例ID列表
            instance_idx: 当前处理的实例索引
            link_idx: 当前处理的链接索引
            params: 链接参数
            failed_links: 失败链接列表
            window: 滑动窗口，包含尚未处理的链接
        """
        # 首先收集窗口中所有尚未处理的链接
        if window is not None:
            for item in window:
                failed_links.append([item['link_params']['remote_device_ip'], 
                                    ErrorCode.TEXT_GENERATOR_PD_LINK_OUT_OF_TIME])
        
        # 添加所有未处理的IP到失败列表
        for idx in range(instance_idx, len(instance_ids)):
            instance_id = instance_ids[idx]
            start_idx = 0 if idx != instance_idx else link_idx
            for j in range(start_idx, len(params.remote_device_ips[instance_id])):
                failed_links.append([params.remote_device_ips[instance_id][j], 
                                    ErrorCode.TEXT_GENERATOR_PD_LINK_OUT_OF_TIME])

    @staticmethod
    def _create_link_params(instance_id, index, params):
        single_link_params = {
            'remote_cluster_id': params.remote_cluster_ids[instance_id][index],
            'remote_physical_device_id': params.remote_physical_device_ids[instance_id][index],
            'remote_device_ip': params.remote_device_ips[instance_id][index],
            'remote_host_ip': params.host_ips[instance_id][index],
        }
        if params.remote_super_device_ids is not None:
            single_link_params['remote_super_device_id'] = params.remote_super_device_ids[instance_id][index]
        if params.remote_super_pod_ids is not None:
            single_link_params['remote_super_pod_id'] = params.remote_super_pod_ids[instance_id][index]
        
        return single_link_params       
        
    def build(self, model_id: int, num_tensors, num_blocks, blockshape, dtype): 
        block_shape = tuple(map(int, blockshape))
        self.cache_desc_map[model_id] = CacheDesc(num_tensors=int(num_tensors), shape=(int(num_blocks), *block_shape), 
                                    data_type=STR_DTYPE_TO_DTYPE[dtype], placement=Placement.DEVICE)
        self.max_block_nums_map[model_id] = num_blocks
    
    # K npu_cache model_id is 0, V npu_cache model_id is 1
    def set_npu_cache(self, model_id: int, npu_addrs: List[int]):
        try:
            cache_key = BlocksCacheKey(cluster_id=self.local_cluster_id, model_id=model_id)
            npu_cache = self.separate_deployment_engine.register_blocks_cache(
                self.cache_desc_map[model_id], npu_addrs, cache_key
                )
            self.separate_deployment_engine.set_npu_cache(model_id, npu_cache)
            logger.info('Register blocks cache success.')
        except Exception as e:
            logger.error(f"Failed to register blocks cache:{e}.")
            raise Exception("Failed to register blocks cache") from e

    def pull_blocks(self, remote_model_instance_id: int,
                    src_block_table: List[int], dst_block_table: List[int]):
        if remote_model_instance_id not in self.cluster_comm_map:
            logger.error(f"Pull_kv error: remote_model_instance_id: {remote_model_instance_id} is not linked.")
            return ErrorCode.TEXT_GENERATOR_PD_MODEL_INSTANCE_ID_ERROR
        
        # 遍历所有注册的model_id,检查block的范围并拉取对应kv cache
        for model_id in self.cache_desc_map:
            if not all(0 <= x < self.max_block_nums_map[model_id] for x in dst_block_table):
                logger.error(f"Pull_kv error: block id out of range, model_id is {model_id}.")
                return ErrorCode.TEXT_GENERATOR_PD_BLOCK_ID_OUT_OF_RANGE
            rt = self.separate_deployment_engine.pull_kv(model_id=model_id, src_block_table=src_block_table, 
                                        dst_block_table=dst_block_table, remote_cluster_id=remote_model_instance_id)
            if rt != MindieLlmStatusCode.SUCCESS:
                # 获取对端device_ip进行详细日志记录
                remote_device_ip = self.cluster_device_ip_map.get(remote_model_instance_id, "unknown")
                logger.error(f"{self.local_device_ip} pull kv from {remote_device_ip} failed, error code is {rt}.")
                return rt
        return MindieLlmStatusCode.SUCCESS

    def link(self, **kwargs):
        """
        使用滑动窗口建立多个链路。
        使用大小为16的滑动窗口批量建立链路，然后依次检查链路状态。
        """
        window_size = 16     # 滑动窗口大小
        remote_device_ip = 'remote_device_ip'
        link_params = 'link_params'
        remote_cluster_id = 'remote_cluster_id'
        
        # 使用 LinkParams 处理参数
        params = LinkParams(**kwargs)
        # 记录开始时间，用于全局超时检测
        self.link_start_time = time.time()
        
        # 准备返回的失败链路列表
        failed_links = []
        
        # 滑动窗口的数据结构，保存发起的链接请求
        # 每项包含: instance_id, index, comm_id, retry_count, link_params
        window = []
        
        # 遍历所有实例并建立链接
        instance_ids = list(params.remote_cluster_ids.keys())
        instance_idx = 0  # 当前处理的实例ID索引
        total_links = sum(len(params.remote_cluster_ids[instance_id]) for instance_id in instance_ids)
        link_count = 0    # 已经处理的链接数量
        current_index = 0 # 当前实例处理到的索引
        
        # 创建一个字典，记录每个实例当前处理到的索引位置
        instance_indices = {instance_id: 0 for instance_id in instance_ids}
        
        # 首先填充滑动窗口
        while len(window) < window_size and instance_idx < len(instance_ids):
            instance_id = instance_ids[instance_idx]
            current_index = instance_indices[instance_id]  # 获取当前实例处理到的索引
            
            while current_index < len(params.remote_cluster_ids[instance_id]):                  
                single_link_params = self._create_link_params(instance_id, current_index, params)
                
                link_status, window_item = self._try_create_link(single_link_params, 
                                                                 instance_id, current_index)
                if link_status:
                    if window_item is not None:
                        window.append(window_item)
                else:
                    logger.error(f"Link from {self.local_device_ip} to {single_link_params[remote_device_ip]} "
                                  f"failed, error code is {ErrorCode.TEXT_GENERATOR_PD_LINK_ERROR}.")
                    failed_links.append([single_link_params[remote_device_ip],
                                          ErrorCode.TEXT_GENERATOR_PD_LINK_ERROR])

                link_count += 1
                current_index += 1  # 增加当前实例的索引
                instance_indices[instance_id] = current_index  # 更新索引记录
                
                # 如果窗口已满，暂时跳出循环
                if len(window) >= window_size or link_count >= total_links:
                    break
            
            # 只有当前实例的所有链接都处理完，才增加实例索引
            if current_index >= len(params.remote_cluster_ids[instance_id]):
                instance_idx += 1
            
            # 如果已经达到总链接数或窗口已满，跳出外层循环
            if link_count >= total_links or len(window) >= window_size:
                break
        # 处理滑动窗口中的链接
        while window:
            # 检查是否超时
            link_item = window.pop(0)
            current_time = time.time()
            if current_time - self.link_start_time > self.link_time_out:
                logger.error(f"Link from {self.local_device_ip} to {link_item[link_params][remote_device_ip]} "
                             f"failed, error code is {ErrorCode.TEXT_GENERATOR_PD_LINK_OUT_OF_TIME}.")
                logger.error(f"Overall link process timed out after {self.link_time_out} seconds, "
                             f"error code is {ErrorCode.TEXT_GENERATOR_PD_LINK_OUT_OF_TIME}.")
                # 解除所有窗口中的链接
                for item in window:
                    ret = self.unlink(item[link_params][remote_cluster_id])
                    if ret != MindieLlmStatusCode.SUCCESS:
                        logger.error(f"Failed to unlink remote_cluster_id {item[link_params][remote_cluster_id]}, "
                                     f"error code is {ret}.")

                self._collect_remaining_failed_links(instance_ids, instance_idx, 
                                                     current_index, params, failed_links, window)
                # 输出汇总日志
                self._log_link_summary(failed_links)
                return failed_links
                
            status, window_item = self._query_mem_status(link_item)
            
            if status == MindieLlmStatusCode.SUCCESS:
                # 链接成功，尝试添加新的链接
                while instance_idx < len(instance_ids) and len(window) < window_size:
                    instance_id = instance_ids[instance_idx]
                    current_index = instance_indices[instance_id]  # 获取当前实例处理到的索引
                    while current_index < len(params.remote_cluster_ids[instance_id]):
                        if link_count >= total_links:
                            break
                        
                        new_link_params = self._create_link_params(instance_id, current_index, params)
                        link_status, window_item = self._try_create_link(new_link_params, instance_id,
                                                                        current_index)
                        if link_status: 
                            if window_item is not None:
                                window.append(window_item)
                        else:
                            failed_links.append([new_link_params[remote_device_ip], 
                                                 ErrorCode.TEXT_GENERATOR_PD_LINK_ERROR])
                        
                        link_count += 1
                        current_index += 1  # 增加当前实例的索引
                        instance_indices[instance_id] = current_index  # 更新索引记录
                        
                        if len(window) >= window_size:
                            break
                    
                    # 只有当前实例的所有链接都处理完，才增加实例索引
                    if current_index >= len(params.remote_cluster_ids[instance_id]):
                        instance_idx += 1
                    
                    # 如果窗口已满或已处理完所有链接，跳出循环
                    if len(window) >= window_size or link_count >= total_links or instance_idx >= len(instance_ids):
                        break
            
            elif status == MindieLlmStatusCode.TEXT_GENERATOR_PD_RETRY_LINK:
                # 失败需要重试，把新的 link_item 添加到窗口末尾
                if window_item is not None:
                    window.append(window_item)
            elif status == ErrorCode.TEXT_GENERATOR_PD_LINK_OUT_OF_TIME:
                logger.error(f"Overall link process timed out after {self.link_time_out} seconds, "
                             f"error code is {ErrorCode.TEXT_GENERATOR_PD_LINK_OUT_OF_TIME}.")
                failed_links.append([link_item[link_params][remote_device_ip], 
                                     ErrorCode.TEXT_GENERATOR_PD_LINK_OUT_OF_TIME])
                for item in window:
                    ret = self.unlink(item[link_params][remote_cluster_id])
                    if ret != MindieLlmStatusCode.SUCCESS:
                        logger.error(f"Failed to unlink remote_cluster_id {item[link_params][remote_cluster_id]}, "
                                     f"error code is {ret}.")
                self._collect_remaining_failed_links(instance_ids, instance_idx, 
                                                     current_index, params, failed_links, window)
                # 输出汇总日志
                self._log_link_summary(failed_links)
                return failed_links
            else:
                logger.error(f"Link from {self.local_device_ip} to {link_item[link_params][remote_device_ip]} "
                              f"failed, error code is {ErrorCode.TEXT_GENERATOR_PD_LINK_ERROR}.")
                failed_links.append([link_item[link_params][remote_device_ip], 
                                     ErrorCode.TEXT_GENERATOR_PD_LINK_ERROR])
                
        # 输出汇总日志
        self._log_link_summary(failed_links)
        
        # 返回失败的链路列表
        return failed_links
    

    def unlink(self, remote_cluster_id):
        if remote_cluster_id not in self.cluster_comm_map:  
            logger.error(f"Unlink failed, remote_cluster_id:{remote_cluster_id} is not linked.")
            return ErrorCode.TEXT_GENERATOR_PD_UNLINK_ERROR
        comm_id = self.cluster_comm_map.pop(remote_cluster_id)
        # 同时从device_ip映射中移除
        self.cluster_device_ip_map.pop(remote_cluster_id, None)
        return self.separate_deployment_engine.unlink(comm_id)
    
    def unlink_all(self):
        for cluster_id in list(self.cluster_comm_map):
            ret = self.unlink(cluster_id)
            if ret != MindieLlmStatusCode.SUCCESS:
                logger.error(f"Failed to unlink cluster {cluster_id}, error code is {ret}.")
                raise Exception("SeparateDeploymentEngine: unlink_all fail")

    def finalize(self):
        self.unlink_all()
        self.separate_deployment_engine.finalize()

    def _try_create_link(self, link_params, instance_id=None, index=None):
        """
        尝试创建一个链接，支持重试机制
        
        参数:
            link_params: 链接参数
            instance_id: 实例ID
            index: 索引
            
        返回:
            (connect_success, window_item): connect_success: 标志是否成功，window_item: 失败时为None,成功时返回一个窗口项
        """
        try:
            # 创建RankInfo对象
            rank_info = RankInfo(
                remote_cluster_id=link_params['remote_cluster_id'], 
                remote_physical_device_id=link_params['remote_physical_device_id'], 
                remote_device_ip=link_params['remote_device_ip'], 
                remote_host_ip=link_params['remote_host_ip'],
                local_host_ip=self.local_host_ip, 
                local_cluster_id=self.local_cluster_id, 
                local_device_ip=self.local_device_ip, 
                local_physical_device_id=self.local_physical_device_id,
                local_super_pod_id=self.local_super_pod_id,
                remote_super_pod_id=link_params.get('remote_super_pod_id'),
                local_super_device_id=self.local_super_device_id,
                remote_super_device_id=link_params.get('remote_super_device_id')
            )
            
            cluster_rank_info = rank_info.get_cluster_rank_info()
            rank_table = rank_info.get_rank_table()
            
            # 尝试链接
            result = self.separate_deployment_engine.link(cluster_rank_info, rank_table)
            
            # 从结果中提取状态和comm_id
            status = result.get("status")
            comm_id = result.get("comm_id")
            
            if status == MindieLlmStatusCode.TEXT_GENERATOR_PD_ALREADY_LINK:
                logger.info(f"Device{link_params['remote_physical_device_id']} already link.")
                return True, None  # 已建链视为成功
            
            elif status == MindieLlmStatusCode.SUCCESS and comm_id is not None:
                self.cluster_comm_map[link_params['remote_cluster_id']] = comm_id
                # 创建窗口项
                window_item = {
                    'instance_id': instance_id,
                    'index': index,
                    'comm_id': comm_id,
                    'retry_count': 0,
                    'link_params': link_params
                }
                return True, window_item
            else:
                #通信域初始化失败，status = ErrorCode.TEXT_GENERATOR_PD_LINK_ERROR
                logger.error(f"Link from {self.local_device_ip} to {link_params['remote_device_ip']} failed, "
                             f"error code is {status}.")
                return False, None
        
        except Exception as e:
            logger.error(f"Link exception from {self.local_device_ip} to {link_params['remote_device_ip']}: "
                        f"{str(e)}.")
            return False, None

    def _query_mem_status(self, link_item):
        """
        查询链路状态并处理结果
        
        参数:
            link_item: 链路项，包含 comm_id, retry_count, link_params 等信息
            
        返回:
            (status, result):   status: 状态码, result: 成功时为None，重试时为新的link_item
        """
        comm_id = link_item['comm_id']
        retry_count = link_item['retry_count']
        link_params = link_item['link_params']
        instance_id = link_item['instance_id']
        index = link_item['index']
        start_time = self.link_start_time
        time_out = self.link_time_out
        remote_cluster_id = link_params['remote_cluster_id']
        
        # 限制查询次数，避免无限轮询
        max_query_attempts = 3
        query_attempt = 0
        
        while query_attempt < max_query_attempts:
            # 每次查询前都检查超时
            current_time = time.time()
            if current_time - start_time > time_out:
                ret = self.unlink(remote_cluster_id)
                if ret != MindieLlmStatusCode.SUCCESS:
                    logger.error(f"Failed to unlink remote_cluster_id {remote_cluster_id}, error code is {ret}.")
                logger.error(f"Link from {self.local_device_ip} to {link_params['remote_device_ip']} "
                             f"timed out after {current_time - start_time:.2f}s, "
                             f"error code is {ErrorCode.TEXT_GENERATOR_PD_LINK_OUT_OF_TIME}.")
                return ErrorCode.TEXT_GENERATOR_PD_LINK_OUT_OF_TIME, None  # 超时，失败不重试
            
            try:
                # 记录查询开始时间
                query_start_time = time.time()
                logger.debug(f"Querying mem status for {link_params['remote_device_ip']}, attempt {query_attempt + 1}")
                
                # 尝试查询内存状态
                query_status = self.separate_deployment_engine.query_register_mem_status(comm_id)
                
                # 记录查询结束时间
                query_end_time = time.time()
                query_duration = query_end_time - query_start_time
                logger.debug(f"Query completed in {query_duration:.2f}s for {link_params['remote_device_ip']}")
                
                if query_status == RegisterMemStatus.OK:
                    logger.info(f"Link from local_device_ip {self.local_device_ip} "
                                f"to remote_device_ip {link_params['remote_device_ip']} success.")
                    # 建立cluster_id到device_ip的映射
                    self.cluster_device_ip_map[link_params['remote_cluster_id']] = link_params['remote_device_ip']
                    return MindieLlmStatusCode.SUCCESS, None  # 成功
                    
                elif query_status == RegisterMemStatus.FAILED:
                    # 查询失败，进行重试
                    logger.warning(f"Mem status query failed for {link_params['remote_device_ip']}, will retry")
                    ret = self.unlink(remote_cluster_id)
                    if ret != MindieLlmStatusCode.SUCCESS:
                        logger.error(f"Failed to unlink remote_cluster_id {remote_cluster_id}, error code is {ret}.")
                    return self._handle_retry_logic(retry_count, link_params, instance_id, index)
                else:
                    # 状态为PENDING或其他，继续下一次查询
                    query_attempt += 1
                    # 短暂等待后再次查询
                    if query_attempt < max_query_attempts:
                        time.sleep(0.05)  
            
            except Exception as e:
                logger.warning(f"Mem status query failed for {link_params['remote_device_ip']} with exception{e}.")
                ret = self.unlink(remote_cluster_id)
                if ret != MindieLlmStatusCode.SUCCESS:
                    logger.error(f"Failed to unlink remote_cluster_id {remote_cluster_id}, error code is {ret}.")
                return self._handle_retry_logic(retry_count, link_params, instance_id, index)
        
        # 达到最大查询次数，但状态仍然不确定，重新加入队列稍后处理
        logger.debug(f"Max query attempts reached for {link_params['remote_device_ip']}, requeueing")
        return MindieLlmStatusCode.TEXT_GENERATOR_PD_RETRY_LINK, link_item

    def _handle_retry_logic(self, retry_count, link_params, instance_id, index):
        """
        处理链接重试逻辑，无重试次数限制
        
        参数:
            retry_count: 当前重试次数
            link_params: 链接参数
            instance_id: 实例ID
            index: 索引
            
        返回:
            (status_code, window_item): 状态码和窗口项(None表示失败)
        """
        # 记录重试警告
        logger.warning(f"Query exception for link to {link_params['remote_device_ip']}, "
                        f"retrying (attempt {retry_count+1})...")
        
        # 重新尝试建链
        link_status, window_item = self._try_create_link(link_params, instance_id, index)
        if link_status and window_item is not None:
            # 更新重试计数
            window_item['retry_count'] = retry_count + 1
            return MindieLlmStatusCode.TEXT_GENERATOR_PD_RETRY_LINK, window_item  # 失败需重试，返回新的link_item
        else:
            # 重试失败
            return ErrorCode.TEXT_GENERATOR_PD_LINK_ERROR, None  # 失败不重试

    def _log_link_summary(self, failed_links):
        """输出链路建立的汇总日志"""
        if len(failed_links) > 0:
            # 记录所有失败的链路
            for failed_link in failed_links:
                logger.error(
                    f"[Link_Summary] from {self.local_device_ip} to {failed_link[0]} failed, "
                    f"error code is {failed_link[1]}."
                )
            # 可选：输出失败链路总数
            logger.error(f"[Link_Summary] Total failed links: {len(failed_links)}.")
        else:
            # 所有建链成功
            logger.info(f"[Link_Summary] from {self.local_device_ip} to all remote_device_ip success.")