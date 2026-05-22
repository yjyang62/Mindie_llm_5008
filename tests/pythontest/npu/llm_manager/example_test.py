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

import time
from llm_manager_python_api_demo.engine import Engine
from llm_manager_python_api_demo.request import Request
from llm_manager_python_api_demo.data import Data
from llm_manager_python_api_demo.request import RequestId
import numpy as np
from llm_manager_python_api_demo.dtype import DType
from llm_manager_python_api_demo.sampling import SamplingParams
from llm_manager_python_api_demo.response import Response
from mindie_llm.utils.log.logging import logger


def response_callback(response: Response):
    logger.info(f"Request id: {response.get_request_id()}")
    logger.info(f"Output id: {response.get_output_id()}")


def async_interface_test():
    # 编写样例
    # 1、初始化engine
    engine = Engine()
    status = engine.init()
    if not status.is_ok():
        raise ValueError(f"engine init error: {status.get_msg()}")
    # 2、资源查询
    remain_blocks, remain_prefill_slots, remain_prefill_tokens = engine.get_request_block_quotas()
    processing_num = engine.get_processing_request()
    logger.info(f"processing_num: {processing_num}")
    # 3、创建请求
    request = Request(RequestId("test"))
    # 4、 设置回调函数
    request.set_send_response_callback(response_callback)
    # 5、设置请求输入数据
    data_val = [
        1, 2627, 300, 30010, 29879, 868, 4684, 6568, 29871, 29896, 29953, 29808, 639, 2462, 29889, 2296, 321, \
        1446, 2211, 363, 26044, 1432, 7250, 322, 289, 6926, 286, 3096, 1144, 363, 902, 7875, 1432, 2462, 411, \
        3023, 29889, 2296, 269, 10071, 278, 21162, 472, 278, 2215, 13269, 29915, 9999, 14218, 363, 395, 29906, \
        639, 10849, 868, 384, 19710, 29889, 1128, 1568, 297, 17208, 947, 1183, 1207, 1432, 2462, 472, 278, \
        2215, 13269, 29915, 9999, 29973
    ]
    data_size = len(data_val)
    shape = np.array([1, data_size], dtype=np.int64)
    data = Data()
    data.set_token_id(DType.TYPE_INT64, shape, np.array(data_val, dtype=np.int64))
    status1 = request.set_data_to_request(data)
    if not status1.is_ok():
        raise ValueError(f"engine set data error : {status1.get_msg()}")
    # 6、设置后处理参数
    request.set_sampling_params(SamplingParams(1.0, 0, 1.0, 1.0, False, 1, 1.0, False, 0.0, 0.0))
    # 7、执行异步推理
    status3 = engine.async_forward(request)
    if not status3.is_ok():
        raise ValueError(f"engine forward error : {status3.get_msg()}")
    time.sleep(10)
    # 8、析构推理引擎
    engine.finalize()

if __name__ == '__main__':
    async_interface_test()