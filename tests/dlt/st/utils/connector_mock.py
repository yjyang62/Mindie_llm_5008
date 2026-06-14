# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import sys
import base64

from mindie_llm.connector.request_listener.shared_mem_communication import SharedMemory, SharedMemCommunication
from mindie_llm.connector.main import main

# 是否需要录制数据
record_data = True
mock_request = False
record_data_file = "/home/connector_mock_data.txt"
record_index = 0


def get_obj_with_record(self, class_name):
    global record_index
    self.sem_consumer.acquire()
    # 前四个字节表示buffer长度
    length = int.from_bytes(self.shm.buf[:4], 'little')
    if length == 0:
        raise ValueError("proto_data length is 0")
    proto_data = bytes(self.shm.buf[4:length + 4])
    # 新增数据写入
    with open(record_data_file, 'a', encoding='utf-8') as fi:
        fi.write(str(record_index) + "request:\n")
        fi.write(base64.b64encode(proto_data).decode('utf-8'))
        fi.write("\n")

    self.sem_producer.release()
    # Protobuf反序列化
    obj = class_name()
    obj.ParseFromString(proto_data)
    return obj


def send_obj_with_record(self, obj, offset):
    global record_index
    # 序列化
    proto_data = obj.SerializeToString()
    length = len(proto_data)

    self.sem_producer.acquire()
    # 前四个字节表示buffer长度
    self.shm.buf[offset:4 + offset] = length.to_bytes(4, 'little')
    self.shm.buf[4 + offset:length + 4 + offset] = proto_data
    with open(record_data_file, 'a', encoding='utf-8') as fi:
        fi.write(str(record_index) + "response:\n")
        fi.write(base64.b64encode(proto_data).decode('utf-8'))
        fi.write("\n")
    self.sem_consumer.release()
    record_index += 1


def get_obj_mock(self, class_name):
    global record_index
    self.sem_consumer.acquire()
    # 前四个字节表示buffer长度
    length = int.from_bytes(self.shm.buf[:4], 'little')
    if length == 0:
        raise ValueError("proto_data length is 0")
    self.sem_producer.release()
    # 根据收到的数据查找对应的结果
    target_line = None
    with open(record_data_file, 'r', encoding='utf-8') as fi:
        for line in fi:
            if line.strip().startswith(str(record_index) + 'response:'):
                target_line = next(fi).strip()
    record_index += 1
    proto_data_response = target_line.encode('utf-8') if target_line else None
    if proto_data_response is not None:
        proto_data_response = base64.b64decode(proto_data_response)
        from mindie_llm.connector.common.model_execute_data_pb2 import ExecuteResponse
        obj = ExecuteResponse()
        obj.ParseFromString(proto_data_response)
        SharedMemCommunication.send_model_execute_response_cls(obj)

    return None


if mock_request:
    # 动态替换方法
    SharedMemory.get_obj = get_obj_mock

if record_data:
    # 文件清空
    with open(record_data_file, 'w') as f:
        f.write("")
    # 动态替换方法
    SharedMemory.get_obj = get_obj_with_record
    SharedMemory.send_obj = send_obj_with_record

if __name__ == '__main__':
    sys.exit(main())