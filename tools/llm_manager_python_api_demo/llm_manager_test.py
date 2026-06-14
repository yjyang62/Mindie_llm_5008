#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from dataclasses import dataclass, field
from collections import defaultdict
import argparse
import json
import os
import signal
import threading
import time
from threading import Lock
from typing import Dict, List
import copy

from llm_manager_python_api_demo.metrics import Metrics, Statistics, TtimeT, \
    format_metrics, print_statistics, write_output_ids
from llm_manager_python_api_demo.io_manager import IOManager
from llm_manager_python_api_demo.engine import Engine
from llm_manager_python_api_demo.data import Data
from llm_manager_python_api_demo.request import Request
from llm_manager_python_api_demo.request_id import RequestId
from llm_manager_python_api_demo.sampling import SamplingParams
from llm_manager_python_api_demo import llm_manager_python
from llm_manager_python_api_demo.status import Status
from mindie_llm.utils.file_utils import safe_open
from mindie_llm.utils.log.logging import logger, print_log
from mindie_llm.utils.env import ENV

g_manager: IOManager = None
g_statistics: Statistics = Statistics()
g_metrics: Dict[str, Metrics] = {}
g_complete_num: int = 0
g_warmup_completed: int = 0
g_warmup_num: int = 10

g_mutex_warmup = Lock()
g_mutex = Lock()
g_metrics_mutex = Lock()

g_record_output = False
g_responses = []

g_use_beam_search = False
g_beam_search_width = 1


@dataclass
class BeamSearchResponse:
    cache_seq: defaultdict = field(default_factory=lambda: defaultdict(list))
    cache_seq_tokens: defaultdict = field(default_factory=lambda: defaultdict(list))
    finished_seq: defaultdict = field(default_factory=lambda: defaultdict(list))
    finished_seq_tokens: defaultdict = field(default_factory=lambda: defaultdict(list))
    finished_seq_id: int = 0

g_beam_search_response: defaultdict = defaultdict(BeamSearchResponse)


@dataclass
class ResponseData:
    req_id: str
    output: llm_manager_python.TensorMap
    is_final: bool
    err_msg: str
    ending_time: TtimeT


def parse_bool(bool_str):
    bool_str = bool_str.lower()
    if bool_str == 'true':
        return True
    elif bool_str == 'false':
        return False
    else:
        raise ValueError('Failed to parse a bool variable, please check if there is a bool arg given a special string '
                         'which is neither True nor False.')


def response_callback(req_id, output, is_final, err_msg):
    global g_record_output
    global g_complete_num
    global g_responses
    llm_req_id = Engine.convert_request_id(req_id)
    now = TtimeT()
    g_manager.set_output_data(str(req_id))
    g_responses.append(ResponseData(llm_req_id, output, is_final, err_msg, now))

    if is_final:
        with g_mutex:
            g_complete_num += 1
            print_log(ENV.rank, logger.info, f"ReqId: {llm_req_id} Finished")


def update_beam_search_cache(req_id, current_cache_id, parent_id):
    global g_beam_search_width
    global g_beam_search_response

    for key in current_cache_id:
        if key not in parent_id:
            g_beam_search_response[req_id].cache_seq.pop(key)

    while len(g_beam_search_response[req_id].finished_seq) > g_beam_search_width:
        finished_seq = g_beam_search_response[req_id].finished_seq
        min_finished_seq = min(finished_seq, key=finished_seq.get)
        g_beam_search_response[req_id].finished_seq.pop(min_finished_seq)
        g_beam_search_response[req_id].finished_seq_tokens.pop(min_finished_seq)


def process_responses():
    global g_responses
    global g_use_beam_search
    global g_beam_search_response
    global g_beam_search_width

    for response_data in g_responses:
        output_len = 1
        response = Engine.construct_response_by_tensor_map(
            response_data.req_id, 
            response_data.output, 
            response_data.is_final, 
            response_data.err_msg)
        response_parse_eos_out = response.parse_eos_attr()
        if isinstance(response_parse_eos_out, Status) and not response_parse_eos_out.is_ok():
            raise RuntimeError(f"{response_parse_eos_out.get_msg()}")
        _, req_output_len = response_parse_eos_out
        output_len = sum(req_output_len)

        try:
            g_metrics[response_data.req_id].tokens_output += output_len
            if g_metrics[response_data.req_id].first_token_cost == 0:
                # prefill 记录首token时间
                decode_time = response_data.ending_time - g_metrics[response_data.req_id].starting_time
                g_metrics[response_data.req_id].first_token_cost = decode_time
            else:
                # decode 记录每次decode的时间
                decode_time = response_data.ending_time - g_metrics[response_data.req_id].last_token_time
                # 针对投机场景适配，decode返回小于等于gamma个token，四舍五入
                avg_decode_time = (decode_time + output_len // 2) // output_len
                for _ in range(output_len):
                    g_metrics[response_data.req_id].decode_time.append(avg_decode_time)

            g_metrics[response_data.req_id].last_token_time = response_data.ending_time

            if g_use_beam_search:
                req_id = response_data.req_id
                cumulative_logprobs = response.get_cumulative_logprobs().tolist()
                seq_ids = response.get_seq_id().tolist()
                output_ids = response.get_output_id().tolist()
                parent_seq_ids = response.get_parent_seq_id().tolist()
                current_cache_id = copy.deepcopy(g_beam_search_response[req_id].cache_seq)
                current_cache_tokens = copy.deepcopy(g_beam_search_response[req_id].cache_seq_tokens)
                for index, seq_id in enumerate(seq_ids):
                    parent_seq_id = parent_seq_ids[index]
                    parent_tokens = current_cache_tokens[parent_seq_id]
                    output_id = output_ids[index]
                    if seq_id == -1:
                        resp_finished_seq_id = g_beam_search_response[req_id].finished_seq_id
                        g_beam_search_response[req_id].finished_seq[resp_finished_seq_id] = cumulative_logprobs[index]
                        g_beam_search_response[req_id].finished_seq_tokens[resp_finished_seq_id].extend(parent_tokens)
                        g_beam_search_response[req_id].finished_seq_tokens[resp_finished_seq_id].extend(output_id)
                        g_beam_search_response[req_id].finished_seq_id += 1
                    else:
                        if seq_id not in current_cache_id.keys():
                            g_beam_search_response[req_id].cache_seq_tokens[seq_id].extend(parent_tokens)
                        g_beam_search_response[req_id].cache_seq_tokens[seq_id].extend(output_id)
                        g_beam_search_response[req_id].cache_seq[seq_ids[index]] = cumulative_logprobs[index]

                update_beam_search_cache(req_id, current_cache_id, parent_seq_ids)

            # 生成token id
            if g_record_output:
                g_metrics[response_data.req_id].output_token_ids.extend(response.get_output_id())
            if response.is_eos():
                g_metrics[response_data.req_id].ending_time = response_data.ending_time
                # 最后一个Token耗时
                g_metrics[response_data.req_id].last_token_cost = decode_time
                if g_use_beam_search:
                    beam_search_response = g_beam_search_response[response_data.req_id]
                    all_finished_seqs = beam_search_response.finished_seq
                    selected_seqs = sorted(all_finished_seqs.items(),
                                            key=lambda x: x[1], reverse=True)[:g_beam_search_width // 2]
                    for index, (selected_seq, _) in enumerate(selected_seqs):
                        selected_seq_tokens = beam_search_response.finished_seq_tokens[selected_seq]
                        print_log(ENV.rank, logger.info,
                            f"request id({response_data.req_id}) - beam_{index} : {selected_seq_tokens}")
                    g_beam_search_response.pop(response_data.req_id)
        except KeyError as e:
            raise KeyError(f"Invalid key {response_data.req_id} in g_metrics") from e


def datas_to_request(data_list: List[Data], sampling_params: SamplingParams) -> List[Request]:
    total_num = len(data_list)
    requests = []
    for i in range(total_num):
        data = data_list[i]
        request = Request(RequestId(data.get_id()))
        status = request.set_data_to_request(data)
        if not status.is_ok():
            raise ValueError(f"engine set data error : {status.get_msg()}")
        status = request.set_sampling_params(sampling_params)
        if not status.is_ok():
            raise ValueError(f"engine set sampling error : {status.get_msg()}")
        status = request.set_input_token_num(len(data))
        if not status.is_ok():
            raise ValueError(f"engine set input token num error : {status.get_msg()}")
        requests.append(request)
    return requests


def warmup(engine: Engine, manager: IOManager, warmup_size: int, sampling_params: SamplingParams):
    global g_warmup_completed
    warmup_data_list = manager.get_warmup_inputs(warmup_size)
    total_warmup_num = len(warmup_data_list)
    print_log(ENV.rank, logger.info, f"Total warm up count: {total_warmup_num}")
    warmup_requests = datas_to_request(warmup_data_list, sampling_params)    
        
    invalid_req_num = 0

    def warmup_response_callback(req_id, output, is_final, err_msg):
        global g_warmup_completed
        if is_final:
            with g_mutex_warmup:
                g_warmup_completed += 1
                print_log(ENV.rank, logger.info, f"Warm up completed count: {g_warmup_completed}")

    for i in range(total_warmup_num):
        warmup_requests[i].set_send_response_callback(warmup_response_callback)
        status = engine.async_forward(warmup_requests[i])
        if not status.is_ok():
            invalid_req_num += 1

    print_log(ENV.rank, logger.info, f"Invalid warmup request count: {invalid_req_num}")

    while g_warmup_completed < total_warmup_num - invalid_req_num:
        with g_mutex_warmup:
            time.sleep(0.01)  # Wait for 10 milliseconds


def forward(engine: Engine, request: Request, req_id: str, invalid_req_num: List[int]):
    ret = engine.async_forward(request)
    if not ret.is_ok():
        with g_metrics_mutex:
            invalid_req_num[0] += 1
            g_statistics.request_number -= 1
            g_metrics.pop(req_id, None)


def send_request_inner(engine: Engine, data: List[Data], sampling_params: SamplingParams, invalid_req_num: List[int]):
    if data:
        requests = datas_to_request(data, sampling_params)
        g_statistics.request_number += len(requests)  # total num

        # 4. forward (asynchronously)
        for i, request in enumerate(requests):
            req_id = request.get_request_id().get_id_value()
            with g_metrics_mutex:
                g_metrics[req_id] = Metrics()
                g_metrics[req_id].starting_time = TtimeT()
                g_metrics[req_id].tokens_input = len(data[i])

            thread = threading.Thread(target=forward, args=(engine, request, req_id, invalid_req_num))
            thread.start()


def send_request(engine: Engine, sampling_params: SamplingParams, max_batch_size):
    processing_num = engine.get_processing_request()
    print_log(ENV.rank, logger.info, f"the processing request num is {processing_num} at first.")

    remain_prefill_slots = 0
    remain_prefill_tokens = 0
    invalid_req_num = [0]
    while not g_manager.empty():
        # 2. 获取可用的slot数目
        _, remain_prefill_slots, remain_prefill_tokens = engine.get_request_block_quotas()
        processing_num = engine.get_processing_request()
        slot_num = max_batch_size - processing_num
        if remain_prefill_slots > 0 and remain_prefill_tokens > 0:
            # 3. Set input
            data = g_manager.get_input_data_by_quotas(remain_prefill_slots, remain_prefill_tokens, slot_num)
            send_request_inner(engine, data, sampling_params, invalid_req_num)
        
        time.sleep(0.02)  # Sleep for 20 milliseconds

    processing_num = engine.get_processing_request()
    print_log(ENV.rank, logger.info, f"the processing request num is {processing_num} when all requests dispatched.")
    print_log(ENV.rank, logger.info, f"invalid request count is {invalid_req_num[0]}")


def get_model_info(config_path: str):
    with safe_open(config_path, 'r') as file:
        config = json.load(file)
        backend_config = config.get('BackendConfig', {})
        model_deploy_config = backend_config.get('ModelDeployConfig', {})
        model_config = model_deploy_config.get('ModelConfig', [])
        if model_config:
            model_name = model_config[0].get('modelName', "")
        multi_nodes_infer_enabled = backend_config.get('multiNodesInferEnabled', False)
        if not multi_nodes_infer_enabled:
            tp = model_config[0].get("worldSize", 1)
            server_count = 1
    return model_name, tp, server_count


def run_engine():
    global g_manager
    g_manager = IOManager()

    args = parse_arguments()

    global g_record_output
    g_record_output = args.record_output
    dataset = args.dataset_path
    config_path = args.config_path
    load_all_data = args.load_all_data
    global g_use_beam_search
    global g_beam_search_width
    g_use_beam_search = args.use_beam_search
    g_beam_search_width = 2 * args.param_n

    sampling_params = SamplingParams(in_temperature=args.temperature,
        in_top_k=args.top_k,
        in_top_p=args.top_p,
        in_typical_p=1.0,
        in_do_sample=args.do_sample,
        in_seed=args.seed,
        in_repetition_penalty=args.repetition_penalty,
        in_watermark=False,
        in_frequency_penalty=args.frequency_penalty,
        in_presence_penalty=args.presence_penalty,
        logprobs=args.logprobs,
        top_logprobs=args.top_logprobs,
        best_of=args.best_of,
        n=args.param_n,
        use_beam_search=args.use_beam_search)
    
    if g_manager.set_input_data(dataset) != 0:
        print_log(ENV.rank, logger.error, "Failed to load data")
        raise RuntimeError("Failed to load data")

    engine = Engine()
    status = engine.init(config_path, response_callback, load_all_data, len(g_manager.get_inputs()))
    if not status.is_ok():
        raise ValueError(f"engine init error: {status.get_msg()}")

    schuduler_config = engine.get_scheduler_config(config_path)
    max_batch_size = schuduler_config["maxBatchSize"]

    if not load_all_data:
        print_log(ENV.rank, logger.info, "*** Warm up ***")
        warmup(engine, g_manager, g_warmup_num, sampling_params)
        print_log(ENV.rank, logger.info, "*** Warm up end***")

    start = TtimeT()
    send_request(engine, sampling_params, max_batch_size)

    while g_complete_num < g_statistics.request_number:
        time.sleep(0.01)  # Sleep for 10 milliseconds

    end = TtimeT()

    process_responses()

    # 5. 统计打点信息
    g_statistics.model_full_name = ""
    g_statistics.model_full_name, g_statistics.tp, g_statistics.server_count = get_model_info(config_path)
    g_statistics.latency_for_all = end - start

    format_metrics(g_metrics, g_statistics)
    print_statistics(g_statistics)

    if g_record_output:
        output_tokens_id: Dict[str, List[int]] = {}
        for key, metric in g_metrics.items():
            output_tokens_id[key] = metric.output_token_ids
        write_output_ids(output_tokens_id, "token_output.csv", "./")

    # 6. 释放资源
    status = engine.finalize()
    print_log(ENV.rank, logger.info, f"inferenceEngine finalize message is : {status.get_msg()}")


def signal_interrupt_handler(signum, frame):
    print_log(ENV.rank, logger.info, f"Received signal[{signum}]")
    print_log(ENV.rank, logger.info, "Test program is exiting...")
    while True:
        pid, status = os.waitpid(0, os.WNOHANG)
        if pid > 0:
            print_log(ENV.rank, logger.info, f"Test program wait pid with {pid}, status {status}")
        else:
            break
    os.killpg(os.getpgrp(), signal.SIGKILL)


def signal_chld_handler(signum, frame):
    print_log(ENV.rank, logger.info, f"received SIGCHLD signal[{signum}]")
    exit_flag = False
    while True:
        try:
            pid, status = os.waitpid(0, os.WNOHANG)
            if pid == 0:
                break
            print_log(ENV.rank, logger.info, f"Test program wait pid with {pid}, status {status}")
            if not os.WIFEXITED(status):
                exit_flag = True
        except ChildProcessError:
            break

    if exit_flag:
        print_log(ENV.rank, logger.info, f"received SIGCHLD signal[{signum}]")
        os.killpg(os.getpgrp(), signal.SIGKILL)


def register_signal():
    try:
        signal.signal(signal.SIGINT, signal_interrupt_handler)
        signal.signal(signal.SIGTERM, signal_interrupt_handler)
        signal.signal(signal.SIGCHLD, signal_chld_handler)
    except ValueError:
        print_log(ENV.rank, logger.error, "Error registering signal handlers.")
    except Exception as e:
        print_log(ENV.rank, logger.error, f"An unexpected error occurred: {e}")


def parse_arguments():
    store_true = 'store_true'

    parser = argparse.ArgumentParser()

    # 数据集与推理引擎参数
    parser.add_argument('--dataset_path', type=str, default='token_input_gsm.csv')
    parser.add_argument('--config_path', type=str, default='config.json')
    parser.add_argument('--load_all_data', type=bool, default=False)
    parser.add_argument('--record_output', type=bool, default=False)

    # 请求后处理参数
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--frequency_penalty', type=float, default=0.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--do_sample', type=parse_bool, default=True)
    parser.add_argument('--top_logprobs', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)

    # 请求生成策略参数
    parser.add_argument('--best_of', type=int, default=None)
    parser.add_argument('--param_n', type=int, default=1)
    parser.add_argument('--use_beam_search', action=store_true)
    parser.add_argument('--logprobs', action=store_true)

    return parser.parse_args()


def main():
    os.setpgrp()
    register_signal()

    business_thread = threading.Thread(target=run_engine)
    business_thread.start()
    business_thread.join()


if __name__ == "__main__":
    main()