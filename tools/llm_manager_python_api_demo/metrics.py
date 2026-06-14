#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from datetime import datetime, timedelta, timezone
import math
from typing import List, Dict
from mindie_llm.utils.file_utils import safe_open
from mindie_llm.utils.log.logging import logger, print_log
from mindie_llm.utils.env import ENV


SHA_TZ = timezone(
    timedelta(hours=8),
    name='Asia/Shanghai',
)


class TtimeT:
    def __init__(self):
        self.time = datetime.now(tz=SHA_TZ)

    def __sub__(self, other):
        if isinstance(other, TtimeT):
            return (self.time - other.time).total_seconds() * 1e3  # 转换为ms
        else:
            raise TypeError("Unsupported operand type(s) for -: 'TtimeT' and '{}'".format(type(other).__name__))


class Metrics:
    def __init__(self):
        self.starting_time = TtimeT()  # 开始时间
        self.ending_time = TtimeT()    # 结束时间
        self.last_token_time = TtimeT() # 最后一个Token的生成时间

        self.tokens_input = 0          # 输入Token数量
        self.tokens_output = 0         # 输出Token数量

        self.first_token_cost = 0       # 首Token耗时
        self.last_token_cost = 0        # 最后Token耗时

        self.decode_time = []           # 每一轮decode的耗时
        self.output_token_ids = []      # 输出的Token Id集合

        self.end = False                # 是否结束


class Statistics:
    def __init__(self):
        self.model_full_name = ""

        self.server_count = 1          # 节点数量
        self.tp = 8    # 张量并行数
        self.pp = 1  # 流水线并行数

        self.latency_for_all = 0        # E2E总耗时
        self.request_number = 0         # 总请求数量
        self.latency_for_per_infer = 0  # 从Forward开始到推理出last token耗时

        self.average_input_length = 0   # 平均输入长度
        self.average_output_length = 0  # 平均输出长度

        self.lpct = 0 # latency_per_complete_token
        self.lpot = 0 # latency_per_output_token

        self.p10_lpot = 0
        self.p20_lpot = 0
        self.p30_lpot = 0
        self.p40_lpot = 0
        self.p50_lpot = 0
        self.p60_lpot = 0
        self.p80_lpot = 0
        self.p90_lpot = 0
        self.p95_lpot = 0
        self.p99_lpot = 0
        self.p_max_lpot = 0

        self.average_first_token = 0    # 平均首Token耗时
        self.p99_first_token = 0        # 99分位首Token耗时
        self.max_first_token = 0        # 最大首Token耗时

        self.average_last_token = 0     # 平均最后Token耗时
        self.p99_last_token = 0         # 99分位最后Token耗时
        self.max_last_token = 0         # 最大最后Token耗时

        self.max_decode = 0             # 最大Decode耗时

        self.qps = 0.0                  # 每分钟完成请求数
        self.qps_per_npu = 0.0          # 每卡每分钟完成请求数


def get_current_time() -> str:
    now = datetime.now(tz=SHA_TZ)
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_time


def write_output_ids(all_output_ids: Dict[str, List[int]], out_csv: str, directory: str) -> None:
    output_vec = list(all_output_ids.items())
    output_vec.sort(key=lambda x: int(x[0]))

    filepath = f"{directory}{out_csv}"

    with safe_open(filepath, 'w') as out_file:
        for _, output_ids in output_vec:
            if output_ids:
                out_file.write(f"{output_ids[0]},")
                out_file.write(",".join(str(id) for id in output_ids[1:]))
                out_file.write("\r\n")


def quantile(data: List[int], q: float) -> int:
    data.sort()

    n = len(data)
    index = (n - 1) * q
    lo = math.floor(index)
    hi = math.ceil(index)
    qs = data[lo]
    h = index - lo
    return (1.0 - h) * qs + h * data[hi]


def calc_lpot(statistics: Statistics, decode_times: List[int]) -> None:
    decode_costs = sum(decode_times)
    statistics.lpot = decode_costs / len(decode_times)
    statistics.p10_lpot = quantile(decode_times, 0.1)
    statistics.p20_lpot = quantile(decode_times, 0.2)
    statistics.p30_lpot = quantile(decode_times, 0.3)
    statistics.p40_lpot = quantile(decode_times, 0.4)
    statistics.p50_lpot = quantile(decode_times, 0.5)
    statistics.p60_lpot = quantile(decode_times, 0.6)
    statistics.p80_lpot = quantile(decode_times, 0.8)
    statistics.p90_lpot = quantile(decode_times, 0.9)
    statistics.p95_lpot = quantile(decode_times, 0.95)
    statistics.p99_lpot = quantile(decode_times, 0.99)
    statistics.p_max_lpot = quantile(decode_times, 1.0)


def format_metrics(metrics: Dict[str, Metrics], statistics: Statistics) -> None:
    if statistics.request_number == 0:
        return

    input_tokens = [m.tokens_input for m in metrics.values()]
    output_tokens = [m.tokens_output for m in metrics.values()]
    first_token_costs = [m.first_token_cost for m in metrics.values()]
    last_token_costs = [m.last_token_cost for m in metrics.values()]
    decode_times = [
        dt
        for m in metrics.values()
        for dt in m.decode_time]
    total_infer_time = sum(m.ending_time - m.starting_time for m in metrics.values())
    input_tokens_sum = sum(input_tokens)
    output_tokens_sum = sum(output_tokens)
    first_token_costs_sum = sum(first_token_costs)
    last_token_costs_sum = sum(last_token_costs)

    if input_tokens_sum != 0:
        statistics.lpct = first_token_costs_sum / input_tokens_sum
    else:
        return

    calc_lpot(statistics, decode_times)

    statistics.latency_for_per_infer = total_infer_time / statistics.request_number
    statistics.average_input_length = input_tokens_sum / statistics.request_number
    statistics.average_output_length = output_tokens_sum / statistics.request_number
    statistics.average_first_token = first_token_costs_sum / statistics.request_number
    statistics.p99_first_token = quantile(first_token_costs, 0.99)
    statistics.max_first_token = max(first_token_costs)
    statistics.average_last_token = last_token_costs_sum / statistics.request_number
    statistics.p99_last_token = quantile(last_token_costs, 0.99)
    statistics.max_last_token = max(last_token_costs)
    statistics.max_decode = max(decode_times)

    statistics.qps = statistics.request_number / (statistics.latency_for_all / 1000.0)
    statistics.qps_per_npu = statistics.qps / statistics.tp


def print_statistics(statistics: Statistics) -> None:
    print_log(ENV.rank, logger.info, f"model_name:\t\t{statistics.model_full_name}")
    print_log(ENV.rank, logger.info, f"server_count:\t\t{statistics.server_count}")
    print_log(ENV.rank, logger.info, f"tp:\t\t\t{statistics.tp}")
    print_log(ENV.rank, logger.info, f"pp:\t\t\t{statistics.pp}")

    print_log(ENV.rank, logger.info, f"request_num:\t\t{statistics.request_number}")
    print_log(ENV.rank, logger.info, f"average_input_length:\t{statistics.average_input_length}")
    print_log(ENV.rank, logger.info, f"average_output_length:\t{statistics.average_output_length}")

    print_log(ENV.rank, logger.info, f"latency_for_all:\t{statistics.latency_for_all}ms")
    print_log(ENV.rank, logger.info, f"latency_for_per_infer:\t{statistics.latency_for_per_infer}ms")
    print_log(ENV.rank, logger.info, f"lpct:\t\t\t{statistics.lpct}ms")
    print_log(ENV.rank, logger.info, f"lpot:\t\t\t{statistics.lpot}ms")
    print_log(ENV.rank, logger.info, f"p10_lpot:\t\t{statistics.p10_lpot}ms")
    print_log(ENV.rank, logger.info, f"p20_lpot:\t\t{statistics.p20_lpot}ms")
    print_log(ENV.rank, logger.info, f"p30_lpot:\t\t{statistics.p30_lpot}ms")
    print_log(ENV.rank, logger.info, f"p40_lpot:\t\t{statistics.p40_lpot}ms")
    print_log(ENV.rank, logger.info, f"p50_lpot:\t\t{statistics.p50_lpot}ms")
    print_log(ENV.rank, logger.info, f"p60_lpot:\t\t{statistics.p60_lpot}ms")
    print_log(ENV.rank, logger.info, f"p80_lpot:\t\t{statistics.p80_lpot}ms")
    print_log(ENV.rank, logger.info, f"p90_lpot:\t\t{statistics.p90_lpot}ms")
    print_log(ENV.rank, logger.info, f"p95_lpot:\t\t{statistics.p95_lpot}ms")
    print_log(ENV.rank, logger.info, f"p99_lpot:\t\t{statistics.p99_lpot}ms")
    print_log(ENV.rank, logger.info, f"pmax_lpot:\t\t{statistics.p_max_lpot}ms")

    print_log(ENV.rank, logger.info, f"average_first_token:\t{statistics.average_first_token}ms")
    print_log(ENV.rank, logger.info, f"p99_first_token:\t{statistics.p99_first_token}ms")
    print_log(ENV.rank, logger.info, f"max_first_token:\t{statistics.max_first_token}ms")
    print_log(ENV.rank, logger.info, f"average_last_token:\t{statistics.average_last_token}ms")
    print_log(ENV.rank, logger.info, f"p99_last_token:\t\t{statistics.p99_last_token}ms")
    print_log(ENV.rank, logger.info, f"max_last_token:\t\t{statistics.max_last_token}ms")
    print_log(ENV.rank, logger.info, f"max_decode_time:\t{statistics.max_decode}ms")

    print_log(ENV.rank, logger.info, f"qps:\t\t\t{statistics.qps}/s")
    print_log(ENV.rank, logger.info, f"qpsPerNpu:\t\t{statistics.qps_per_npu}/s")
    print_log(ENV.rank, logger.info, f"generate speed\t\t{statistics.qps * statistics.average_output_length}")

    print_log(ENV.rank, logger.info, f"current time:\t\t{get_current_time()}")