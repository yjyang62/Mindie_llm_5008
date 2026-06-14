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

class SamplingParams:

    def __init__(self, in_temperature=1.0, in_top_k=0, in_top_p=1.0, in_typical_p=1.0, in_do_sample=False, \
                 in_seed=1, in_repetition_penalty=1.0, in_watermark=False, in_frequency_penalty=0.0, \
                 in_presence_penalty=0.0, logprobs=False, top_logprobs=None, best_of=None, n=1, \
                 use_beam_search=False):
        self.temperature = in_temperature
        self.top_k = in_top_k
        self.top_p = in_top_p
        self.typical_p = in_typical_p
        self.do_sample = in_do_sample
        self.seed = in_seed
        self.repetition_penalty = in_repetition_penalty
        self.watermark = in_watermark
        self.frequency_penalty = in_frequency_penalty
        self.presence_penalty = in_presence_penalty
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs
        self.best_of = best_of
        self.n = n
        self.use_beam_search = use_beam_search