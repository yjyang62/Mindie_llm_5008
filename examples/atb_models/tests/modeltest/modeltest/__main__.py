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

import argparse
import os
from modeltest.api.runner import get_runner_cls, RunnerConfig
from atb_llm.utils.log.logging import logger


def setup_parser():
    parser = argparse.ArgumentParser(description="ModelTest supported arguments")
    parser.add_argument(
        "--model_config_path",
        type=str,
        required=True,
        help="Path to model config yaml"
    )
    parser.add_argument(
        "--task_config_path",
        type=str,
        required=True,
        help="Path to task config yaml"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batchsize"
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Tensor Parallel Num"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Specify the test result output dir"
    )
    parser.add_argument(
        "--lcoc_disable",
        action="store_true",
        help="Whether to turn off the loc, ATB_LLM_LCOC_ENABLE=0.",
    )
    parser.add_argument(
        "--save_debug_enable",
        action="store_true",
        help="Whether to save debug csv"
    )
    return parser


def cli_evaluate():
    parser = setup_parser()
    args = parser.parse_args()
    output_dir = (os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs") 
                  if args.output_dir is None 
                  else args.output_dir)

    if args.lcoc_disable:
        os.environ["ATB_LLM_LCOC_ENABLE"] = "0"
    else:
        os.environ["ATB_LLM_LCOC_ENABLE"] = "1"
    
    runner_config = RunnerConfig(args.model_config_path,
                                 args.task_config_path,
                                 args.tp,
                                 args.batch_size,
                                 args.save_debug_enable)
    logger.info(f"runner config:\n{runner_config}")
    runner = get_runner_cls()(runner_config, output_dir)
    runner.start()


if __name__ == "__main__":
    cli_evaluate()