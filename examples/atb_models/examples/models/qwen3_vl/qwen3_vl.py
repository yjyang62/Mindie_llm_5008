# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import os
import math
from functools import wraps
import time
import torch
import torch.nn.functional as F
import torch_npu
import numpy
from transformers import AutoProcessor
from atb_llm.utils import argument_utils
from atb_llm.utils.env import ENV
from atb_llm.utils.cpu_binding import NpuHbmInfo
from atb_llm.utils.log import logger, print_log
from atb_llm.utils.file_utils import safe_listdir, standardize_path, check_file_safety, is_path_exists
from atb_llm.utils.multimodal_utils import is_image, is_video
from atb_llm.utils.shm_utils import release_shared_memory
from atb_llm.runner.tokenizer_wrapper import TokenizerWrapper
from atb_llm.models.base.model_utils import safe_from_pretrained
from examples.multimodal_runner import MultimodalPARunner, parser
from examples.multimodal_runner import path_validator
from examples.server.cache import CacheManager, CacheConfig
from examples.server.generate import decode_token, generate_req
from examples.server.request import MultiModalRequest


_SHM_PATH = "./shm_name.txt"
IMAGE = "image"
VIDEO = "video"
TEXT = "text"


def profiling_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not ENV.profiling_enable:
            result = func(*args, **kwargs)
        else:
            profiling_path = ENV.profiling_filepath
            if not os.path.exists(profiling_path):
                os.makedirs(profiling_path, exist_ok=True)
            experimental_config = torch_npu.profiler._ExperimentalConfig(
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                l2_cache=False,
                data_simplification=False,
            )
            with torch_npu.profiler.profile(
                    activities=[
                        torch_npu.profiler.ProfilerActivity.CPU,
                        torch_npu.profiler.ProfilerActivity.NPU,
                    ],
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                        profiling_path
                    ),
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=False,
                    with_flops=False,
                    with_modules=False,
                    experimental_config=experimental_config,
            ) as _:
                result = func(*args, **kwargs)
        return result
    return wrapper


class Qwen3vlPARunner(MultimodalPARunner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.processor = None
        self.tokenizer_wrapper = TokenizerWrapper(self.model_path)
        self.spatial_merge_size = self.model.config.vision_config.spatial_merge_size
        self.image_token_id = self.model.config.image_token_id
        self.video_token_id = self.model.config.video_token_id
        self.vision_start_token_id = self.model.config.vision_start_token_id
        self.vision_end_token_id = self.model.config.vision_end_token_id
    
    def init_processor(self):
        try:
            self.processor = safe_from_pretrained(AutoProcessor, self.model_path)
        except AssertionError as e:
            raise RuntimeError("Loading Qwen3_vl Processor failed.") from e
    
    def warm_up(self):
        warm_up_image = self.warm_up_inputs.image_path[0]
        if is_video(warm_up_image):
            key = "video"
        elif is_image(warm_up_image):
            key = "image"
        else:
            raise TypeError("The multimodal input field currently only supports 'image' and 'video'.")
        warm_up_text = "describe this image"
        inputs = [
            {
                key: warm_up_image,
                "text": warm_up_text
            }
        ]
        input_ids = self.tokenizer_wrapper.tokenize(inputs, shm_name_save_path=_SHM_PATH)
        if len(input_ids) < self.max_input_length:
            input_ids = F.pad(input_ids, [0, self.max_input_length - len(input_ids)])
        else:
            raise RuntimeError(
                f"The input sequence length: {len(input_ids)} exceeds the max_input_length: {self.max_input_length}.")
        print_log(self.rank, logger.info, "---------------Begin warm_up---------------")
        try:
            self.warm_up_num_blocks = math.ceil((self.max_input_length + self.max_output_length) /
                                                self.block_size) * self.max_batch_size
        except ZeroDivisionError as e:
            raise ZeroDivisionError(f"block_size: {self.block_size} cannot be zero") from e
        cache_config = CacheConfig(self.warm_up_num_blocks, self.block_size)
        self.cache_manager = CacheManager(cache_config, self.model_config)
        max_output_length = 2
        self.model.postprocessor.max_new_tokens = max_output_length
        if self.max_prefill_tokens == -1:
            self.max_prefill_tokens = self.max_batch_size * (self.max_input_length + self.max_output_length)
        single_req = self._request_from_token_qwen3_vl(
            input_ids,
            max_output_length,
            self.block_size,
            req_idx=0,
        )
        generate_req([single_req], self.model, self.max_batch_size, self.max_prefill_tokens, self.cache_manager)
        self.warm_up_memory = int(
            self.max_memory
            * NpuHbmInfo.get_hbm_usage(
                self.local_rank, self.world_size, self.model.soc_info.need_nz
            )
        )
        print_log(
            self.rank,
            logger.info,
            f"warmup_memory(GB): {self.warm_up_memory / (1024 ** 3): .2f}",
        )
        print_log(self.rank, logger.info, "---------------End warm_up---------------")

    def infer(self, mm_inputs, max_output_length, **kwargs):
        self.make_cache_manager()
        self.model.postprocessor.max_new_tokens = max_output_length
        return self._generate(mm_inputs, max_output_length)
    
    @profiling_decorator
    def _generate(self, mm_inputs, max_output_length):
        req_list = []
        torch.npu.synchronize()
        e2e_start = time.time()
        for i, mm_input in enumerate(mm_inputs):
            input_ids = self.tokenizer_wrapper.tokenize(mm_input, shm_name_save_path=_SHM_PATH)
            single_req = self._request_from_token_qwen3_vl(
                input_ids,
                max_output_length,
                self.block_size,
                req_idx=i,
            )
            req_list.append(single_req)
        generate_req(
            req_list,
            self.model,
            self.max_batch_size,
            self.max_prefill_tokens,
            self.cache_manager,
        )
        generate_text_list, token_num_list = decode_token(req_list, self.tokenizer, skip_special_tokens=True)
        torch.npu.synchronize()
        e2e_end = time.time()
        e2e_time = e2e_end - e2e_start
        return generate_text_list, token_num_list, e2e_time

    def _request_from_token_qwen3_vl(self, input_ids, max_out_length, block_size, req_idx=0):
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.int64)
        input_tokens = input_ids.numpy()
        position_ids = self.tokenizer_wrapper.input_builder.generate_position_ids(input_tokens)
        position_ids = torch.tensor(position_ids, dtype=torch.int64)
        request = MultiModalRequest(
            max_out_length,
            block_size,
            req_idx,
            input_ids,
            adapter_id=None,
            position_ids=position_ids
        )
        return request
    

def parse_arguments():
    string_validator = argument_utils.StringArgumentValidator(min_length=0, max_length=1000)
    list_str_validator = argument_utils.ListArgumentValidator(string_validator, max_length=1000)
    qwen3vl_parser = parser
    qwen3vl_parser.add_argument('--image_path',
        help="image path",
        required=True,
        validator=path_validator,
        )
    qwen3vl_parser.add_argument(
        '--input_texts_for_image',
        type=str,
        nargs='+',
        default=["describe this image"],
        validator=list_str_validator
        )
    return qwen3vl_parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    rank = ENV.rank
    local_rank = ENV.local_rank
    world_size = ENV.world_size
    image_path = standardize_path(args.image_path)
    check_file_safety(image_path, 'r')
    file_name = safe_listdir(image_path)
    image_path = [os.path.join(image_path, f) for f in file_name]
    texts = args.input_texts_for_image
    image_length = len(image_path)
    if len(texts) != image_length:
        texts.extend([texts[-1]] * (image_length - len(texts)))
    input_dict = {
        'rank': rank,
        'world_size': world_size,
        'local_rank': local_rank,
        **vars(args)
    }
    input_dict['image_path'] = image_path
    input_dict['input_texts'] = texts

    pa_runner = Qwen3vlPARunner(**input_dict)

    remainder = image_length % args.max_batch_size
    if remainder != 0:
        num_to_add = args.max_batch_size - remainder
        image_path.extend([image_path[-1]] * num_to_add)
        texts.extend([texts[-1]] * num_to_add)
    
    mm_inputs = []
    for i, fname in enumerate(image_path):
        if is_video(fname):
            key = VIDEO
        elif is_image(fname):
            key = IMAGE
        else:
            raise TypeError("The multimodal input field currently only supports 'image' and 'video'.")
        mm_inputs.append([{
            key: fname,
            TEXT: texts[i]
        }])
    infer_params = {
        "mm_inputs": mm_inputs,
        "batch_size": args.max_batch_size,
        "max_output_length": args.max_output_length
    }
    pa_runner.warm_up()
    generate_texts, token_nums, e2e_time_gen = pa_runner.infer(**infer_params)
    
    length = len(args.input_texts_for_image)
    for i, generate_text in enumerate(generate_texts):
        if i < length:
            print_log(rank, logger.info, f'Question[{i}]: {args.input_texts_for_image[i]}', need_filter=True)
            print_log(rank, logger.info, f'Answer[{i}]: {generate_text}')
            print_log(rank, logger.info, f'Generate[{i}] token num: {token_nums[i]}')

    if rank == 0 and is_path_exists(_SHM_PATH):
        try:
            release_shared_memory(_SHM_PATH)
        except Exception as e:
            print_log(rank, logger.error, f"Release shared memory failed: {e}")
        try:
            os.remove(_SHM_PATH)
        except Exception as e:
            print_log(rank, logger.error, f"Remove shared memory file failed: {e}")