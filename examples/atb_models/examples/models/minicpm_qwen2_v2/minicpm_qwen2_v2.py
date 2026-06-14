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

import math
import os
from transformers import AutoProcessor

from atb_llm.utils.env import ENV
from atb_llm.utils import argument_utils
from atb_llm.utils.cpu_binding import NpuHbmInfo
from atb_llm.utils.log import logger, print_log
from atb_llm.utils.file_utils import safe_listdir, standardize_path, check_file_safety
from atb_llm.utils.multimodal_utils import MultimodalInput, MultimodalRequestOut
from atb_llm.models.base.model_utils import safe_from_pretrained
from atb_llm.utils.log.error_code import ErrorCode
from atb_llm.utils.multimodal_utils import is_image_path, is_video_path
from examples.multimodal_runner import MultimodalPARunner, parser
from examples.multimodal_runner import path_validator, num_validator
from examples.run_pa import parse_ids
from examples.server.request import MultiModalRequest, MultiModalReqParams
from examples.server.cache import CacheManager, CacheConfig
from examples.server.generate import generate_req


STORE_TRUE = "store_true"
PERF_FILE = "./minicpm_qwen2_v2_performance.csv"
PERF_COLUMNS = "batch, input_len, output_len, embedding_len, first_token_time(ms), \
                non_first_token_time(ms), ResponseTime(ms),E2E Throughput Average(Tokens/s)\n"
PRED_FILE = "./predict_result.json"


class MinicpmQwen2V2Runner(MultimodalPARunner):
    def __init__(self, **kwargs):
        self.processor = None
        super().__init__(**kwargs)
        
    def init_processor(self):
        try:
            self.processor = safe_from_pretrained(AutoProcessor, self.model_path, 
                                                  trust_remote_code=self.trust_remote_code)
        except AssertionError:
            self.processor = self.model.tokenizer
            
    def precision_save(self, precision_inputs, **kwargs):
        all_input_texts = precision_inputs.all_input_texts
        all_generate_text_list = precision_inputs.all_generate_text_list
        image_file_list = precision_inputs.image_file_list
        image_answer_pairs = {}
        for text_index in range(len(all_input_texts)):
            image_answer_pairs[image_file_list[text_index]] = all_generate_text_list[text_index]
            image_answer_pairs = dict(sorted(image_answer_pairs.items()))
        super().precision_save(precision_inputs, answer_pairs=image_answer_pairs)

    def warm_up(self):
        if self.max_prefill_tokens == -1:
            self.max_prefill_tokens = self.max_batch_size * (self.max_input_length + self.max_output_length)
        print_log(self.rank, logger.info, "---------------begin warm_up---------------")
        try:
            self.warm_up_num_blocks = math.ceil((self.max_input_length + self.max_output_length) /
                                                self.block_size) * self.max_batch_size
        except ZeroDivisionError as e:
            raise ZeroDivisionError from e
        cache_config = CacheConfig(self.warm_up_num_blocks, self.block_size)
        self.cache_manager = CacheManager(cache_config, self.model_config)
        req_list = self.prepare_request(self.warm_up_inputs, 1, self.max_output_length, 0).req_list
        self.model.postprocessor.max_new_tokens = 2
        generate_req(req_list, self.model, self.max_batch_size, self.max_prefill_tokens, self.cache_manager)
        self.warm_up_memory = int(
            self.max_memory * NpuHbmInfo.get_hbm_usage(self.local_rank, self.world_size, self.model.soc_info.need_nz))
        print_log(self.rank, logger.info, f'warmup_memory(GB): {self.warm_up_memory / (1024 ** 3): .2f}')
        print_log(self.rank, logger.info, "---------------end warm_up---------------")

    def infer(self, mm_inputs, batch_size, max_output_length, ignore_eos, max_iters=None, **kwargs):
        input_texts = mm_inputs.input_texts
        image_path_list = mm_inputs.image_path
        video_path_list = mm_inputs.video_path
        path_list = image_path_list if image_path_list else video_path_list
        if len(input_texts) != len(path_list):
            raise RuntimeError("input_text length must equal input_images length")
        if not ENV.profiling_enable:
            if self.max_batch_size > 0:
                max_iters = math.ceil(len(path_list) / self.max_batch_size)
            else:
                raise RuntimeError("f{self.max_batch_size} max_batch_size should > 0, please check")
        return super().infer(mm_inputs, batch_size, max_output_length, ignore_eos, max_iters=max_iters)
    
    def prepare_request(self, multimodalinputs, batch_size, max_output_length, current_iter):
        input_texts = multimodalinputs.input_texts
        image_path = multimodalinputs.image_path
        video_path = multimodalinputs.video_path
        audio_path = multimodalinputs.audio_path
        multimodalparams_list = [MultiModalReqParams(
                                        input_texts[current_iter * batch_size
                                                        + single_batch] if input_texts else None,
                                        image_path[current_iter * batch_size
                                                        + single_batch] if image_path else None,
                                        video_path[current_iter * batch_size
                                                        + single_batch] if video_path else None,
                                        audio_path[current_iter * batch_size
                                                        + single_batch] if audio_path else None,
                                        max_output_length,
                                        self.block_size,
                                        req_idx=single_batch) for single_batch in range(batch_size)]
        
        prefill_inputs_embeds = self.model.model.prepare_prefill_token(multimodalparams_list, self.processor)
        req_list = [MultiModalRequest(max_output_length, self.block_size, i, 
                                      prefill_inputs_embeds[i], self.adapter_id) for i in range(batch_size)]

        print_log(self.rank, logger.debug, f'req_list[0].input_ids: {req_list[0].input_ids}')
        
        return MultimodalRequestOut(req_list, batch_size, image_path, video_path, audio_path, input_texts)
    
    
def parse_arguments():
    parser_minicpmqwen = parser
    string_validator = argument_utils.StringArgumentValidator(min_length=0, max_length=1000)
    list_num_validator = argument_utils.ListArgumentValidator(num_validator, 
                                                              max_length=1000, 
                                                              allow_none=True)
    texts_validator_dict_element = argument_utils.DictionaryArgumentValidator({'role': string_validator, 
                                                                               'content': string_validator})
    texts_validator_dict = argument_utils.ListArgumentValidator(texts_validator_dict_element)
    texts_validator_str = argument_utils.ListArgumentValidator(string_validator)
    texts_validator = {str: texts_validator_str, dict: texts_validator_dict}

    parser_minicpmqwen.add_argument('--image_or_video_path',
                        help="image_or_video path",
                        default="/data/acltransformer_testdata/minicpm_qwen2_v2",
                        validator=path_validator,
                        )
    parser_minicpmqwen.add_argument(
        '--input_texts',
        type=str,
        nargs='+',
        default=[{'role': 'user', 'content': 'Describe this video in less than 20 words.'}],
        validator=texts_validator)
    parser_minicpmqwen.add_argument(
        '--input_ids',
        type=parse_ids,
        nargs='+',
        default=None,
        validator=list_num_validator)
    return parser_minicpmqwen.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    image_or_video_path = standardize_path(args.image_or_video_path)
    check_file_safety(image_or_video_path, 'r')
    file_name = safe_listdir(image_or_video_path)
    input_dict = {
        'rank': rank,
        'world_size': world_size,
        'local_rank': local_rank,
        'perf_file': PERF_FILE,
        'pred_file': PRED_FILE,
        **vars(args)
    }
    if is_image_path(image_or_video_path):
        image_path = [os.path.join(image_or_video_path, f) for f in file_name]
        video_path = None
        input_dict['image_path'] = image_path
        texts = args.input_texts
        image_length = len(image_path)
        if len(texts) != image_length:
            texts.extend([texts[-1]] * (image_length - len(texts)))
    elif is_video_path(image_or_video_path):
        video_path = [os.path.join(image_or_video_path, f) for f in file_name]
        image_path = None
        input_dict['video_path'] = video_path
        texts = args.input_texts
        video_length = len(video_path)
        if len(texts) != video_length:
            texts.extend([texts[-1]] * (video_length - len(texts)))
    else:
        logger.error("Unsupported media type, the input files should be videos of format {mp4, wmv, avi}"
                    " or images of format{jpeg, bmp, png, jpg}, please check your input.",
                             ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
        raise KeyError("Unsupported media type, the input files should be videos of format {mp4, wmv, avi}"
                    " or images of format{jpeg, bmp, png, jpg}, please check your input.")
    
    input_dict['input_texts'] = texts
    pa_runner = MinicpmQwen2V2Runner(**input_dict)

    if image_path:
        image_length = len(image_path)
        remainder = image_length % args.max_batch_size
        if remainder != 0:
            num_to_add = args.max_batch_size - remainder
            image_path.extend([image_path[-1]] * num_to_add)
            texts.extend([texts[-1]] * num_to_add)
    elif video_path:
        video_length = len(video_path)
        remainder = video_length % args.max_batch_size
        if remainder != 0:
            num_to_add = args.max_batch_size - remainder
            video_path.extend([video_path[-1]] * num_to_add)
            texts.extend([texts[-1]] * num_to_add)

    print_log(rank, logger.info, f'pa_runner: {pa_runner}')

    infer_params = {
        "mm_inputs": MultimodalInput(texts,
                                image_path,
                                video_path,
                                None),
        "batch_size": args.max_batch_size,
        "max_output_length": args.max_output_length,
        "ignore_eos": args.ignore_eos,
    }
    pa_runner.warm_up()
    generate_texts, token_nums, latency = pa_runner.infer(**infer_params)
    
    for i, generate_text in enumerate(generate_texts):
        print_log(rank, logger.info, f'Answer[{i}]: {generate_text}')
        print_log(rank, logger.info, f'Generate[{i}] token num: {token_nums[i]}')
        print_log(rank, logger.info, f"Latency: {latency}")