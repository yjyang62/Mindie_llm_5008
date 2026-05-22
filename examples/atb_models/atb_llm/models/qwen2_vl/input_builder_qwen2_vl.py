# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import os
from typing import Dict, List

import numpy as np
import torch
from atb_llm.utils.shm_utils import decode_shape_from_int64
from atb_llm.utils.shm_utils import get_data_from_shm

from atb_llm.models.base.input_builder import InputBuilder
from atb_llm.models.qwen2_vl.data_preprocess_qwen2_vl import fetch_image, fetch_video, process_shared_memory
from atb_llm.utils.log.error_code import ErrorCode
from atb_llm.utils.log.logging import logger, print_log

IMAGE_START_ID = 151652
IMAGE_END_ID = 151653
IMAGE_TOKEN_ID = 151655
VIDEO_TOKEN_ID = 151656
IMAGE_FACTOR = 28
IMAGE_THW_TOKEN_OFFSET = 3
SECOND_PER_GRID_T_SHM_OFFSET = 4
SECOND_PER_GRID_T_SHAPE_OFFSET = 5
PYTORCH_TENSOR = "pt"
CONTENT = "content"
IMAGE = "image"
VIDEO = "video"


class Qwen2vlInputBuilder(InputBuilder):
    def __init__(self, tokenizer, image_processor, processor, config, **kwargs):
        self.config = config
        self.image_processor = image_processor
        self.processor = processor
        super().__init__(tokenizer, **kwargs)

    def generate_position_ids(self, input_ids):
        position_ids = np.arange(len(input_ids), dtype=np.int64)
        if np.any(np.equal(input_ids, IMAGE_START_ID)):
            bos_pos = np.where(np.equal(input_ids, IMAGE_START_ID))[0]
            eos_pos = np.where(np.equal(input_ids, IMAGE_END_ID))[0]
            vision_num = bos_pos.shape[0]
            deltas = 0
            for i in range(vision_num):
                thw_shape_value = input_ids[bos_pos[i] + 3]
                thw_shape = decode_shape_from_int64(thw_shape_value)

                vision_feature_len = eos_pos[i] - bos_pos[i] - 1
                t_shape = thw_shape[0]
                max_hw = max(thw_shape[1:])
                if self.config.model_type == "qwen2_5_vl":
                    tokens_per_second = self.config.vision_config.tokens_per_second
                    second_per_grid_t_shm_value = input_ids[bos_pos[i] + SECOND_PER_GRID_T_SHM_OFFSET]
                    second_per_grid_t_shape_value = input_ids[bos_pos[i] + SECOND_PER_GRID_T_SHAPE_OFFSET]
                    if second_per_grid_t_shm_value < 0:
                        second_per_grid_t_value = get_data_from_shm(
                            second_per_grid_t_shm_value,
                            second_per_grid_t_shape_value,
                            np.float32
                        )
                        max_tokens_t = int(second_per_grid_t_value[0][0] * tokens_per_second * (thw_shape[0] - 1))
                        t_shape = max_tokens_t
                if t_shape > (max_hw // 2):
                    deltas += vision_feature_len - t_shape
                else:
                    deltas += vision_feature_len - max_hw // 2
            position_ids[-1] = position_ids[-1] - deltas
        return position_ids
    
    def generate_position_ids_for_cloud(self, input_ids):
        position_ids = np.arange(len(input_ids), dtype=np.int64)
        if np.any(np.equal(input_ids, IMAGE_START_ID)):
            bos_pos = np.where(np.equal(input_ids, IMAGE_START_ID))[0]
            eos_pos = np.where(np.equal(input_ids, IMAGE_END_ID))[0]
            vision_num = bos_pos.shape[0]
            deltas = 0
            for i in range(vision_num):
                thw_shape_value = input_ids[bos_pos[i] + 3]
                thw_shape = decode_shape_from_int64(thw_shape_value)
                vision_feature_len = eos_pos[i] - bos_pos[i] - 1
                t_shape = thw_shape[0]
                max_hw = max(thw_shape[1:])
                if t_shape > (max_hw // 2):
                    deltas += vision_feature_len - t_shape
                else:
                    deltas += vision_feature_len - max_hw // 2
            position_ids[-1] = position_ids[-1] - deltas
        return position_ids

    def make_context(
            self,
            rank: int,
            conversation: List[Dict[str, List[Dict]]],
            **kwargs):
        if isinstance(conversation[0][CONTENT], str):
            for item in conversation:
                item[CONTENT] = [{"text": item[CONTENT]}]
        elif not isinstance(conversation[0][CONTENT], list):
            logger.error("The `conversation` \"content\" should be a List[Dict] or str.",
            ErrorCode.ATB_MODELS_PARAM_INVALID)
            raise TypeError("The `conversation` \"content\" should be a List[Dict] or str.")

        shm_name_save_path = kwargs.get('shm_name_save_path', None)
        context_tokens = self.apply_chat_template(
            conversation,
            shm_name_save_path=shm_name_save_path,
        )
        return context_tokens

    def apply_chat_template(
            self,
            conversation: List[Dict[str, List[Dict]]],
            shm_name_save_path: str = None,
            **kwargs):

        image_token_id = getattr(self.config, "image_token_id", IMAGE_TOKEN_ID)
        video_token_id = getattr(self.config, "video_token_id", VIDEO_TOKEN_ID)

        vision_info_list = []
        for single_conversation in conversation:
            for single_input in single_conversation[CONTENT]:
                if single_input.get(IMAGE, None):
                    if shm_name_save_path is None:
                        shm_name_save_dir = os.path.dirname(os.path.dirname(single_input[IMAGE]))
                        shm_name_save_path = os.path.join(shm_name_save_dir, "shm_name.txt")

                    images_inputs, feature_lens = fetch_image(self.image_processor, single_input)

                    shared_memory_result = process_shared_memory(
                        images_inputs.pixel_values,
                        shm_name_save_path,
                        images_inputs.image_grid_thw
                    )
                    vision_info_list.append([
                        shared_memory_result['pixel_values_shm_name'],
                        shared_memory_result['pixel_values_shape_value'],
                        shared_memory_result['thw_value'],
                        feature_lens,
                        image_token_id
                    ])

                elif single_input.get(VIDEO, None):
                    if shm_name_save_path is None:
                        shm_name_save_dir = os.path.dirname(os.path.dirname(single_input[VIDEO]))
                        shm_name_save_path = os.path.join(shm_name_save_dir, "shm_name.txt")

                    # 默认 fps 为 2
                    video_single_message = {"type": "video", "video": single_input[VIDEO], "fps": 2.0}
                    video_inputs, feature_lens, second_per_grid_t = fetch_video(
                        self.image_processor,
                        video_single_message,
                        IMAGE_FACTOR
                    )
                    shared_memory_result = process_shared_memory(
                        video_inputs.pixel_values_videos,
                        shm_name_save_path,
                        video_inputs.video_grid_thw,
                        second_per_grid_t
                    )
                    vision_info_list.append([
                        shared_memory_result['pixel_values_shm_name'],
                        shared_memory_result['pixel_values_shape_value'],
                        shared_memory_result['thw_value'],
                        shared_memory_result['second_per_grid_t_shm_name'],
                        shared_memory_result['second_per_grid_t_shape_value'],
                        feature_lens,
                        video_token_id
                    ])
        return self.process_token(vision_info_list, conversation)

    def process_token(self, vision_info_list, conversation):
        prompt = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )

        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].flatten()
        new_input_ids = input_ids
        vision_start_token_id = getattr(self.config, "vision_start_token_id", IMAGE_START_ID)
        vision_end_token_id = getattr(self.config, "vision_end_token_id", IMAGE_END_ID)
        image_token_id = getattr(self.config, "image_token_id", IMAGE_TOKEN_ID)
        video_token_id = getattr(self.config, "video_token_id", VIDEO_TOKEN_ID)

        bos_pos = torch.where(torch.eq(input_ids, vision_start_token_id))[0]
        eos_pos = torch.where(torch.eq(input_ids, vision_end_token_id))[0]

        vision_num = bos_pos.size(0)
        expand_token_ids = []
        pre = 0
        for i in range(0, vision_num, 1):
            feature_lens = vision_info_list[i][-2]
            text_token = input_ids[pre: bos_pos[i]]
            pre = eos_pos[i] + 1
            if vision_info_list[i][-1] == image_token_id:
                vision_pad_token = torch.cat(
                    [
                        torch.tensor([vision_start_token_id], dtype=input_ids.dtype),
                        torch.tensor([vision_info_list[i][0]], dtype=input_ids.dtype),
                        torch.tensor([vision_info_list[i][1]], dtype=input_ids.dtype),
                        torch.tensor([vision_info_list[i][2]], dtype=input_ids.dtype),
                        torch.full((feature_lens - 3,), vision_info_list[i][-1], dtype=input_ids.dtype),
                        torch.tensor([vision_end_token_id], dtype=input_ids.dtype),
                    ]
                )
            elif vision_info_list[i][-1] == video_token_id:
                vision_pad_token = torch.cat(
                    [
                        torch.tensor([vision_start_token_id], dtype=input_ids.dtype),
                        torch.tensor([vision_info_list[i][0]], dtype=input_ids.dtype),
                        torch.tensor([vision_info_list[i][1]], dtype=input_ids.dtype),
                        torch.tensor([vision_info_list[i][2]], dtype=input_ids.dtype),
                        torch.tensor([vision_info_list[i][3]], dtype=input_ids.dtype),
                        torch.tensor([vision_info_list[i][4]], dtype=input_ids.dtype),
                        torch.full((feature_lens - 5,), vision_info_list[i][-1], dtype=input_ids.dtype),
                        torch.tensor([vision_end_token_id], dtype=input_ids.dtype),
                    ]
                )

            if text_token.size(0) != 0:
                expand_token_ids.append(text_token)

            if vision_pad_token.size(0) != 0:
                expand_token_ids.append(vision_pad_token)

        text_token = input_ids[pre:]
        if text_token.size(0) != 0:
            expand_token_ids.append(text_token)

        if expand_token_ids:
            new_input_ids = torch.cat(expand_token_ids)
        return new_input_ids