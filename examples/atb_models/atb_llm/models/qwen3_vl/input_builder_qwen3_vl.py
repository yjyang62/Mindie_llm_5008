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
import copy
from typing import Dict, List
import numpy as np
import torch
import PIL

from atb_llm.utils.shm_utils import process_shared_memory, get_data_from_shm
from atb_llm.utils.multimodal_utils import safe_open_image, check_video_path
from ..base.input_builder import InputBuilder

CONTENT = "content"
IMAGE = "image"
VIDEO = "video"
TEXT = "text"
TYPE = "type"
URL = "url"

template = {
        "role": "",
        "content": [],
    }


class Qwen3vlInputBuilder(InputBuilder):
    def __init__(self, tokenizer, config, processor, **kwargs):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        super().__init__(tokenizer, **kwargs)
        self.spatial_merge_size = self.config.vision_config.spatial_merge_size
        self.image_token_id = self.config.image_token_id
        self.video_token_id = self.config.video_token_id
        self.vision_start_token_id = self.config.vision_start_token_id
        self.vision_end_token_id = self.config.vision_end_token_id
    
    @staticmethod
    def get_shm_name_save_path(file_path: str):
        """
        Generates a file path for storing shared memory names by navigating up two directory levels.

        Args:
            file_path: The original file path from which to derive the shared memory save path
        Returns:
            str: A full file path pointing to 'shm_name.txt' located two directory levels
                above the input file_path 
        Example:
            If file_path is '/project/data/inputs/current/file.jpg'
            Returns: '/project/data/shm_name.txt'
        """
        shm_name_save_dir = os.path.dirname(os.path.dirname(file_path))
        shm_name_save_path = os.path.join(shm_name_save_dir, "shm_name.txt")
        return shm_name_save_path

    def generate_position_ids(self, input_ids):
        position_ids = np.arange(len(input_ids), dtype=np.int64)
        if np.any(np.equal(input_ids, self.vision_start_token_id)):
            image_grid_thw, video_grid_thw = self._parse_inputs_ids_with_shm(input_ids)
            position_delta = self._compute_llm_pos_delta(input_ids, image_grid_thw, video_grid_thw)
            position_ids[-1] = position_ids[-1] + position_delta
        return position_ids

    def make_context(
        self,
        rank: int,
        conversation: List[Dict[str, List[Dict]]],
        **kwargs
    ):
        """Build input context from multimodal conversation."""
        # Normalize content format: convert string to list format
        if isinstance(conversation[0][CONTENT], str):
            for item in conversation:
                item[CONTENT] = [{TEXT: item[CONTENT]}]
        elif not isinstance(conversation[0][CONTENT], list):
            raise TypeError("The conversation \"content\" should be a List[Dict] or str.")
        
        shm_name_save_path = kwargs.get('shm_name_save_path', None)
        conversation_list = []
        
        # Process each conversation turn: extract images, videos, and text
        for single_conversation in conversation:
            message_list = []
            for item in single_conversation[CONTENT]:
                if item.get(IMAGE, None):
                    img_fname = item[IMAGE]
                    img = safe_open_image(PIL.Image, img_fname)
                    message_list.append({
                        TYPE: IMAGE,
                        IMAGE: img
                    })
                    if shm_name_save_path is None:
                        shm_name_save_path = self.get_shm_name_save_path(img_fname)
                if item.get(VIDEO, None):
                    video_path = check_video_path(item[VIDEO])
                    message_list.append({
                        TYPE: VIDEO,
                        URL: video_path
                    })
                    if shm_name_save_path is None:
                        shm_name_save_path = self.get_shm_name_save_path(video_path)
                if item.get(TEXT, None):
                    text = item[TEXT]
                    message_list.append({
                        TYPE: TEXT,
                        TEXT: text
                    })
            conversation_list.append({"role": single_conversation["role"], CONTENT: copy.deepcopy(message_list)})
        
        # Apply chat template and tokenize
        inputs = self.processor.apply_chat_template(
            conversation_list,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors='pt'
        )
        
        # Process shared memory and update token IDs with SHM metadata
        shm_info = process_shared_memory(inputs, shm_name_save_path)
        input_ids = self.update_token_id(inputs, shm_info)
        return input_ids
        
    def update_token_id(self, inputs: Dict[str, torch.Tensor], shm_info: Dict):
        """
        Replaces special image tokens in the input sequence with shared memory metadata.

        This method identifies the position between image start/end tokens and replaces
        the subsequent tokens with encoded shared memory information (names and shapes)
        for both image and video data. This allows the model to reference external
        memory segments containing large tensor data instead of carrying the full data
        in the token sequence.

        Args:
            inputs: Dictionary containing the input token sequence with key 'input_ids'
            shm_info: Dictionary containing encoded shared memory metadata from 
                    process_shared_memory(), including shm names and shape values
                    for pixel values and grid dimensions
        Returns:
            Modified input_ids tensor with shared memory references inserted between
            the image start and end token positions.
        Raises:
            RuntimeError: If insufficient space exists between BOI and EOI tokens for
                    all required metadata entries (8 positions needed)
        """
        input_ids = inputs['input_ids'].flatten()
        has_image = torch.any(torch.eq(input_ids, self.image_token_id))
        has_video = torch.any(torch.eq(input_ids, self.video_token_id))
        if not has_image and not has_video:
            return input_ids
        # begin of inserting shm_info position is vision_start_token_id position + 2
        # vision_start_token_id position + 1 is image_token_id or video_token_id reserved for recovery
        boi_pos = torch.where(torch.eq(input_ids, self.vision_start_token_id))[0][0].item() + 1
        eoi_pos = torch.where(torch.eq(input_ids, self.vision_end_token_id))[0][0].item()
        if eoi_pos - boi_pos - 1 < len(shm_info) + 1:
            msg = "Load share memory info to input ids failed."
            raise RuntimeError(msg)
        input_ids[boi_pos + 1] = shm_info['pixel_values_shm_name']
        input_ids[boi_pos + 2] = shm_info['pixel_values_shape_value']
        input_ids[boi_pos + 3] = shm_info['image_grid_thw_shm_name']
        input_ids[boi_pos + 4] = shm_info['image_grid_thw_shape_value']
        input_ids[boi_pos + 5] = shm_info['pixel_values_videos_shm_name']
        input_ids[boi_pos + 6] = shm_info['pixel_values_videos_shape_value']
        input_ids[boi_pos + 7] = shm_info['video_grid_thw_shm_name']
        input_ids[boi_pos + 8] = shm_info['video_grid_thw_shape_value']
        return input_ids
    
    def _parse_inputs_ids_with_shm(self, input_ids):
        boi_pos = np.where(np.equal(input_ids, self.vision_start_token_id))[0][0].item() + 1
        image_grid_thw_shm_name = input_ids[boi_pos + 3]
        image_grid_thw_shape_value = input_ids[boi_pos + 4]
        video_grid_thw_shm_name = input_ids[boi_pos + 7]
        video_grid_thw_shape_value = input_ids[boi_pos + 8]
        image_grid_thw = get_data_from_shm(image_grid_thw_shm_name, image_grid_thw_shape_value, np.int32) \
                if image_grid_thw_shm_name else None
        video_grid_thw = get_data_from_shm(video_grid_thw_shm_name, video_grid_thw_shape_value, np.int32) \
                if video_grid_thw_shm_name else None
        return image_grid_thw, video_grid_thw
    
    def _compute_llm_pos_delta(self, input_ids, image_grid_thw, video_grid_thw):
        if video_grid_thw is not None:
            indices = np.repeat(np.arange(video_grid_thw.shape[0]), video_grid_thw[:, 0])
            video_grid_thw = np.take(video_grid_thw, indices, axis=0)
            video_grid_thw[:, 0] = 1
        image_nums, video_nums = 0, 0
        image_index, video_index = 0, 0
        vision_start_indices = np.argwhere(input_ids == self.vision_start_token_id).squeeze(1)
        vision_tokens = input_ids[vision_start_indices + 1]
        image_nums = (vision_tokens == self.image_token_id).sum()
        video_nums = (vision_tokens == self.video_token_id).sum()
        input_tokens = input_ids.tolist()
        llm_pos_ids_offset = 0
        st = 0
        remain_images, remain_videos = image_nums, video_nums
        for _ in range(image_nums + video_nums):
            if self.image_token_id in input_tokens and remain_images > 0:
                ed_image = input_tokens.index(self.image_token_id, st)
            else:
                ed_image = len(input_tokens) + 1
            if self.video_token_id in input_tokens and remain_videos > 0:
                ed_video = input_tokens.index(self.video_token_id, st)
            else:
                ed_video = len(input_tokens) + 1
            if ed_image < ed_video:
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    image_grid_thw[image_index][0].item(),
                    image_grid_thw[image_index][1].item() // self.spatial_merge_size,
                    image_grid_thw[image_index][2].item() // self.spatial_merge_size,
                )
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    video_grid_thw[video_index][0].item(),
                    video_grid_thw[video_index][1].item() // self.spatial_merge_size,
                    video_grid_thw[video_index][2].item() // self.spatial_merge_size,
                )
                video_index += 1
                remain_videos -= 1
                ed = ed_video
            text_len = ed - st
            llm_pos_ids_offset += text_len
            llm_pos_ids_offset += max(llm_grid_h, llm_grid_w)
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w
        if st < len(input_tokens):
            text_len = len(input_tokens) - st
            llm_pos_ids_offset += text_len
        position_delta = llm_pos_ids_offset - len(input_tokens)
        return position_delta