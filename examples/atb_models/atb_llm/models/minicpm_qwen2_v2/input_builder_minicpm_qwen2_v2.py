# Copyright 2023 the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import copy
import os
from typing import Dict, List
import av
import torch
import numpy as np

from PIL import Image

from atb_llm.models.base.input_builder import InputBuilder
from atb_llm.utils import multimodal_utils
from atb_llm.utils.log.error_code import ErrorCode
from atb_llm.utils.log.logging import logger
from atb_llm.utils.multimodal_utils import safe_open_image
from atb_llm.utils.shm_utils import encode_shm_name_to_int64, encode_shape_to_int64, create_shm

MSG_CONTENT = 'content'
MSG_ROLE = 'role'
MSG_ROLE_SYSTEM = 'system'
MSG_ROLE_USER = 'user'
MSG_ROLE_ASSISTATN = 'assistant'
IMAGE_BOUND = 'image_bound'
TGT_SIZES = 'tgt_sizes'
PIXEL_VALUES = 'pixel_values'
INPUT_IDS = 'input_ids'
IMAGE_BOUND = 'image_bound'
SHM_FILE_NAME = 'shm_name.txt'
IMAGE_PATTERN = '(<image>./</image>)'
IMAGE_MARK = 'image'
VIDEO_MARK = 'video'
TEXT_MARK = 'text'
GENERAL_TEMPLATE = [{"role": "user", "content": ""}]
MAX_NUM_FRAMES = 64
VISION_START_TOKEN_ID = 151660
VISION_END_TOKEN_ID = 151661


def uniform_sample(frame_idx, n):
    gap = len(frame_idx) / n
    idxs = [int(i * gap + gap / 2) for i in range(n)]
    return [frame_idx[i] for i in idxs]


def encode_video(video_path):
    try:
        # 打开视频并获取视频流
        container = multimodal_utils.safe_load_multimodal_source(av.open, video_path)
        video_stream = next(s for s in container.streams if s.type == 'video')
    except Exception as e:
        logger.error(f'Read video error:{e}.',
                     ErrorCode.ATB_MODELS_EXECUTION_FAILURE)
        raise RuntimeError(f'Read video error:{e}.') from e

    frames = []
    for frame in container.decode(video_stream):
        frame_img = frame.to_image()
        frames.append(frame_img)

    sample_fps = round(video_stream.average_rate)
    frame_idx = [i for i in range(0, len(frames), sample_fps)]
    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    sampled_frames = [frames[i] for i in frame_idx]

    return sampled_frames


def from_list_format(list_format: List[Dict]):
    text_list, image_list, image_path_list = [], [], []
    for ele in list_format:
        if IMAGE_MARK in ele:
            img = safe_open_image(Image, ele[IMAGE_MARK])
            image_list.append(img)
            image_path_list.append(ele[IMAGE_MARK])
            text_list.append(IMAGE_PATTERN)
        if VIDEO_MARK in ele:
            frames = encode_video(ele[VIDEO_MARK])
            image_list.extend(frames)
            image_path_list.append(ele[VIDEO_MARK])
            text_list.append((IMAGE_PATTERN + '\n') * len(frames))
        if TEXT_MARK in ele:
            text = ele[TEXT_MARK]
            text_list.append(text)
    return text_list, image_list, image_path_list


def process_shm(image_pixel, shm_name_save_path, dtype=np.float32):
    shm = create_shm(image_pixel.nbytes, shm_name_save_path)
    shared_array = np.ndarray(image_pixel.shape, dtype=dtype, buffer=shm.buf)
    shared_array[:] = image_pixel
    shm_name = encode_shm_name_to_int64(shm.name)
    shape_value = encode_shape_to_int64(image_pixel.shape)
    return shm_name, shape_value


def process_prompts(msgs, image_list, image_path_list, config, processor):
    prompt_list = []
    prompt = processor.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    if len(image_list) == 0:
        return processor.tokenizer.encode(prompt), image_path_list
    
    prompt_list.append(prompt)

    data = processor(
        text=prompt_list,
        images=image_list,
        max_slice_nums=config.slice_config.max_slice_nums,
        use_image_id=processor.image_processor.use_image_id,
        return_tensors="pt")
    for image in image_list:
        image.close()
    
    return data, image_path_list


def apply_general_template(inputs, config, processor):
    text_list, image_list, image_path_list = from_list_format(inputs)
    msgs = copy.deepcopy(GENERAL_TEMPLATE)
    msgs[0][MSG_CONTENT] = "\n".join(text_list)
    
    return process_prompts(msgs, image_list, image_path_list, config, processor)


def apply_chat_template(inputs, config, processor):
    image_list, image_path_list = [], []
    msgs = copy.deepcopy(inputs)

    for msg in msgs:
        role = msg[MSG_ROLE]
        content = msg[MSG_CONTENT]
        if role not in [MSG_ROLE_SYSTEM, MSG_ROLE_USER, MSG_ROLE_ASSISTATN]:
            logger.error(f"""`role` should be in ['system', 'user', 'assistant'], but the role is {role}.""",
            ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(f"""`role` should be in ['system', 'user', 'assistant'], but the role is {role}.""")
        
        if isinstance(content, str):
            content = [{TEXT_MARK: content}]
        elif isinstance(content, dict):
            content = [content]
        
        text_list, tmp_image_list, tmp_image_path_list = from_list_format(content)
        msg[MSG_CONTENT] = '\n'.join(text_list)
        image_list.extend(tmp_image_list)
        image_path_list.extend(tmp_image_path_list)
    
    return process_prompts(msgs, image_list, image_path_list, config, processor)


def inner_tokenize(inputs, config, processor, **kwargs):
    template_func = apply_chat_template if MSG_ROLE in inputs[0] else apply_general_template
    data, image_path_list = template_func(
        inputs=inputs,
        config=config,
        processor=processor)
    if len(image_path_list) == 0:
        return torch.tensor(data, dtype=torch.int64)
    
    image_path = image_path_list[0]

    tgt_sizes = data[TGT_SIZES]
    input_ids = data[INPUT_IDS][0].clone().detach().type(torch.int64)
    image_bounds = data[IMAGE_BOUND][0]
    image_pad_mask = []
    for pair in image_bounds:
        image_pad_mask.append(torch.arange(pair[0], pair[1]))
    image_pad_mask = torch.cat(image_pad_mask, dim=0)

    shm_name_save_path = kwargs.get('shm_name_save_path', None)
    shm_name_list, shape_value_list = [], []

    if shm_name_save_path is None:
        shm_name_save_dir = os.path.dirname(os.path.dirname(image_path))
        shm_name_save_path = os.path.join(shm_name_save_dir, SHM_FILE_NAME)

    shm_name, shape_value = process_shm(tgt_sizes[0], shm_name_save_path, dtype=np.int64)
    shm_name_list.append(shm_name)
    shape_value_list.append(shape_value)

    if PIXEL_VALUES in data.keys():
        image_pixels = data[PIXEL_VALUES]
        for _, image_pixel in enumerate(image_pixels[0]):
            shm_name, shape_value = process_shm(image_pixel, shm_name_save_path)
            shm_name_list.append(shm_name)
            shape_value_list.append(shape_value)

    shm_size = len(shm_name_list)
    input_ids[image_pad_mask[0]] = VISION_START_TOKEN_ID
    for i in range(shm_size):
        input_ids[image_pad_mask[i * 2 + 1]] = torch.tensor(shm_name_list[i], dtype=torch.int64)
        input_ids[image_pad_mask[i * 2 + 2]] = torch.tensor(shape_value_list[i], dtype=torch.int64)
    eos_pos = image_pad_mask[shm_size * 2 + 1]
    input_ids[eos_pos] = VISION_END_TOKEN_ID

    return input_ids


class Minicpmqwen2v2InputBuilder(InputBuilder):
    def __init__(self, config, processor, **kwargs):
        self.config = config
        self.processor = processor
        tokenizer = processor.tokenizer
        super().__init__(tokenizer, **kwargs)

    def make_context(
            self,
            rank: int,
            conversation: List[Dict[str, List[Dict]]],
            **kwargs):
        
        if not isinstance(conversation[0][MSG_CONTENT], list):
            logger.error('The `conversation` \"content\" should be a List[Dict].',
            ErrorCode.ATB_MODELS_PARAM_INVALID)
            raise ValueError('The `conversation` \"content\" should be a List[Dict].')

        context_tokens = inner_tokenize(
            inputs=conversation,
            config=self.config,
            processor=self.processor,
            **kwargs)
        
        return context_tokens