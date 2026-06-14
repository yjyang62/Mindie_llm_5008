# coding=utf-8
# Copyright 2024 The Qwen team and Alibaba Group. All rights reserved.
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
# Implement round_by_factor based on round_by_factor from QwenLM/Qwen3-VL
# Implement ceil_by_factor based on ceil_by_factor from QwenLM/Qwen3-VL
# Implement floor_by_factor based on floor_by_factor from QwenLM/Qwen3-VL
# Implement smart_resize based on smart_resize from QwenLM/Qwen3-VL
# Implement fetch_image based on fetch_image from QwenLM/Qwen3-VL
# Implement smart_nframes based on smart_nframes from QwenLM/Qwen3-VL
# Implement _read_video_torchvision based on _read_video_torchvision from QwenLM/Qwen3-VL
# Implement fetch_video based on fetch_video from QwenLM/Qwen3-VL
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.buque
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import math

import torch
import numpy as np
from PIL import Image
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode

from atb_llm.utils.multimodal_utils import safe_open_image, check_video_path
from atb_llm.utils.shm_utils import encode_shm_name_to_int64, encode_shape_to_int64, create_shm
from atb_llm.utils.log.error_code import ErrorCode
from atb_llm.utils.log.logging import logger, print_log

IMAGE_FACTOR = 28
MERGER_FACTOR = 4
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
VIDEO_TOTAL_PIXELS = 24576 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768
PYTORCH_TENSOR = "pt"


def process_shared_memory(pixel_values, shm_name_save_path, grid_thw, second_per_grid_t=None):
    # 处理 pixel_values
    pixel_values_shm = create_shm(pixel_values.nbytes, shm_name_save_path)
    shared_array = np.ndarray(pixel_values.shape, dtype=np.uint8, buffer=pixel_values_shm.buf)
    shared_array[:] = pixel_values
    pixel_values_shm_name = encode_shm_name_to_int64(pixel_values_shm.name)
    pixel_values_shape_value = encode_shape_to_int64(pixel_values.shape)

    # 处理 grid_thw
    thw_value = encode_shape_to_int64(grid_thw[0])

    result = {
        'pixel_values_shm_name': pixel_values_shm_name,
        'pixel_values_shape_value': pixel_values_shape_value,
        'thw_value': thw_value,
        'second_per_grid_t_shm_name': None,
        'second_per_grid_t_shape_value': None
    }

    # 处理 second_per_grid_t
    if second_per_grid_t is not None:
        second_per_grid_t_shm = create_shm(second_per_grid_t.nbytes, shm_name_save_path)
        shared_array = np.ndarray(second_per_grid_t.shape, dtype=np.float32, buffer=second_per_grid_t_shm.buf)
        shared_array[:] = second_per_grid_t
        result['second_per_grid_t_shm_name'] = encode_shm_name_to_int64(second_per_grid_t_shm.name)
        result['second_per_grid_t_shape_value'] = encode_shape_to_int64(second_per_grid_t.shape)

    return result


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
        height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        logger.error(
            f"Absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}.",
            ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE
        )
        raise ValueError(
            f"Absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}."
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def fetch_image(processor, ele: dict[str, str | Image.Image], size_factor: int = IMAGE_FACTOR) -> Image.Image:
    if "image" in ele:
        image_path = ele["image"]
    else:
        image_path = ele["image_url"]
    image_obj = None
    image_obj = safe_open_image(Image, image_path)
    if image_obj is None:
        logger.error(f"Unrecognized image path input, only support local path,  got {image_path}.",
        ErrorCode.ATB_MODELS_PARAM_INVALID)
        raise ValueError(f"Unrecognized image path input, only support local path,  got {image_path}")
    image_obj = image_obj.convert("RGB")
    ## resize
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=size_factor,
        )
    else:
        width, height = image_obj.size
        min_pixels = ele.get("min_pixels", MIN_PIXELS)
        max_pixels = ele.get("max_pixels", MAX_PIXELS)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    image_obj = image_obj.resize((resized_width, resized_height))
    images_inputs = processor(images=image_obj,
                              videos=None,
                              do_normalize=False,
                              do_rescale=False,
                              return_tensors=PYTORCH_TENSOR)
    image_obj.close()
    feature_lens = images_inputs['image_grid_thw'][0][1] * images_inputs['image_grid_thw'][0][2] // MERGER_FACTOR
    return images_inputs, feature_lens


def smart_nframes(ele: dict, total_frames: int, video_fps: int | float) -> int:
    """calculate the number of frames for video used for model inputs.

    Args:
        ele (dict): a dict contains the configuration of video.
            support only `fps`:
                - fps: the fps to extract frames for model inputs.
                    - min_frames: the minimum number of frames of the video, only used when fps is provided.
                    - max_frames: the maximum number of frames of the video, only used when fps is provided.
        total_frames (int): the original total number of frames of the video.
        video_fps (int | float): the original fps of the video.

    Raises:
        ValueError: nframes should in interval [FRAME_FACTOR, total_frames].

    Returns:
        int: the number of frames for video used for model inputs.
    """
    fps = ele.get("fps", FPS)
    min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
    max_frames = floor_by_factor(ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR)
    nframes = total_frames / video_fps * fps
    nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
    nframes = floor_by_factor(nframes, FRAME_FACTOR)
    if not (FRAME_FACTOR <= nframes and nframes <= total_frames):
        logger.error(f"`nframes` should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}.",
        ErrorCode.ATB_MODELS_PARAM_INVALID)
        raise ValueError(f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}.")
    return nframes


def _read_video_torchvision(ele: dict) -> torch.Tensor:
    """read video using torchvision.io.read_video

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. only support and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    video_path = ele["video"]
    video_path = check_video_path(video_path)

    video, audio, info = io.read_video(
        video_path,
        start_pts=ele.get("video_start", 0.0),
        end_pts=ele.get("video_end", None),
        pts_unit="sec",
        output_format="TCHW",
    )
    total_frames, video_fps = video.size(0), info["video_fps"]
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(0, total_frames - 1, nframes).round().long()
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    video = video[idx]
    return video, sample_fps


def fetch_video(processor, ele: dict, image_factor: int = IMAGE_FACTOR):
    video_obj, sample_fps = _read_video_torchvision(ele)
    nframes, _, height, width = video_obj.shape

    min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
    total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
    max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
    max_pixels = ele.get("max_pixels", max_pixels)
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=image_factor,
        )
    else:
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=image_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    video_obj = transforms.functional.resize(
        video_obj,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).float()
    video_inputs = processor(images=None,
                             videos=video_obj,
                             do_normalize=False,
                             do_rescale=False,
                             return_tensors=PYTORCH_TENSOR)
    second_per_grid_ts = torch.tensor([processor.temporal_patch_size / sample_fps], dtype=torch.float32).unsqueeze(0)
    feature_lens = torch.prod(video_inputs['video_grid_thw'][0]).item() // MERGER_FACTOR
    return video_inputs, feature_lens, second_per_grid_ts
