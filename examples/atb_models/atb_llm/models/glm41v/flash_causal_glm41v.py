# coding=utf-8
# Copyright 2025 The ZhipuAI Inc. team and HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#          http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Implement part of this file based on transformers
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import itertools
from collections import namedtuple
from typing import Optional, List, Tuple
import torch
import numpy as np

from atb_llm.utils.shm_utils import get_data_from_shm
from atb_llm.utils.log.error_code import ErrorCode
from atb_llm.utils.log.logging import logger
from atb_llm.models.base.flash_causal_multimodal import MultiModalLLm
from .modeling_glm41v_vit_atb import Glm41vVisionModel
from .modeling_glm41v_text import FlashGlm41vTextModelForCausalLM


IMAGE = "image"
VIDEO = "video"
TEXT = "text"
_SHM_TOKEN_LEN = 8


ShmInfo = namedtuple('ShmInfo', [
    'pixel_values_shm_name', 'pixel_values_shape_value',
    'image_grid_thw_shm_name', 'image_grid_thw_shape_value',
    'pixel_values_videos_shm_name', 'pixel_values_videos_shape_value',
    'video_grid_thw_shm_name', 'video_grid_thw_shape_value'
])


class FlashGlm41vForCausalLM(MultiModalLLm):
    def __init__(self, config, weights, **kwargs):
        super().__init__(config, weights, **kwargs)
        self.config = config
        self.weights = weights
        self.npu_id = weights.device.index
        self.device = f"npu:{self.npu_id}"
        self.image_token_id = self.config.image_token_id
        self.spatial_merge_size = self.config.vision_config.spatial_merge_size
        self.image_start_token_id = self.config.image_start_token_id
        self.image_end_token_id = self.config.image_end_token_id
        self.video_start_token_id = self.config.video_start_token_id
        self.video_end_token_id = self.config.video_end_token_id
        self.inference_mode = kwargs.get("inference_mode", None)

    def init_vit(self):
        setattr(self.config.vision_config, "quantize", self.config.quantize)
        self.visual = Glm41vVisionModel(self.config.vision_config, self.weights)
        self.visual = self.visual.to(self.weights.device)
        self.visual.init_graph()

    def init_llm(self):
        self.language_model = FlashGlm41vTextModelForCausalLM(self.config.text_config,
                                                               self.weights,
                                                               llm_config=self.llm_config,
                                                               inference_mode=self.inference_mode)
    
    def prepare_prefill_token_service(self, total_input_ids, position_ids, input_lengths):
        if not torch.any(torch.eq(total_input_ids, self.image_token_id)):
            inputs_embeds, position_ids_thw = \
                self._get_llm_model_inputs_without_vision_info(total_input_ids, position_ids)
        else:
            seqlen_offset = 0
            inputs_embeds_list = []
            position_ids_thw_list = []
            for input_length in input_lengths.tolist():
                input_ids = total_input_ids[seqlen_offset: seqlen_offset + input_length]
                position_ids_ = position_ids[seqlen_offset: seqlen_offset + input_length]
                if not torch.any(torch.eq(input_ids, self.image_token_id)):
                    inputs_embeds, position_ids_thw = \
                        self._get_llm_model_inputs_without_vision_info(input_ids, position_ids_)
                else:
                    inputs_embeds, position_ids_thw = \
                        self._get_llm_model_inputs_with_vision_info(input_ids, position_ids_)
                inputs_embeds_list.append(inputs_embeds)
                position_ids_thw_list.append(position_ids_thw)
                seqlen_offset += input_length
            inputs_embeds = torch.cat(inputs_embeds_list, dim=0)
            position_ids_thw = torch.cat(position_ids_thw_list, dim=-1)
        return inputs_embeds, position_ids_thw

    def forward(
            self,
            input_ids: torch.Tensor,
            position_ids: torch.Tensor,
            is_prefill: bool,
            kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
            block_tables: torch.Tensor,
            slots: torch.Tensor,
            input_lengths: torch.Tensor,
            max_seq_len: int,
            lm_head_indices: Optional[torch.Tensor] = None,
            **kwargs):
        if is_prefill:
            inputs_embeds, position_ids_thw = self.prepare_prefill_token_service(input_ids,
                                                                                 position_ids,
                                                                                 input_lengths)
            kwargs.update({"position_ids_thw": position_ids_thw})
        else:
            inputs_embeds = self.language_model.embed_tokens(input_ids)
        return self.language_model.forward(inputs_embeds,
                                           position_ids,
                                           is_prefill,
                                           kv_cache,
                                           block_tables,
                                           slots,
                                           input_lengths,
                                           max_seq_len,
                                           lm_head_indices,
                                           **kwargs)
    
    def dap_forward(
            self,
            input_ids: List[torch.Tensor],
            position_ids: List[torch.Tensor],
            is_prefill: List[bool],
            kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
            block_tables: List[torch.Tensor],
            slots: List[torch.Tensor],
            input_lengths: List[torch.Tensor],
            max_seq_len: List[int],
            lm_head_indices: List[torch.Tensor | None],
            dap_kwargs: List[dict],
    ) -> torch.Tensor:
        inputs_embeds_list = []
        for i, input_id in enumerate(input_ids):
            if is_prefill[i]:
                inputs_embeds, position_ids_thw = self.prepare_prefill_token_service(input_id,
                                                                                     position_ids[i],
                                                                                     input_lengths[i])
                dap_kwargs[i].update({"position_ids_thw": position_ids_thw})
            else:
                err_msg = "Dap is not supported for decoder."
                logger.error(err_msg, ErrorCode.ATB_MODELS_PARAM_INVALID)
                raise NotImplementedError(err_msg)
            inputs_embeds_list.append(inputs_embeds)
        return self.language_model.dap_forward(inputs_embeds_list,
                                               position_ids,
                                               is_prefill,
                                               kv_cache,
                                               block_tables,
                                               slots,
                                               input_lengths,
                                               max_seq_len,
                                               lm_head_indices,
                                               dap_kwargs)
    
    def _get_token_type(self, input_ids: torch.Tensor):
        """
        Analyzes and groups input tokens by their semantic type (TEXT, IMAGE, VIDEO).

        This method categorizes tokens in the input sequence and groups consecutive tokens
        of the same type together, creating segments that represent different modalities
        in a multimodal input sequence.

        Args:
            input_ids: Tensor of token IDs representing the input sequence
        Returns:
            List of tuples where each tuple contains:
            - token_type: The type of tokens in the segment (TEXT/IMAGE/VIDEO)
            - start_index: Starting index of the segment in the input sequence
            - end_index: Ending index (exclusive) of the segment
        Example Output:
            [(TEXT, 0, 5), (IMAGE, 5, 7), (TEXT, 7, 10), (VIDEO, 10, 15), (TEXT, 15, 20)]

        """
        input_tokens = input_ids.cpu().numpy()
        if not np.any(np.equal(input_tokens, self.image_token_id)):
            return [(TEXT, 0, len(input_tokens))]
        input_token_type = np.full(len(input_tokens), TEXT, dtype=object)
        img_mask = np.equal(input_tokens, self.image_token_id)
        input_token_type[img_mask] = IMAGE
        if np.any(np.equal(input_tokens, self.video_start_token_id)):
            video_starts = np.where(input_tokens == self.video_start_token_id)[0]
            video_stops = np.where(input_tokens == self.video_end_token_id)[0]
            for start, stop in zip(video_starts, video_stops):
                video_mask = np.equal(input_tokens[start: stop + 1], self.image_token_id)
                input_token_type[start: stop + 1][video_mask] = VIDEO
        input_token_type = input_token_type.tolist()
        input_type_group = []
        for key, group in itertools.groupby(enumerate(input_token_type), lambda x: x[1]):
            group = list(group)
            start_index = group[0][0]
            end_index = group[-1][0] + 1
            input_type_group.append((key, start_index, end_index))
        return input_type_group
    
    def _generate_position_ids(
            self,
            input_ids: torch.Tensor,
            image_grid_thw: torch.Tensor,
            video_grid_thw: torch.Tensor):
        """
        Generates 3D position IDs for multimodal input sequences containing text, images, and videos.
        
        This method creates specialized position encodings that account for the spatial and temporal
        structure of visual data, providing distinct positional information for each modality type
        in the sequence.
        
        Args:
            input_ids: Token sequence containing text and special image/video tokens
            image_grid_thw: Tensor containing image grid dimensions (T, H, W) for each image
            video_grid_thw: Tensor containing video grid dimensions (T, H, W) for each video
            
        Returns:
            torch.Tensor: 3D position IDs of shape (3, sequence_length) where the three dimensions
                        represent (temporal, height, width) positions for visual tokens and
                        sequential positions for text tokens.
        
        """
        image_index, video_index = 0, 0
        video_group_index = 0
        input_type_group = self._get_token_type(input_ids)
        llm_pos_ids_list = []
        video_frame_num = 1
        for modality_type, start_idx, end_idx in input_type_group:
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            if modality_type == IMAGE:
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    image_grid_thw[image_index][0].item(),
                    image_grid_thw[image_index][1].item() // self.spatial_merge_size,
                    image_grid_thw[image_index][2].item() // self.spatial_merge_size,
                )
                t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + st_idx)
                image_index += 1
                video_frame_num = 1
            elif modality_type == VIDEO:
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    video_frame_num,
                    video_grid_thw[video_index][1].item() // self.spatial_merge_size,
                    video_grid_thw[video_index][2].item() // self.spatial_merge_size,
                )
                for t_idx in range(llm_grid_t):
                    t_index = torch.tensor(t_idx).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(1, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(1, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + st_idx)
                video_group_index += 1
                if video_group_index >= video_grid_thw[video_index][0]:
                    video_index += 1
                    video_group_index = 0
                video_frame_num += 1
            else:
                text_len = end_idx - start_idx
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                video_frame_num = 1
        position_ids = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1).to(self.weights.device).to(input_ids.dtype)
        return position_ids
    
    def _get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: torch.LongTensor = None):
        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        return image_embeds
    
    def _get_video_features(
        self, pixel_values_videos: torch.FloatTensor, video_grid_thw: torch.LongTensor = None):
        pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
        # reshape video_grid_thw -> [b, 3] -> [1, h, w] * frames
        temp_frames_hw = []
        for t, h, w in video_grid_thw:
            repeated_row = torch.tensor([1, h.item(), w.item()]).unsqueeze(0).repeat(t, 1)
            temp_frames_hw.append(repeated_row)
        flattened_video_grid_thw = torch.cat(temp_frames_hw, dim=0)
        video_embeds = self.visual(pixel_values_videos, grid_thw=flattened_video_grid_thw)
        split_sizes = (video_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        video_embeds = torch.split(video_embeds, split_sizes)
        return video_embeds
    
    def _get_llm_model_inputs_without_vision_info(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Prepares LLM model inputs when no visual information (image/video) is available.
        
        This method generates the basic input components for the language model in scenarios
        where visual embeddings are not provided, creating placeholder visual embeddings
        to maintain consistent input structure.

        Args:
            input_ids (torch.Tensor): 
                Token IDs from the text input with shape [seq_len].
            position_ids (torch.Tensor): 
                Position IDs for the input sequences.
        """
        inputs_embeds = self.language_model.embed_tokens(input_ids)
        position_ids_thw = position_ids.view(1, -1).expand(3, -1)
        return inputs_embeds, position_ids_thw

    def _get_llm_model_inputs_with_vision_info(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Prepares LLM model inputs when visual information (image/video) is present in the input.
    
        This method processes inputs that contain visual tokens, extracts visual information from SHM (Shared Memory),
        and integrates visual embeddings with text embeddings for multimodal processing.

        Args:
            input_ids (torch.Tensor): 
                Token IDs from the text input with shape [seq_len].
            position_ids (torch.Tensor): 
                Position IDs for the input sequences.
        """
        image_grid_thw, video_grid_thw = None, None
        inputs_embeds = self.language_model.embed_tokens(input_ids)
        boi_pos = torch.where(torch.eq(input_ids, self.image_start_token_id))[0][0].item()
        shm_info_ids = input_ids[boi_pos + 1: boi_pos + _SHM_TOKEN_LEN + 1].cpu().tolist()
        shm_info = ShmInfo(*shm_info_ids)
        try:
            if shm_info.pixel_values_shm_name:
                input_image = get_data_from_shm(
                    shm_info.pixel_values_shm_name, shm_info.pixel_values_shape_value, np.float32, self.device
                ).to(dtype=inputs_embeds.dtype)
                image_grid_thw = get_data_from_shm(
                    shm_info.image_grid_thw_shm_name, shm_info.image_grid_thw_shape_value, np.int32, self.device
                ).to(dtype=torch.int64)
                image_embeds = self._get_image_features(input_image, image_grid_thw)
            if shm_info.pixel_values_videos_shm_name:
                input_video = get_data_from_shm(
                    shm_info.pixel_values_videos_shm_name, shm_info.pixel_values_videos_shape_value,
                    np.float32, self.device
                ).to(dtype=inputs_embeds.dtype)
                video_grid_thw = get_data_from_shm(
                    shm_info.video_grid_thw_shm_name, shm_info.video_grid_thw_shape_value, np.int32, self.device
                ).to(dtype=torch.int64)
                image_embeds = self._get_video_features(input_video, video_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0)
            input_ids[boi_pos + 1: boi_pos + 1 + _SHM_TOKEN_LEN].copy_(
                torch.tensor([self.image_token_id] * _SHM_TOKEN_LEN, dtype=input_ids.dtype))
            image_mask = input_ids == self.image_token_id
            inputs_embeds[image_mask] = image_embeds
            position_ids_thw = self._generate_position_ids(input_ids, image_grid_thw, video_grid_thw)
        except Exception as e:
            logger.warning(f"Get vision info from share memory failed: {e}")
            position_ids_thw = position_ids.view(1, -1).expand(3, -1)
        return inputs_embeds, position_ids_thw
