# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.buque
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""PyTorch Qwen2_vl model."""

from typing import Optional, List, Tuple
from dataclasses import dataclass
import numpy as np
import torch
from atb_llm.utils.shm_utils import get_data_from_shm, decode_shape_from_int64
from atb_llm.models.base.flash_causal_multimodal import MultiModalLLm
from atb_llm.models.qwen2_vl.flash_causal_qwen2_using_mrope import FlashQwen2UsingMROPEForCausalLM
from atb_llm.models.qwen2_vl.modeling_qwen2_5_vl_vit_atb import Qwen25VisionTransformerPretrainedModelATB
from atb_llm.models.qwen2_vl.modeling_qwen2_vl_vit_atb import Qwen2VisionTransformerPretrainedModelATB

SPATIAL_MERGE_SIZE = 2
VISION_START_TOKEN_ID = 151652
VISION_END_TOKEN_ID = 151653
IMAGE_TOKEN_ID = 151655
VIDEO_TOKEN_ID = 151656
MROPE_SECTION = [16, 24, 24]
SHM_VALUE_TOKEN_OFFSET = 1
SHAPE_VALUE_TOKEN_OFFSET = 2
IMAGE_THW_TOKEN_OFFSET = 3
SECOND_PER_GRID_T_SHM_OFFSET = 4
SECOND_PER_GRID_T_SHAPE_OFFSET = 5
IMAGE_MEAN = [0.48145466, 0.4578275, 0.40821073]
IMAGE_STD = [0.26862954, 0.26130258, 0.27577711]
IMAGE_SCALE = 1 / 255
NORMALIZATION_CHANNELS = 3
NORMALIZATION_OUTPUT_SIZE = 2 * 14 * 14 * 3
PATCH_SIZE = 2 * 14 * 14


@dataclass
class GetPositionIdsTHWInputData:
    input_ids: any
    position_ids: any
    input_lengths: any
    image_grid_thw: any = None
    video_grid_thw: any = None
    second_per_grid_ts: any = None


class FlashQwen2vlForCausalLM(MultiModalLLm):
    def __init__(self, config, weights, **kwargs):
        self.npu_id = weights.device.index
        self.tp_rank = weights.process_group.rank()
        self.tp_world_size = weights.process_group.size()
        self.config = config
        self.kwargs = kwargs
        self.image_token_id = getattr(self.config, "image_token_id", IMAGE_TOKEN_ID)
        self.video_token_id = getattr(self.config, "video_token_id", VIDEO_TOKEN_ID)
        self.vision_start_token_id = getattr(self.config, "vision_start_token_id", VISION_START_TOKEN_ID)
        self.vision_end_token_id = getattr(self.config, "vision_end_token_id", VISION_END_TOKEN_ID)
        self.spatial_merge_size = getattr(self.config.vision_config, "spatial_merge_size", SPATIAL_MERGE_SIZE)
        self.mrope_section = self.config.mrope_section.get('mrope_section', MROPE_SECTION)
        self.language_model = None
        self.vision_tower = None
        self.enable_atb_vit_tp = kwargs.pop("enable_atb_vit_tp", True)
        self.norm_rescale_w = (IMAGE_SCALE / torch.Tensor(IMAGE_STD)).npu()
        self.norm_rescale_b = (- torch.Tensor(IMAGE_MEAN) / torch.Tensor(IMAGE_STD)).npu()
        super().__init__(config, weights, **kwargs)

    def get_input_embeddings(self):
        return self.language_model.transformer.wte

    def init_vit(self):
        if not self.layerwise_disaggregated:
            
            setattr(self.config.vision_config, "enable_atb_vit_tp", self.enable_atb_vit_tp)
            if self.config.model_type == "qwen2_5_vl":
                setattr(self.config.vision_config, "rms_norm_eps", self.config.rms_norm_eps)
                self.vision_tower = Qwen25VisionTransformerPretrainedModelATB(
                    self.config.vision_config, self.weights, self.config.max_position_embeddings
                )
            else:
                self.vision_tower = Qwen2VisionTransformerPretrainedModelATB(
                    self.config.vision_config, self.weights, self.config.max_position_embeddings
                )
            self.vision_tower = self.vision_tower.to(self.weights.device)
            self.vision_tower.encoder.init_graph()
        else:
            if self.kwargs['layerwise_disaggregated_role_type'] == "master":
                setattr(self.config.vision_config, "enable_atb_vit_tp", self.enable_atb_vit_tp)
                if self.config.model_type == "qwen2_5_vl":
                    setattr(self.config.vision_config, "rms_norm_eps", self.config.rms_norm_eps)
                    self.vision_tower = Qwen25VisionTransformerPretrainedModelATB(
                        self.config.vision_config, self.weights, self.config.max_position_embeddings
                    )
                else:
                    self.vision_tower = Qwen2VisionTransformerPretrainedModelATB(
                        self.config.vision_config, self.weights, self.config.max_position_embeddings
                    )
                self.vision_tower = self.vision_tower.to(self.weights.device)
                self.vision_tower.encoder.init_graph()

    def init_llm(self):
        if not self.layerwise_disaggregated:
            self.language_model = FlashQwen2UsingMROPEForCausalLM(self.config.text_config, self.weights)
        else:
            self.language_model = FlashQwen2UsingMROPEForCausalLM(self.config.text_config, self.weights, **self.kwargs)

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
            **kwargs
    ):
        
        
        if not self.layerwise_disaggregated:
            if is_prefill:
                if not torch.any(torch.eq(input_ids, self.image_token_id) | torch.eq(input_ids, self.video_token_id)):
                    inputs_embeds = self.get_input_embeddings()(input_ids)
                else:
                    inputs_embeds, image_grid_thw, video_grid_thw, second_per_grid_ts = \
                        self.prepare_prefill_token_service(input_ids)
                    pos_input_data = GetPositionIdsTHWInputData(
                        input_ids=input_ids,
                        position_ids=position_ids,
                        input_lengths=input_lengths,
                        image_grid_thw=image_grid_thw,
                        video_grid_thw=video_grid_thw,
                        second_per_grid_ts=second_per_grid_ts
                    )
                    if image_grid_thw is not None or video_grid_thw is not None:
                        position_ids_thw_list = self._get_position_ids_thw(pos_input_data)
                    else:
                        position_ids_thw_list = []
                    kwargs.update({"position_ids_thw_list": position_ids_thw_list})
                    kwargs.update({"mrope_section": self.mrope_section})
            else:
                inputs_embeds = input_ids
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
        else:
            
            if is_prefill:
                end_layer_flag = 0
                if kwargs['layerwise_disaggregated_exe_stage']:
                    if kwargs['layerwise_disaggregated_exe_stage'].end_exec_layer == 1:
                        end_layer_flag = 1
                if self.kwargs['layerwise_disaggregated_role_type'] == "master" and end_layer_flag != 1:
                    if not torch.any(torch.eq(input_ids, self.image_token_id) | \
                        torch.eq(input_ids, self.video_token_id)):
                        inputs_embeds = self.get_input_embeddings()(input_ids)
                    else:
                        inputs_embeds, image_grid_thw, video_grid_thw, second_per_grid_ts = \
                            self.prepare_prefill_token_service(input_ids)
                        pos_input_data = GetPositionIdsTHWInputData(
                            input_ids=input_ids,
                            position_ids=position_ids,
                            input_lengths=input_lengths,
                            image_grid_thw=image_grid_thw,
                            video_grid_thw=video_grid_thw,
                            second_per_grid_ts=second_per_grid_ts
                        )
                        if image_grid_thw is not None or video_grid_thw is not None:
                            position_ids_thw_list = self._get_position_ids_thw(pos_input_data)
                        else:
                            position_ids_thw_list = []
                        kwargs.update({"position_ids_thw_list": position_ids_thw_list})
                        kwargs.update({"mrope_section": self.mrope_section})
                        
                else:
                    inputs_embeds = input_ids
            else:
                inputs_embeds = input_ids
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

    def prepare_prefill_token_service(self, input_ids):
        bos_pos = torch.where(torch.eq(input_ids, self.vision_start_token_id))[0]
        eos_pos = torch.where(torch.eq(input_ids, self.vision_end_token_id))[0]
        vision_num = bos_pos.shape[0]
        video_grid_thw = None
        image_grid_thw = None
        image_pixel_array = []
        video_pixel_array = []
        image_grid_thw_list = []
        video_grid_thw_list = []
        if self.config.model_type == "qwen2_5_vl":
            second_per_grid_ts = []
        else:
            second_per_grid_ts = None
        for i in range(vision_num):
            if input_ids[eos_pos[i] - 1] == self.image_token_id:
                image_thw_value = input_ids[bos_pos[i] + IMAGE_THW_TOKEN_OFFSET]
                image_grid_thw = torch.tensor(decode_shape_from_int64(image_thw_value), dtype=input_ids.dtype).npu()
                image_grid_thw_list.append(image_grid_thw)

                shm_value = input_ids[bos_pos[i] + SHM_VALUE_TOKEN_OFFSET]
                shape_value = input_ids[bos_pos[i] + SHAPE_VALUE_TOKEN_OFFSET]
                shared_array = get_data_from_shm(shm_value, shape_value, np.uint8, self.weights.device)
                input_ids[bos_pos[i] + 1: bos_pos[i] + SECOND_PER_GRID_T_SHM_OFFSET] = self.image_token_id
                image_pixel_array.append(shared_array)
            elif input_ids[eos_pos[i] - 1] == self.video_token_id:
                video_thw_value = input_ids[bos_pos[i] + IMAGE_THW_TOKEN_OFFSET]
                video_grid_thw = torch.tensor(decode_shape_from_int64(video_thw_value), dtype=input_ids.dtype).npu()
                if self.config.model_type == "qwen2_5_vl":
                    second_per_grid_t_shm_value = input_ids[bos_pos[i] + SECOND_PER_GRID_T_SHM_OFFSET]
                    second_per_grid_t_shape_value = input_ids[bos_pos[i] + SECOND_PER_GRID_T_SHAPE_OFFSET]
                    second_per_grid_t_value = get_data_from_shm(second_per_grid_t_shm_value,
                                                                second_per_grid_t_shape_value, np.float32)
                    second_per_grid_ts.append(second_per_grid_t_value)
                video_grid_thw_list.append(video_grid_thw)

                shm_value = input_ids[bos_pos[i] + SHM_VALUE_TOKEN_OFFSET]
                shape_value = input_ids[bos_pos[i] + SHAPE_VALUE_TOKEN_OFFSET]
                shared_array = get_data_from_shm(shm_value, shape_value, np.uint8, self.weights.device)
                input_ids[bos_pos[i] + 1: bos_pos[i] + SECOND_PER_GRID_T_SHAPE_OFFSET + 1] = self.video_token_id
                video_pixel_array.append(shared_array)

        inputs_embeds = self.get_input_embeddings()(input_ids)

        if image_pixel_array:
            image_grid_thw = torch.stack(image_grid_thw_list, dim=0)
            image_pixel = torch.cat(image_pixel_array)
            image_pixel = image_pixel.float().reshape(-1, NORMALIZATION_CHANNELS, PATCH_SIZE).transpose(1, 2)
            image_pixel = image_pixel * self.norm_rescale_w + self.norm_rescale_b
            image_pixel = image_pixel.transpose(1, 2).reshape(-1, NORMALIZATION_OUTPUT_SIZE)
            image_features = self.vision_tower(image_pixel.to(self.vision_tower.dtype), image_grid_thw)
            image_mask = input_ids == self.image_token_id
            inputs_embeds[image_mask] = image_features

        if video_pixel_array:
            video_grid_thw = torch.stack(video_grid_thw_list, dim=0)
            video_pixel = torch.cat(video_pixel_array)
            video_pixel = video_pixel.float().reshape(-1, NORMALIZATION_CHANNELS, PATCH_SIZE).transpose(1, 2)
            video_pixel = video_pixel * self.norm_rescale_w + self.norm_rescale_b
            video_pixel = video_pixel.transpose(1, 2).reshape(-1, NORMALIZATION_OUTPUT_SIZE)
            video_features = self.vision_tower(video_pixel.to(self.vision_tower.dtype), video_grid_thw)
            video_mask = input_ids == self.video_token_id
            inputs_embeds[video_mask] = video_features
        return inputs_embeds, image_grid_thw, video_grid_thw, second_per_grid_ts

    def _get_position_ids_thw(self, pos_input_data: GetPositionIdsTHWInputData):        
        id_start = 0
        lengths_list = pos_input_data.input_lengths.tolist()
        position_ids_thw_list = []
        image_num_before = 0
        video_num_before = 0
        for length in lengths_list:
            single_image_grid_thw = None
            single_video_grid_thw = None
            single_prefill_ids = pos_input_data.input_ids[id_start:id_start + length]
            if not torch.any(torch.eq(single_prefill_ids, self.vision_start_token_id)):
                # 纯文本Batch
                single_position_ids = pos_input_data.position_ids[id_start:id_start + length]
                position_ids_thw_list.append(single_position_ids.repeat(3, 1))
                continue
            vision_start_indices = torch.argwhere(single_prefill_ids == self.vision_start_token_id)
            vision_tokens = single_prefill_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == self.image_token_id).sum()
            video_nums = (vision_tokens == self.video_token_id).sum()

            if pos_input_data.image_grid_thw is not None:
                single_image_grid_thw = pos_input_data.image_grid_thw[image_num_before:image_num_before + image_nums]
            if pos_input_data.video_grid_thw is not None:
                single_video_grid_thw = pos_input_data.video_grid_thw[video_num_before:video_num_before + video_nums]
            image_num_before = image_num_before + image_nums
            video_num_before = video_num_before + video_nums

            position_ids_thw = self._get_rope_index(
                single_prefill_ids,
                single_image_grid_thw,
                single_video_grid_thw,
                pos_input_data.second_per_grid_ts
            )
            id_start = id_start + length
            position_ids_thw_list.append(position_ids_thw)
        return position_ids_thw_list

    def _get_rope_index(
            self,
            input_ids: torch.Tensor,
            image_grid_thw: torch.Tensor,
            video_grid_thw: torch.Tensor,
            second_per_grid_ts=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image_index, video_index = 0, 0
        vision_start_indices = torch.argwhere(input_ids == self.vision_start_token_id).squeeze(1)
        vision_tokens = input_ids[vision_start_indices + 1]
        image_nums = (vision_tokens == self.image_token_id).sum()
        video_nums = (vision_tokens == self.video_token_id).sum()
        input_tokens = input_ids.tolist()
        llm_pos_ids_list: list = []
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
                t, h, w = (
                    image_grid_thw[image_index][0],
                    image_grid_thw[image_index][1],
                    image_grid_thw[image_index][2],
                )
                second_per_grid_t = 0
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                t, h, w = (
                    video_grid_thw[video_index][0],
                    video_grid_thw[video_index][1],
                    video_grid_thw[video_index][2],
                )
                if second_per_grid_ts is not None:
                    second_per_grid_t = second_per_grid_ts[video_index]
                else:
                    second_per_grid_t = 1.0
                video_index += 1
                remain_videos -= 1
                ed = ed_video
            llm_grid_t, llm_grid_h, llm_grid_w = (
                t.item(),
                h.item() // self.spatial_merge_size,
                w.item() // self.spatial_merge_size,
            )
            text_len = ed - st

            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
            if self.config.model_type == "qwen2_5_vl":
                range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)
                time_tensor = expanded_range * second_per_grid_t * self.config.vision_config.tokens_per_second
                time_tensor_long = time_tensor.long()
                t_index = time_tensor_long.flatten()
            else:
                t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
            h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
            w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
            llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

        llm_positions = torch.cat(llm_pos_ids_list, dim=1)
        position_ids_thw = llm_positions.to(self.weights.device).to(input_ids.dtype)
        return position_ids_thw
