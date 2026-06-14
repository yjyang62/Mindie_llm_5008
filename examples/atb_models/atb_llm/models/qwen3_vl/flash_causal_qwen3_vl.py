# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from typing import Optional, List, Tuple
import torch
import numpy as np

from atb_llm.utils.shm_utils import get_data_from_shm
from atb_llm.utils.log.logging import logger
from atb_llm.models.base.flash_causal_multimodal import MultiModalLLm
from .modeling_qwen3_vl_vit import Qwen3VLVisionModel
from .modeling_qwen3_vl_text import FlashQwen3VLTextModelForCausalLM


IMAGE = "image"
VIDEO = "video"
TEXT = "text"
_SHM_TOKEN_LEN = 8


class FlashQwen3vlForCausalLM(MultiModalLLm):
    def __init__(self, config, weights, **kwargs):
        super().__init__(config, weights, **kwargs)
        self.config = config
        self.weights = weights
        self.npu_id = weights.device.index
        self.device = f"npu:{self.npu_id}"
        self.spatial_merge_size = self.config.vision_config.spatial_merge_size
        self.image_token_id = self.config.image_token_id
        self.video_token_id = self.config.video_token_id
        self.vision_start_token_id = self.config.vision_start_token_id
        self.inference_mode = kwargs.get("inference_mode", None)

    def init_vit(self):
        setattr(self.config.vision_config, "quantize", self.config.quantize)
        self.visual = Qwen3VLVisionModel(self.config.vision_config, self.weights)
        self.visual = self.visual.to(self.weights.device)
        self.visual.init_graph()

    def init_llm(self):
        self.language_model = FlashQwen3VLTextModelForCausalLM(self.config.text_config,
                                                               self.weights,
                                                               llm_config=self.llm_config,
                                                               inference_mode=self.inference_mode)

    def prepare_prefill_token_service(self, total_input_ids, total_position_ids, input_lengths):
        """
        Generates model inputs for the prefill stage of large language model (LLM) inference,
        supporting multimodal sequences with or without visual information.

        This method first checks for visual start tokens in the batch-level input sequence,
        processes text-only batches in bulk for efficiency, and splits multimodal batches into
        individual samples for modality-specific processing before concatenating results to
        maintain batch consistency for LLM inference.

        Args:
            total_input_ids: Combined token sequence for all samples in the batch, including text
                             and special vision start tokens
            total_position_ids: Position ID sequence corresponding to the total_input_ids tensor
            input_lengths: Sequence length of each individual sample in the batch, used to split
                           the combined input tensor into per-sample sequences

        Returns:
            torch.Tensor: Concatenated input embeddings, shape [batch_size, seq_len, hidden_dim]
            torch.Tensor: Concatenated position IDs (thw: time/height/width for vision-lang model),
                          shape [batch_size, ..., total_seq_len]
            list[torch.Tensor]: Deepstack visual embeddings list,
                                each tensor shape [batch_size, num_patches, visual_hidden_dim]

        """
        has_vision = torch.any(torch.eq(total_input_ids, self.vision_start_token_id))
        
        if not has_vision:
            inputs_embeds, position_ids_thw, deepstack_visual_embeds = self._get_llm_model_inputs_without_vision_info(
                total_input_ids, total_position_ids
            )
            return inputs_embeds, position_ids_thw, deepstack_visual_embeds
        
        seqlen_offset = 0
        inputs_embeds_list = []
        position_ids_thw_list = []
        deepstack_visual_embeds_list = []
        
        for input_length in input_lengths.tolist():
            input_ids = total_input_ids[seqlen_offset: seqlen_offset + input_length]
            position_ids = total_position_ids[seqlen_offset: seqlen_offset + input_length]
            
            if torch.any(torch.eq(input_ids, self.vision_start_token_id)):
                inputs_embeds, position_ids_thw, deepstack_visual_embeds = \
                    self._get_llm_model_inputs_with_vision_info(input_ids, position_ids)
            else:
                inputs_embeds, position_ids_thw, deepstack_visual_embeds = \
                    self._get_llm_model_inputs_without_vision_info(input_ids, position_ids)
            
            inputs_embeds_list.append(inputs_embeds)
            position_ids_thw_list.append(position_ids_thw)
            deepstack_visual_embeds_list.append(deepstack_visual_embeds)
            seqlen_offset += input_length
        
        inputs_embeds = torch.cat(inputs_embeds_list, dim=0)
        position_ids_thw = torch.cat(position_ids_thw_list, dim=-1)
        deepstack_visual_embeds = [torch.cat(tensors, dim=0) for tensors in zip(*deepstack_visual_embeds_list)]
        return inputs_embeds, position_ids_thw, deepstack_visual_embeds

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
            inputs_embeds, position_ids_thw, deepstack_visual_embeds = self.prepare_prefill_token_service(input_ids,
                                                                                                          position_ids,
                                                                                                          input_lengths)
            kwargs.update({"position_ids_thw": position_ids_thw})
            kwargs.update({"deepstack_visual_embeds": deepstack_visual_embeds})
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
        input_ids = input_ids.cpu()
        image_grid_thw = image_grid_thw.cpu() if image_grid_thw is not None else None
        if video_grid_thw is not None:
            video_grid_thw = video_grid_thw.cpu()
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1
        image_index, video_index = 0, 0
        image_nums, video_nums = 0, 0
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
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                t, h, w = (
                    video_grid_thw[video_index][0],
                    video_grid_thw[video_index][1],
                    video_grid_thw[video_index][2],
                )
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
            t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
            h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
            w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
            llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w
        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
        position_ids = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1).to(self.weights.device).to(input_ids.dtype)
        return position_ids
    
    def _get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: torch.LongTensor = None):
        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds, deepstack_image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        return image_embeds, deepstack_image_embeds
    
    def _get_visual_features_from_shm(self, input_ids, shm_info_idx):
        deepstack_image_embeds, deepstack_video_embeds = None, None
        image_grid_thw, video_grid_thw = None, None
        pixel_values_shm_name = shm_info_idx[0]
        pixel_values_shape_value = shm_info_idx[1]
        image_grid_thw_shm_name = shm_info_idx[2]
        image_grid_thw_shape_value = shm_info_idx[3]
        pixel_values_videos_shm_name = shm_info_idx[4]
        pixel_values_videos_shape_value = shm_info_idx[5]
        video_grid_thw_shm_name = shm_info_idx[6]
        video_grid_thw_shape_value = shm_info_idx[7]
        inputs_embeds = self.language_model.embed_tokens(input_ids)
        image_mask, video_mask = None, None
        if pixel_values_shm_name:
            input_image = get_data_from_shm(
                pixel_values_shm_name, pixel_values_shape_value, np.float32, self.device
            ).to(dtype=inputs_embeds.dtype).to(input_ids.device)
            image_grid_thw = get_data_from_shm(
                image_grid_thw_shm_name, image_grid_thw_shape_value, np.int32, self.device
            ).to(dtype=torch.int64).to(input_ids.device)
            image_embeds, deepstack_image_embeds = self._get_image_features(input_image, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0)
            image_mask = input_ids == self.image_token_id
            indices = torch.where(image_mask)[0]
            inputs_embeds.index_copy_(0, indices, image_embeds)
        if pixel_values_videos_shm_name:
            input_video = get_data_from_shm(
                pixel_values_videos_shm_name, pixel_values_videos_shape_value, np.float32, self.device
            ).to(dtype=inputs_embeds.dtype).to(input_ids.device)
            video_grid_thw = get_data_from_shm(
                video_grid_thw_shm_name, video_grid_thw_shape_value, np.int32, self.device
            ).to(dtype=torch.int64).to(input_ids.device)
            video_embeds, deepstack_video_embeds = self._get_image_features(input_video, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0)
            video_mask = input_ids == self.video_token_id
            indices = torch.where(video_mask)[0]
            inputs_embeds.index_copy_(0, indices, video_embeds)
        position_ids_thw = self._generate_position_ids(input_ids, image_grid_thw, video_grid_thw)
        vision_mask = (image_mask, video_mask)
        return inputs_embeds, vision_mask, deepstack_image_embeds, deepstack_video_embeds, position_ids_thw

    def _get_deepstack_embeds_for_llm_model(
        self,
        inputs_embeds: torch.Tensor,
        image_mask: Optional[torch.Tensor],
        video_mask: Optional[torch.Tensor],
        deepstack_image_embeds: Optional[List[torch.Tensor]],
        deepstack_video_embeds: Optional[List[torch.Tensor]]
    ) -> List[torch.Tensor]:
        """
        Prepares DeepStack visual embeddings for the LLM model by aligning them with input positions.
        
        This function combines image and video embeddings from DeepStack into the appropriate positions
        in the input embedding space based on provided masking information. It creates joint embeddings
        that maintain the original input structure while incorporating visual information at specified locations.

        Args:
            inputs_embeds (torch.Tensor): 
                The base input embeddings tensor with shape [seq_len, hidden_dim] that defines the template structure.
            image_mask (Optional[torch.Tensor]): 
                Boolean mask tensor indicating positions where image embeddings should be inserted.
            video_mask (Optional[torch.Tensor]): 
                Boolean mask tensor indicating positions where video embeddings should be inserted.
            deepstack_image_embeds (Optional[List[torch.Tensor]]): 
                List of image embedding tensors from DeepStack, each matching the hidden dimension of inputs_embeds.
            deepstack_video_embeds (Optional[List[torch.Tensor]]): 
                List of video embedding tensors from DeepStack, each matching the hidden dimension of inputs_embeds.
        """
        deepstack_visual_embeds = []
        
        if image_mask is not None and video_mask is not None:
            indices_img = torch.where(image_mask)[0]
            indices_vid = torch.where(video_mask)[0]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = torch.zeros_like(
                    inputs_embeds, dtype=inputs_embeds.dtype, device=inputs_embeds.device
                )
                embed_joint.index_copy_(0, indices_img, img_embed)
                embed_joint.index_copy_(0, indices_vid, vid_embed)
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            indices_img = torch.where(image_mask)[0]
            for img_embed in deepstack_image_embeds:
                embed_joint = torch.zeros_like(
                    inputs_embeds, dtype=inputs_embeds.dtype, device=inputs_embeds.device
                )
                embed_joint.index_copy_(0, indices_img, img_embed)
                deepstack_visual_embeds.append(embed_joint)
        elif video_mask is not None:
            indices_vid = torch.where(video_mask)[0]
            for vid_embed in deepstack_video_embeds:
                embed_joint = torch.zeros_like(
                    inputs_embeds, dtype=inputs_embeds.dtype, device=inputs_embeds.device
                )
                embed_joint.index_copy_(0, indices_vid, vid_embed)
                deepstack_visual_embeds.append(embed_joint)
        
        return deepstack_visual_embeds
    
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
        num_deepstack = len(self.config.vision_config.deepstack_visual_indexes)
        deepstack_visual_embeds = [
            torch.zeros_like(inputs_embeds, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            for _ in range(num_deepstack)
        ]

        return inputs_embeds, position_ids_thw, deepstack_visual_embeds
    
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
                Token IDs containing both text and visual tokens, with shape [seq_len].
            position_ids (torch.Tensor): 
                Position IDs for the input sequences.
        """
        boi_pos = torch.where(torch.eq(input_ids, self.vision_start_token_id))[0][0].item() + 1
        shm_info_idx = input_ids[boi_pos + 1: boi_pos + 1 + _SHM_TOKEN_LEN].detach().cpu().tolist()
        input_ids[boi_pos + 1: boi_pos + 1 + _SHM_TOKEN_LEN].copy_(
            torch.tensor([input_ids[boi_pos]] * _SHM_TOKEN_LEN, dtype=input_ids.dtype))
        try:
            inputs_embeds, vision_mask, deepstack_image_embeds, deepstack_video_embeds, position_ids_thw = \
                self._get_visual_features_from_shm(input_ids, shm_info_idx)
        except Exception as e:
            logger.warning(
                f"Get vision features from share memory failed. The request will be handled without vision info."
            )
            inputs_embeds, position_ids_thw, deepstack_visual_embeds = \
                self._get_llm_model_inputs_without_vision_info(input_ids, position_ids)
            return inputs_embeds, position_ids_thw, deepstack_visual_embeds
        image_mask, video_mask = vision_mask
        deepstack_visual_embeds = self._get_deepstack_embeds_for_llm_model(
            inputs_embeds, image_mask, video_mask,
            deepstack_image_embeds, deepstack_video_embeds
        )
        return inputs_embeds, position_ids_thw, deepstack_visual_embeds