# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import os
import torch
import torch.nn as nn
from transformers import AutoConfig, CLIPImageProcessor

from atb_llm.models.vita.modeling_vita_vit_atb import InternVisionModel
from atb_llm.utils.env import ENV
from atb_llm.utils.dist import set_device_from_ranktable
from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode
from .configuration_vita_vit import InternVisionConfig


class InternViTVisionTower(nn.Module):
    def __init__(self, model_name_or_path, weights, mm_vision_tower, process_group, delay_load=False):
        super().__init__()
        self.is_loaded = False
        self.use_atb = True
        self.vision_tower_name = mm_vision_tower
        self.process_group = process_group
        self.select_layer = -1
        self.scale_pix_shuffle = 0.5
        self.model_name_or_path = model_name_or_path
        self.vision_path = os.path.join(self.model_name_or_path, self.vision_tower_name)
        self.weights = weights

        if not delay_load:
            self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_path)
            if self.use_atb:
                config = InternVisionConfig.from_pretrained(self.vision_path)
                rank_table = ENV.rank_table_file
                local_rank = ENV.local_rank
                if rank_table:
                    device = set_device_from_ranktable(self.process_group.rank, rank_table)
                else:
                    device = torch.device(f"npu:{local_rank}")
                self.vision_tower_model = InternVisionModel(config, self.weights)
                vision_weights = []
                for vision_weight in self.vision_tower_model.state_dict().keys():
                    if vision_weight.startswith("embeddings"):
                        vision_weights.append(vision_weight)
                for vision_weight in vision_weights:
                    saved_weight = torch.nn.Parameter(
                            self.weights.get_tensor(f"model.vision_tower.vision_tower.{vision_weight}"),
                            requires_grad=False
                    )
                    vision_weight_list = vision_weight.split(".")
                    target_module = self.vision_tower_model
                    for nxt_module in vision_weight_list[:-1]:
                        target_module = getattr(target_module, nxt_module)
                    setattr(target_module, vision_weight_list[-1], saved_weight)
                self.vision_tower_model = self.vision_tower_model.to(device)
                self.vision_tower_model.encoder.init_graph()
            else:
                self.vision_tower_model = InternVisionModel.from_pretrained(
                    self.vision_path
                )

            self.vision_tower_model.requires_grad_(False)
            self.is_loaded = True
        else:
            self.cfg_only = AutoConfig.from_pretrained(
                self.vision_tower_name
            )

    @property
    def dtype(self):
        return self.vision_tower_model.dtype

    @property
    def device(self):
        return self.vision_tower_model.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower_model.config
        else:
            return self.cfg_only
    
    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs["hidden_states"]
        image_features = image_features[:, 1:]
        return image_features

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(
            n, int(h * scale_factor), int(w * scale_factor), int(c / (scale_factor * scale_factor))
        )
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    @torch.no_grad()
    def forward(self, images):
        image_forward_outs = self.vision_tower_model(
            images.to(device=self.device, dtype=self.dtype)
        )
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        h = w = int(image_features.shape[1] ** 0.5)
        if image_features.shape[1] != h * w:
            logger.error("`image_features` shape error.",
            ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError("`image_features` shape error.")
        image_features = image_features.reshape(image_features.shape[0], h, w, -1)
        image_features = self.pixel_shuffle(image_features * self.scale_pix_shuffle)
        image_features = image_features.reshape(
            image_features.shape[0], -1, image_features.shape[-1]
        )

        return image_features