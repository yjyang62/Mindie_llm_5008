# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import importlib

import torch
from torch import nn
from einops import rearrange
from janus.models.clip_encoder import CLIPVisionTower
from janus.models.projector import MlpProjector
from janus.models.vq_model import VQ_models

from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode
from ..base.flash_causal_multimodal import get_supported_models
from ..base.config import QuantizationConfig, BaseConfig
from ..base.causal_lm import CausalLM


MODEL_TYPE = "model_type"
LLAMA = "llama"


class VisionHead(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.output_mlp_projector = torch.nn.Linear(
            params.n_embed, params.image_token_embed
        )
        self.vision_activation = torch.nn.Sigmoid()
        self.vision_head = torch.nn.Linear(
            params.image_token_embed, params.image_token_size
        )

    def forward(self, x):
        x = x @ self.output_mlp_projector.weight.transpose(1, 0) + self.output_mlp_projector.bias
        x = x * self.vision_activation(1.702 * x)

        x = (x @ self.vision_head.weight.transpose(1, 0)) + self.vision_head.bias
        return x


def get_llm_model(model_type):
    supported_models = get_supported_models()
    if model_type not in supported_models:
        msg = "Unsupported model type." + \
              " Verify that the corresponding folder exists in the atb_llm.models path."
        logger.error(
            msg,
            ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE
        )
        raise NotImplementedError(msg)
    
    model_file_dir_name = f"atb_llm.models.{model_type}."
    model_file_name = 'causal'
    module_path = f"{model_file_dir_name}{model_file_name}_{model_type}_edge"
    module = importlib.import_module(module_path)
    model_cls_name = f"{model_type.capitalize()}ForCausalLM"
    model_cls = getattr(module, model_cls_name)
    return model_cls


class MultiModalLLm(CausalLM):
    def __init__(self, config, weights, **kwargs):
        if getattr(config, "language_config"):
            if not config.quantize:
                setattr(config.language_config, 'quantize', None)
            else:
                setattr(config.language_config, 'quantize', config.quantize)
            setattr(config.language_config, 'quantization_config', QuantizationConfig(**{}))
            super().__init__(config.language_config, weights, **kwargs)
        else:
            super().__init__(config, weights, **kwargs)
        self.config = config
        self.weights = weights
        self.vocab_size = config.language_config.vocab_size
        self.vision_model = CLIPVisionTower(**config.vision_config.params)
        self.aligner = MlpProjector(config.aligner_config.params)
        self.gen_vision_model = VQ_models["VQ-16"]()
        self.gen_aligner = MlpProjector(config.gen_aligner_config.params)
        self.gen_head = VisionHead(config.gen_head_config.params)

        self.gen_embed = torch.nn.Embedding(
            config.gen_vision_config.params.image_token_size, config.gen_vision_config.params.n_embed
        )

        self.language_model = None
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.init_vision_model()
        self.init_aligner()
        self.init_gen_vision_model()
        self.init_gen_aligner()
        self.init_gen_head()
        self.init_gen_embed()
        self.init_llm()
        self.model_type = None

    @staticmethod
    def init_vision_weight(module, weights, prefix):
        vision_weights = [vision_weight for vision_weight in module.state_dict().keys()]
        for vision_weight in vision_weights:
            saved_weight = torch.nn.Parameter(
                    weights.get_tensor(f"{prefix}.{vision_weight}"),
                    requires_grad=False
                )
            vision_weight_list = vision_weight.split(".")
            target_module = module
            for nxt_module in vision_weight_list[:-1]:
                target_module = getattr(target_module, nxt_module)
            setattr(target_module, vision_weight_list[-1], saved_weight)

    def init_vision_model(self):
        self.init_vision_weight(self.vision_model, self.weights, "vision_model")
    
    def init_aligner(self):
        self.init_vision_weight(self.aligner, self.weights, "aligner")
    
    def init_gen_vision_model(self):
        self.init_vision_weight(self.gen_vision_model, self.weights, "gen_vision_model")

    def init_gen_aligner(self):
        self.init_vision_weight(self.gen_aligner, self.weights, "gen_aligner")

    def init_gen_head(self):
        self.init_vision_weight(self.gen_head, self.weights, "gen_head")
    
    def init_gen_embed(self):
        self.init_vision_weight(self.gen_embed, self.weights, "gen_embed")

    def init_llm(self):
        self.model_type = self.config.language_config.model_type
        if self.model_type in [LLAMA]:
            self.model_type = LLAMA
        model_cls = get_llm_model(self.model_type)
        self.language_model = model_cls(self.config.language_config,
                                  self.weights,
                                  model_prefix="language_model.model",
                                  lmhead_prefix="language_model.lm_head")
        self.language_model.skip_word_embedding = True


class JanusForCausalLM(MultiModalLLm):
    def __init__(self, config, weights, **kwargs):
        super().__init__(config, weights, **kwargs)
        self.config = config
        self.vocab_size = config.language_config.vocab_size
    
    def prepare_inputs_embeds(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        images_seq_mask: torch.LongTensor,
        images_emb_mask: torch.LongTensor,
        **kwargs,
    ):
        bs, n = pixel_values.shape[0:2]
        images = rearrange(pixel_values, "b n c h w -> (b n) c h w")
        images_embeds = self.aligner(self.vision_model(images))

        images_embeds = rearrange(images_embeds, "(b n) t d -> b (n t) d", b=bs, n=n)
        images_emb_mask = rearrange(images_emb_mask, "b n t -> b (n t)")

        input_ids[input_ids < 0] = 0  # ignore the image embeddings
        inputs_embeds = self.get_input_embeddings_cpu()(input_ids.cpu()).npu()
        self.language_model.model.embed_tokens.weight = nn.Parameter(
            self.language_model.model.embed_tokens.weight.npu())
        # replace with the image embeddings
        inputs_embeds[images_seq_mask] = images_embeds[images_emb_mask]

        return inputs_embeds
    
    def prepare_gen_img_embeds(self, image_ids: torch.LongTensor):
        return self.gen_aligner(self.gen_embed(image_ids))
    
    def get_input_embeddings_cpu(self):
        self.language_model.model.embed_tokens.weight = nn.Parameter(
            self.language_model.model.embed_tokens.weight.cpu())
        return self.language_model.model.embed_tokens

    def init_ascend_operations(self, config: BaseConfig):
        pass

    def init_ascend_weight(self):
        pass