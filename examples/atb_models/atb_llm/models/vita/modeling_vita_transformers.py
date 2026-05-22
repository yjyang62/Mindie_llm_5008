# coding=utf-8
# Copyright (c) 2019 Shigeki Karita
#               2020 Mobvoi Inc (Binbin Zhang)
#               2022 Ximalaya Inc (Yuguang Yang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Implement subsequent_chunk_mask based on subsequent_chunk_mask from wenet-e2e/wenet
# Implement add_optional_chunk_mask based on add_optional_chunk_mask from wenet-e2e/wenet
# Implement MultiHeadedAttention based on MultiHeadedAttention from wenet-e2e/wenet
# Implement PositionalEncoding based on PositionalEncoding from wenet-e2e/wenet
# Implement RelPositionalEncoding based on RelPositionalEncoding from wenet-e2e/wenet
# Implement PositionwiseFeedForward based on PositionwiseFeedForward from wenet-e2e/wenet
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.buque
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import math
from dataclasses import dataclass

from distutils.util import strtobool as dist_strtobool
import torch
import torch.nn as nn
from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode


@dataclass
class TransformerConfig:
    attention_dim: int = None
    attention_dropout_rate: float = None
    attention_heads: int = None
    chunk_size: int = None
    concat_after: bool = None
    dropout_rate: float = None
    dynamic_chunks: bool = None
    input_dim: int = None
    input_layer: str = None
    left_chunks: int = None
    linear_units: int = None
    normalize_before: bool = None
    num_blocks: int = None
    output_dim: int = None
    pos_enc_class: str = None
    positional_dropout_rate: float = None
    positionwise_layer_type: str = None


@dataclass
class ChunkParams:
    xs: torch.Tensor
    masks: torch.Tensor
    use_dynamic_chunk: bool
    use_dynamic_left_chunk: bool
    decoding_chunk_size: int
    static_chunk_size: int
    num_decoding_left_chunks: int


def strtobool(x):
    return bool(dist_strtobool(x))


def subsequent_chunk_mask(
    size: int,
    ck_size: int,
    num_l_cks: int = -1,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    ret = torch.zeros(size, size, device=device, dtype=torch.bool)
    for i in range(size):
        if num_l_cks < 0:
            start = 0
        else:
            start = max((i // ck_size - num_l_cks) * ck_size, 0)
        ending = min((i // ck_size + 1) * ck_size, size)
        ret[i, start:ending] = True
    return ret


def add_optional_chunk_mask(chunkparams: ChunkParams):  
    if chunkparams.use_dynamic_chunk:
        max_len = chunkparams.xs.size(1)
        if chunkparams.decoding_chunk_size < 0:
            chunk_size = max_len
            num_l_cks = -1
        elif chunkparams.decoding_chunk_size > 0:
            chunk_size = chunkparams.decoding_chunk_size
            num_l_cks = chunkparams.num_decoding_left_chunks
        else:
            chunk_size = torch.randint(1, max_len, (1,)).item()
            num_l_cks = -1
            if chunk_size > max_len // 2:
                chunk_size = max_len
            else:
                chunk_size = chunk_size % 25 + 1
                if chunkparams.use_dynamic_left_chunk:
                    max_left_chunks = (max_len - 1) // chunk_size
                    num_l_cks = torch.randint(0, max_left_chunks, (1,)).item()
        ck_masks = subsequent_chunk_mask(
            chunkparams.xs.size(1), chunk_size, num_l_cks, chunkparams.xs.device
        )  # (L, L)
        ck_masks = ck_masks.unsqueeze(0)  # (1, L, L)
        ck_masks = chunkparams.masks & ck_masks  # (B, L, L)
    elif chunkparams.static_chunk_size > 0:
        num_l_cks = chunkparams.num_decoding_left_chunks
        ck_masks = subsequent_chunk_mask(
            chunkparams.xs.size(1), chunkparams.static_chunk_size, num_l_cks, chunkparams.xs.device
        )  # (L, L)
        ck_masks = ck_masks.unsqueeze(0)  # (1, L, L)
        ck_masks = chunkparams.masks & ck_masks  # (B, L, L)
    else:
        ck_masks = chunkparams.masks
    return ck_masks


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.

    :param int config.attention_heads: the number of head s
    :param int config.attention_dim: the number of features
    :param float dropout_rate: dropout rate

    """

    def __init__(self, config: TransformerConfig):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadedAttention, self).__init__()
        if config.attention_dim % config.attention_heads != 0:
            logger.error("N-feat must be divisible by N-head.",
            ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError("N-feat must be divisible by N-head.")
        # We assume d_v always equals d_k
        self.d_k = config.attention_dim // config.attention_heads
        self.h = config.attention_heads
        self.linear_q = nn.Linear(config.attention_dim, config.attention_dim)
        self.linear_k = nn.Linear(config.attention_dim, config.attention_dim)
        self.linear_v = nn.Linear(config.attention_dim, config.attention_dim)
        self.linear_out = nn.Linear(config.attention_dim, config.attention_dim)
        self.dropout = nn.Dropout(p=config.attention_dropout_rate)
        self.min_value = float(torch.finfo(torch.float16).min)
        # chunk par
        if config.chunk_size > 0 and config.left_chunks > 0:  # for streaming mode
            self.buffersize = config.chunk_size * (config.left_chunks)
            self.left_chunk_size = config.chunk_size * config.left_chunks
        else:  # for non-streaming mode
            self.buffersize = 1
            self.left_chunk_size = 1
        self.chunk_size = config.chunk_size

        # encoding setup
        if config.pos_enc_class == "rel-enc":
            self.rel_enc = True
            self.linear_pos = nn.Linear(config.attention_dim, config.attention_dim, bias=False)
            # these two learnable bias are used in matrix c and matrix d
            self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
            self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
            torch.nn.init.xavier_uniform_(self.pos_bias_u)
            torch.nn.init.xavier_uniform_(self.pos_bias_v)
        else:
            self.rel_enc = False
            self.linear_pos = nn.Identity()
            self.pos_bias_u = torch.tensor([0])
            self.pos_bias_v = torch.tensor([0])

        # buffer
        self.key_buffer_size = 1 * self.h * self.buffersize * self.d_k
        self.value_buffer_size = 1 * self.h * self.buffersize * self.d_k
        if self.chunk_size > 0:
            self.buffer_mask_size = 1 * self.h * self.chunk_size * self.buffersize
        else:
            self.buffer_mask = torch.ones([1, self.h, 1, 1], dtype=torch.bool)

    @torch.jit.unused
    def rel_shift(self, x, zero_triu: bool = False):
        """Compute relative positinal encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, size).
            zero_triu (bool): If true, return the lower triangular part of
                the matrix.
        Returns:
            torch.Tensor: Output tensor.
        """

        zero_pad = torch.zeros(
            (x.size()[0], x.size()[1], x.size()[2], 1), device=x.device, dtype=x.dtype
        )
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(x.size()[0], x.size()[1], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(2), x.size(3)))
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]
        return x

    @torch.jit.export
    def forward(self, query, key, value, mask=None, pos_emb=torch.tensor(1.0)):
        # type: (Tensor, Tensor, Tensor, Optional[Tensor], Tensor) -> Tensor
        """Compute 'Scaled Dot Product Attention'.

        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor key: (batch, time2, size)
        :param torch.Tensor value: (batch, time2, size)
        :param torch.Tensor mask: (batch, time1, time2)
        :param torch.nn.Dropout dropout:
        :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
             weighted by the query dot key attention (batch, head, time1, time2)
        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        if self.rel_enc:
            q = q.transpose(1, 2)  # (batch, time1, head, d_k)
            n_batch_pos = pos_emb.size(0)
            p = self.linear_pos(pos_emb.to(query.dtype)).view(n_batch_pos, -1, self.h, self.d_k)
            p = p.transpose(1, 2)  # (batch, head, time1, d_k)
            q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
            q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)
            # compute attention score
            # first compute matrix a and matrix c
            matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
            # compute matrix b and matrix d
            matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
            # Remove rel_shift since it is useless in speech recognition,
            # and it requires special attention for streaming.
            scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)  # (batch, head, time1, time2)
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch, head, time1, time2)

        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, time1, time2)
            scores = scores.masked_fill(mask, self.min_value)
            attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
        else:
            attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(attn)

        x = torch.matmul(p_attn, v)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)
        return self.linear_out(x)  # (batch, time1, d_model)



class PositionalEncoding(torch.nn.Module):
    """Positional encoding.
    :param int d_model: embedding dim
    :param float dropout_rate: dropout rate
    :param int max_len: maximum input length
    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))
    """
    def __init__(
        self, config: TransformerConfig, max_len: int = 1500
    ):
        """Construct an PositionalEncoding object."""
        super().__init__()
        self.xscale = math.sqrt(config.attention_dim)
        self.dropout = torch.nn.Dropout(p=config.positional_dropout_rate)
        self.max_len = max_len
        self.pe = torch.zeros(self.max_len, config.attention_dim)
        position = torch.arange(0, self.max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, config.attention_dim, 2, dtype=torch.float32)
            * -(math.log(10000.0) / config.attention_dim)
        )
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)

    def forward(self, x: torch.Tensor, offset: int = 0):
        """Add positional encoding.
        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, ...)
            offset (int): position offset
        Returns:
            torch.Tensor: Encoded tensor. Its shape is (batch, time, ...)
            torch.Tensor: for compatibility to RelPositionalEncoding
        """
        if offset + x.size(1) >= self.max_len:
            logger.error("Input of PositionalEncoding is too long.",
            ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError("Input of PositionalEncoding is too long.")
        self.pe = self.pe.to(x.device)
        pos_emb = self.pe[:, offset: offset + x.size(1)]
        x = x * self.xscale + pos_emb
        return self.dropout(x), self.dropout(pos_emb)

    def position_encoding(self, offset: int, size: int):
        """For getting encoding in a streaming fashion
        Attention!!!!!
        we apply dropout only once at the whole utterance level in a none
        streaming way, but will call this function several times with
        increasing input size in a streaming scenario, so the dropout will
        be applied several times.
        Args:
            offset (int): start offset
            size (int): requried size of position encoding
        Returns:
            torch.Tensor: Corresponding encoding
        """
        if offset + size >= self.max_len:
            logger.error("Input of PositionalEncoding is too long.",
            ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError("Input of PositionalEncoding is too long.")
        return self.dropout(self.pe[:, offset: offset + size])


class RelPositionalEncoding(PositionalEncoding):
    """Relative positional encoding module.
    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
    """

    def __init__(
        self,
        config: TransformerConfig,
        max_len: int = 5000,
    ):
        """Initialize class."""
        super().__init__(config, max_len)
        self.full_chunk_size = (config.left_chunks + 1) * config.chunk_size

        self.div_term = torch.exp(
            torch.arange(0, config.attention_dim, 2, dtype=torch.float32)
            * -(math.log(10000.0) / config.attention_dim)
        )
        self.max_len = config.chunk_size * (max_len // config.chunk_size) - self.full_chunk_size

    @torch.jit.export
    def forward(self, x: torch.Tensor, offset: int = 0):
        """Compute positional encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).
        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
            torch.Tensor: Positional embedding tensor (1, time, `*`).
        """
        self.pe = self.pe.to(x.device)
        x = x * self.xscale
        pos_emb = self.pe[:, offset: offset + x.size(1)]
        return self.dropout(x), self.dropout(pos_emb)


class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.
    :param int config.attention_dim: input dimenstion
    :param int config.linear_units: number of hidden units
    :param float dropout_rate: dropout rate
    """

    def __init__(self, config: TransformerConfig):
        """Construct an PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(config.attention_dim, config.linear_units)
        self.w_2 = torch.nn.Linear(config.linear_units, config.attention_dim)
        self.dropout = torch.nn.Dropout(config.dropout_rate)

    def forward(self, x):
        """Forward funciton."""
        return self.w_2(self.dropout(torch.relu(self.w_1(x))))


class TransformerLayer(nn.Module):
    """Transformer layer module.

    :param int size: input dim
    :param self_attn: self attention module
    :param feed_forward: feed forward module
    :param float dropout_rate: dropout rate
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied. i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)

    """

    def __init__(self, config, self_attn, feed_forward):
        """Construct an TransformerLayer object."""
        super(TransformerLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = torch.nn.LayerNorm(config.attention_dim)
        self.norm2 = torch.nn.LayerNorm(config.attention_dim)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.normalize_before = config.normalize_before
        self.concat_after = config.concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(config.attention_dim + config.attention_dim, config.attention_dim)
        else:
            self.concat_linear = nn.Identity()

    @torch.jit.unused
    def forward(self, x, mask, pos_emb):
        """Compute encoded features.

        :param torch.Tensor x: encoded source features (batch, max_time_in, size)
        :param torch.Tensor mask: mask for x (batch, max_time_in)
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        residual = x
        if self.normalize_before:
            x = self.norm1(x)
        if self.concat_after:
            x_concat = torch.cat((x, self.self_attn(x, x, x, mask, pos_emb)), dim=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(self.self_attn(x, x, x, mask, pos_emb))
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)

        return x, mask, pos_emb


class Transformer(torch.nn.Module):
    def __init__(self, config):
        """Construct an Encoder object."""
        super(Transformer, self).__init__()
        self.config = config
        if self.config.pos_enc_class == "abs-enc":
            self.pe = PositionalEncoding(self.config)
        elif self.config.pos_enc_class == "rel-enc":
            self.pe = RelPositionalEncoding(self.config)

        if self.config.input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(self.config.input_dim, self.config.attention_dim),
                torch.nn.LayerNorm(self.config.attention_dim),
                torch.nn.Dropout(self.config.dropout_rate),
                torch.nn.ReLU(),
            )
        elif self.config.input_layer == "none":
            self.embed = torch.nn.Sequential(torch.nn.Identity())
        else:
            logger.error("Unknown `input_layer`: " + self.config.input_layer,
            ErrorCode.ATB_MODELS_PARAM_INVALID)
            raise ValueError("Unknown `input_layer`: " + self.config.input_layer)
        self.embed_layer_num = len(self.embed)

        if self.config.positionwise_layer_type != "linear":
            logger.error("Support only linear or conv1d.",
            ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise NotImplementedError("Support only linear or conv1d.")

        self.encoders = nn.ModuleList([TransformerLayer(
                self.config,
                MultiHeadedAttention(self.config),
                PositionwiseFeedForward(self.config),
            ) for _ in range(self.config.num_blocks)])
        if self.config.normalize_before:
            self.after_norm = torch.nn.LayerNorm(self.config.attention_dim)
      

    @torch.jit.unused
    def forward(self, xs, ilens=None, masks=None):
        """Embed positions in tensor.

        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :return: position embedded tensor and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]:
        """
        if self.config.dynamic_chunks is True:  # and self.training:
            chunk_masks = add_optional_chunk_mask(ChunkParams(xs, masks, True, True, 0, 0, -1))
        else:
            chunk_masks = add_optional_chunk_mask(ChunkParams(xs, masks, False, False, self.config.chunk_size,
                                                              self.config.chunk_size, self.config.left_chunks)
            ).to(xs.device)
        xs = self.embed(xs)
        xs, pos_emb = self.pe(xs)
        for _, encoder_layer in enumerate(self.encoders):
            xs, chunk_masks, pos_emb = encoder_layer(xs, chunk_masks, pos_emb)
        if self.config.normalize_before:
            xs = self.after_norm(xs)
        return xs, ilens, masks