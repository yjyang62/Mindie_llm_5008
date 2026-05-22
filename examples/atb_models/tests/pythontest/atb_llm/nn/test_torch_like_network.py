# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import unittest
import logging

import torch
import torch_npu
import torch.nn as nn

import atb_llm
import atb_llm.nn
import atb_llm.nn.modules
from atb_llm.nn.network_manager import get_default_net
from atb_llm.nn.tensor import Tensor
from atb_llm.nn.functional import gather, split, reshape_and_cache
import atb_llm.nn.functional as F
from atb_llm.nn.modules import Linear
from tests.pythontest.atb_llm.utils import NO_BF16_SUPPORT_SOC_VERSION


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


HIDDEN_STATES = "hidden_states"


# torch golden
class RMSNorm(nn.Module):
    def __init__(self, weight, eps=1e-5):
        super(RMSNorm, self).__init__()
        self.weight = weight
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        if torch.any(rms == 0):
            return 0
        x = x / rms
        return self.weight * x


class LlamaTorchLayer:
    def __init__(self, head_num, head_dim, op_name='llama_layer'):
        self.head_num = head_num
        self.head_dim = head_dim
        self.op_name = op_name
        #只支持1 batch, 方便测试
        self.bs = 1

    @staticmethod
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def swish(x):
        return x * nn.functional.sigmoid(x)

    def rope(self, q, k, cos, sin):
        #只支持单batch, 方便测试
        cos = cos.unsqueeze(0).unsqueeze(2)  # [1, 1, seq_len, dim]
        sin = sin.unsqueeze(0).unsqueeze(2)  # [1, 1, seq_len, dim]
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed

    def forward(self, inputs):
        bs = self.bs
        seqlen = inputs[HIDDEN_STATES].shape[0] // bs
        rms_norm_1 = RMSNorm(inputs["llama_layer_norm_1.weight"])
        norm1_out = rms_norm_1(inputs[HIDDEN_STATES])
        qkv = torch.matmul(norm1_out, inputs["llama_layer_qkv.weight"].t())
        # Tensor shape: [b*s, h]
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # Tensor shape: [b, s, hn, hd]
        q = q.view(1, q.shape[0], self.head_num, self.head_dim)
        k = k.view(1, k.shape[0], self.head_num, self.head_dim)
        q_embed, k_embed = self.rope(q, k, inputs["cos"][:seqlen, :], inputs["sin"][:seqlen, :])
        # Tensor shape: [b, hn, s, hd]
        q_embed = q_embed.permute(0, 2, 1, 3)
        k_embed = k_embed.permute(0, 2, 1, 3)
        v = v.view(1, v.shape[0], self.head_num, self.head_dim).permute(0, 2, 1, 3)
        # Tensor shape: [b, hn, s, s]
        attn_weights = torch.matmul(q_embed, k_embed.transpose(2, 3))
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32).to(q_embed.dtype)
        # Tensor shape: [b, hn, s, hd]
        attn_output = torch.matmul(attn_weights, v)
        # Tensor shape: [b*s, h]
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(seqlen * bs, self.head_num * self.head_dim)
        # Tensor shape: [b*s, h]
        attn_linear_out = torch.matmul(attn_output, inputs["llama_layer_atten.weight"].t())

        res_add_out = attn_linear_out + inputs[HIDDEN_STATES]
        rms_norm_2 = RMSNorm(inputs["llama_layer_norm_2.weight"])
        norm1_out = rms_norm_2(res_add_out)

        # Tensor shape: [b*s, 2*h]
        up_gate = torch.matmul(norm1_out, inputs["llama_layer_mlp_up_gate.weight"].t())
        up, gate = torch.chunk(up_gate, 2, dim=-1)
        gate = self.swish(gate)
        swish_out = up * gate
        down = torch.matmul(swish_out, inputs["llama_layer_mlp_down.weight"].t())
        # Tensor shape: [b*s, h]
        mlp_linear_out = torch.matmul(down, inputs["llama_layer_mlp.weight"].t())

        layer_out = mlp_linear_out + attn_linear_out

        return layer_out


class LlamaTorchModel():
    def __init__(self, layer_num, head_num, head_dim, model_name='llama_model'):
        self.layer_num = layer_num
        self.head_num = head_num
        self.head_dim = head_dim
            
        self.layer_list = []
        for i in range(layer_num):
            self.layer_list.append(LlamaTorchLayer(head_num, head_dim, f'{model_name}_layer_{i}'))

    def forward(self, inputs):                
        h = inputs["word_embed.weight"].shape[1]
        seqlen = inputs["input_ids"].shape[0]
        hidden_states = torch.gather(inputs["word_embed.weight"],
            0, inputs["input_ids"].unsqueeze(1).expand(seqlen, h))
        cos = torch.gather(inputs["cos_table"], 0, 
                           inputs["position_ids"].unsqueeze(1).expand(seqlen, self.head_dim))
        sin = torch.gather(inputs["sin_table"], 0, 
                           inputs["position_ids"].unsqueeze(1).expand(seqlen, self.head_dim))
        layer_inputs = {'hidden_states': hidden_states, 'cos': cos, 'sin': sin}
        layer_out = hidden_states
        for i in range(self.layer_num):
            layer_inputs[HIDDEN_STATES] = layer_out
            layer_inputs["llama_layer_norm_1.weight"] = inputs[f'llama_layer_{i}_norm_1.weight']
            layer_inputs["llama_layer_qkv.weight"] = inputs[f'llama_layer_{i}_qkv.weight']
            layer_inputs["llama_layer_atten.weight"] = inputs[f'llama_layer_{i}_atten.weight']
            layer_inputs["llama_layer_norm_2.weight"] = inputs[f'llama_layer_{i}_norm_2.weight']
            layer_inputs["llama_layer_mlp_up_gate.weight"] = inputs[f'llama_layer_{i}_mlp_up_gate.weight']
            layer_inputs["llama_layer_mlp_down.weight"] = inputs[f'llama_layer_{i}_mlp_down.weight']
            layer_inputs["llama_layer_mlp.weight"] = inputs[f'llama_layer_{i}_mlp.weight']
            layer_out = self.layer_list[i].forward(layer_inputs)        
        final_norm = RMSNorm(inputs["final_norm.weight"])
        final_norm_out = final_norm(layer_out)
        model_out = torch.matmul(final_norm_out, inputs["lm_head.weight"].t())

        return model_out


#--------------------------------------------------------------------------------------------
# layer
class LlamaLayer:
    def __init__(self, layer_prefix: str, head_dim, head_num) -> None:
        self.qkv_linear = Linear(f"{layer_prefix}_qkv")
        self.attn_linear = Linear(f"{layer_prefix}_atten")
        self.up_gate = Linear(f"{layer_prefix}_mlp_up_gate")
        self.down = Linear(f"{layer_prefix}_mlp_down")
        self.mlp_linear = Linear(f"{layer_prefix}_mlp")
        self.input_norm = atb_llm.nn.modules.RmsNorm(f"{layer_prefix}_norm_1", 1e-5)
        self.post_norm = atb_llm.nn.modules.RmsNorm(f"{layer_prefix}_norm_2", 1e-5)

        self.head_num = head_num
        self.head_dim = head_dim

    def forward(self, hidden_states, cos_t, sin_t, k_cache, v_cache, slot_mapping, seqlen):
        hidden_states_ = hidden_states.reshape(lambda org_shape: [org_shape[0], org_shape[1]])
        input_norm_out = self.input_norm(hidden_states_)
        qkv = self.qkv_linear(input_norm_out)
        hidden_size = self.head_dim * self.head_num
        q = qkv[:, :hidden_size]
        kv = qkv[:, hidden_size:3 * hidden_size]
        k, v = split(tensor=kv, split_size_or_sections=2, dim=1)
        q_embed, k_embed = atb_llm.nn.functional.rope(q, k, cos_t, sin_t, seqlen)
        q_embed_ = q_embed.reshape(lambda org_shape: [org_shape[0], self.head_num, self.head_dim])
        k_embed_ = k_embed.reshape(lambda org_shape: [org_shape[0], self.head_num, self.head_dim])
        v_ = v.reshape(lambda org_shape: [org_shape[0], self.head_num, self.head_dim])
        reshape_and_cache(k_embed_, v_, k_cache, v_cache, slot_mapping)
        atten_score = F.paged_attention(
            q=q_embed_,
            k=k_embed_,
            v=v_,
            qk_scale=1,
            head_num=self.head_num,
            kv_lens=seqlen
        )
        atten_score_ = atten_score.reshape(lambda org_shape: [org_shape[0], org_shape[1] * org_shape[2]])

        atten_out = self.attn_linear(atten_score_)
        res_add_1 = hidden_states + atten_out

        post_norm_out = self.post_norm(res_add_1)
        up_gate_out = self.up_gate(post_norm_out)
        up, gate = split(tensor=up_gate_out, split_size_or_sections=2, dim=1)
        swi_out = atb_llm.nn.functional.activation(gate, atb_llm.nn.functional.ActType.SWISH)
        down_in = up * swi_out
        mlp_out = self.down(down_in)

        mlp_linear_out = self.mlp_linear(mlp_out)
        layer_out = mlp_linear_out + atten_out

        return layer_out


# model
class LlamaModel:
    def __init__(self, layer_num: int, head_num: int, head_dim, model_prefix: str) -> None:
        self.layers = [LlamaLayer(f"{model_prefix}_layer_{i}", head_dim, head_num) for i in range(layer_num)]
        self.lm_head = Linear("lm_head")
        self.final_norm = atb_llm.nn.modules.RmsNorm("final_norm", 1e-5)
        self.embedding = atb_llm.nn.Parameter(prefix="word_embed", suffix="weight")

        self.layer_num = layer_num
        self.head_num = head_num
        self.head_dim = head_dim

    def forward(self, input_ids, position_ids, cos_table, sin_table, k_cache, v_cache, slot_mapping, seqlen):
        hidden_states = gather(self.embedding.get_tensor(), 0, input_ids, batch_dims=0)
        cos_t = gather(cos_table, 0, position_ids, batch_dims=0)
        sin_t = gather(sin_table, 0, position_ids, batch_dims=0)
        for i in range(self.layer_num):
            hidden_states = self.layers[i].forward(hidden_states, cos_t, sin_t,
                k_cache, v_cache, slot_mapping, seqlen)
        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits


# layer engine
def get_llama_layer_engine(head_dim, head_num):
    hidden_states = Tensor(HIDDEN_STATES)
    cos_t = Tensor("cos")
    sin_t = Tensor("sin")
    k_cache = Tensor("k_cache")
    v_cache = Tensor("v_cache")
    slots_mapping = Tensor("slots_mapping")
    seqlen = Tensor("seq_len")
    llama_layer = LlamaLayer("llama_layer", head_dim, head_num)
    layer_out = llama_layer.forward(hidden_states, cos_t, sin_t, k_cache, v_cache, slots_mapping, seqlen)
    get_default_net().mark_output(layer_out, "layer_out")
    logger.info(get_default_net())
    engine = get_default_net().build_engine()
    logger.info(engine)
    return engine


# model engine
def get_llama_model_engine(layer_num, head_dim, head_num):
    input_ids = Tensor("input_ids")
    position_ids = Tensor("position_ids")
    cos_table = Tensor("cos_table")
    sin_table = Tensor("sin_table")
    k_cache = Tensor("k_cache")
    v_cache = Tensor("v_cache")
    slots_mapping = Tensor("slots_mapping")
    seqlen = Tensor("seq_len")

    llama_model = LlamaModel(layer_num, head_num, head_dim, "llama")
    logits = llama_model.forward(input_ids, position_ids, cos_table, sin_table, k_cache, v_cache, slots_mapping, seqlen)
    get_default_net().mark_output(logits, "logits")
    logger.info(get_default_net())
    engine = get_default_net().build_engine()
    logger.info(engine)
    return engine


def test_llama_layer_engine():
    hn = head_num = 8
    hd = head_dim = 128
    b = 1
    s = 512
    h = hn * hd
    max_s = 1024
    bn = 1024
    bs = 128
    width = 0.2

    llama_layer_weights = {}
    llama_layer_weights["llama_layer_norm_1.weight"] = torch.rand(h).half().npu() * width - width / 2
    llama_layer_weights["llama_layer_qkv.weight"] = torch.rand(3 * h, h).half().npu() * width - width / 2
    llama_layer_weights["llama_layer_norm_2.weight"] = torch.rand(h).half().npu() * width - width / 2
    llama_layer_weights["llama_layer_mlp_up_gate.weight"] = torch.rand(8 * h, h).half().npu() * width - width / 2
    llama_layer_weights["llama_layer_mlp_down.weight"] = torch.rand(h, 4 * h).half().npu() * width - width / 2
    llama_layer_weights["llama_layer_mlp.weight"] = torch.rand(h, h).half().npu() * width - width / 2
    llama_layer_weights["llama_layer_atten.weight"] = torch.rand(h, h).half().npu() * width - width / 2

    
    llama_layer_inputs = {}
    llama_layer_inputs[HIDDEN_STATES] = torch.rand(b * s, h).half().npu() * width - width / 2
    llama_layer_inputs["cos"] = torch.rand(max_s, hd).half().npu() * width - width / 2
    llama_layer_inputs["sin"] = torch.rand(max_s, hd).half().npu() * width - width / 2

    k_cache = torch.zeros(bn, bs, hn, hd).half().npu()
    v_cache = torch.zeros(bn, bs, hn, hd).half().npu()
    if torch_npu._C._npu_get_soc_version() in NO_BF16_SUPPORT_SOC_VERSION:
        k_cache = k_cache.reshape(bn, bs, hn * hd // 16, 16).permute(0, 2, 1, 3).contiguous()
        torch_npu.npu_format_cast_(k_cache, 29)
        v_cache = v_cache.reshape(bn, bs, hn * hd // 16, 16).permute(0, 2, 1, 3).contiguous()
        torch_npu.npu_format_cast_(v_cache, 29)
    llama_layer_inputs["k_cache"] = k_cache
    llama_layer_inputs["v_cache"] = k_cache
    llama_layer_inputs["slots_mapping"] = torch.zeros(b * s, dtype=torch.int).npu()
    seqlen = torch.ones(b, dtype=torch.int) * s     # host tensor
    llama_layer_inputs["seq_len"] = seqlen.npu()    # device tensor

    llama_layer_outputs = {}
    llama_layer_outputs["layer_out"] = torch.ones(b * s, h).half().npu()

    bind_map = {}
    bind_map['seq_len'] = seqlen

    engine = get_llama_layer_engine(head_dim, head_num)
    engine.set_weights(llama_layer_weights)
    engine.forward(llama_layer_inputs, llama_layer_outputs, bind_map)
    
    llama_torch_layer = LlamaTorchLayer(head_num=head_num, head_dim=head_dim, op_name='llama_layer')
    llama_torch_layer_out = llama_torch_layer.forward({**llama_layer_inputs, **llama_layer_weights})

    rt = torch.allclose(llama_torch_layer_out, llama_layer_outputs["layer_out"], rtol=1e-03, atol=1e-03)
    logger.info('\nTest Llama layer precision: %s\n', rt)


def test_llama_model_engine():
    hn = head_num = 8
    hd = head_dim = 128
    b = 1
    s = 512
    h = hn * hd
    max_s = 1024
    bn = 1024
    bs = 128
    layer_num = 30
    vocab_size = 12800
    width = 0.2
    
    llama_model_weights = {}
    llama_model_weights["word_embed.weight"] = torch.rand(vocab_size, h).half().npu() * width - width / 2
    for i in range(layer_num):
        llama_model_weights[f'llama_layer_{i}_norm_1.weight'] = torch.rand(h).half().npu() * width - width / 2
        llama_model_weights[f'llama_layer_{i}_qkv.weight'] = \
            torch.rand(3 * h, h).half().npu() * width - width / 2
        llama_model_weights[f'llama_layer_{i}_norm_2.weight'] = torch.rand(h).half().npu() * width - width / 2
        llama_model_weights[f'llama_layer_{i}_mlp_up_gate.weight'] = (torch.rand(8 * h, h).
                                                                       half().npu() * width - width / 2)
        llama_model_weights[f'llama_layer_{i}_mlp_down.weight'] = (torch.rand(h, 4 * h).
                                                                         half().npu() * width - width / 2)
        llama_model_weights[f'llama_layer_{i}_mlp.weight'] = torch.rand(h, h).half().npu() * width - width / 2
        llama_model_weights[f'llama_layer_{i}_atten.weight'] = torch.rand(h, h).half().npu() * width - width / 2
    llama_model_weights["final_norm.weight"] = torch.rand(h).half().npu() * width - width / 2
    llama_model_weights["lm_head.weight"] = torch.rand(vocab_size, h).half().npu() * width - width / 2

    llama_model_inputs = {}
    llama_model_inputs["input_ids"] = torch.arange(s).npu()
    llama_model_inputs["position_ids"] = torch.arange(s).npu()
    llama_model_inputs["cos_table"] = torch.rand(max_s, hd).half().npu() * width - width / 2
    llama_model_inputs["sin_table"] = torch.rand(max_s, hd).half().npu() * width - width / 2

    k_cache = torch.zeros(bn, bs, hn, hd).half().npu()
    v_cache = torch.zeros(bn, bs, hn, hd).half().npu()
    if torch_npu._C._npu_get_soc_version() in NO_BF16_SUPPORT_SOC_VERSION:
        k_cache = k_cache.reshape(bn, bs, hn * hd // 16, 16).permute(0, 2, 1, 3).contiguous()
        torch_npu.npu_format_cast_(k_cache, 29)
        v_cache = v_cache.reshape(bn, bs, hn * hd // 16, 16).permute(0, 2, 1, 3).contiguous()
        torch_npu.npu_format_cast_(v_cache, 29)

    llama_model_inputs["k_cache"] = k_cache
    llama_model_inputs["v_cache"] = v_cache
    llama_model_inputs["slots_mapping"] = torch.zeros(b * s, dtype=torch.int).npu()
    seqlen = torch.ones(b, dtype=torch.int) * s     # host tensor
    llama_model_inputs["seq_len"] = seqlen.npu()    # device tensor

    llama_model_outputs = {}
    llama_model_outputs["logits"] = torch.ones(b * s, vocab_size).half().npu()

    bind_map = {}
    bind_map["seq_len"] = seqlen

    engine = get_llama_model_engine(layer_num, head_dim, head_num)
    engine.set_weights(llama_model_weights)
    engine.forward(llama_model_inputs, llama_model_outputs, bind_map)

    llama_torch_model = LlamaTorchModel(layer_num=layer_num,
                                    head_num=head_num, head_dim=head_dim, model_name='llama_model')
    llama_torch_model_out = llama_torch_model.forward({**llama_model_inputs, **llama_model_weights})

    # 精度比较
    rt = torch.allclose(llama_torch_model_out, llama_model_outputs["logits"], rtol=1e-02, atol=1e-02)
    logger.info('\nTest Llama model precision: %s\n', rt)


class TestNetworkFunction(unittest.TestCase):
    def test_llama_layer(self):
        test_llama_layer_engine()

    def test_llama_model(self):
        test_llama_model_engine()


if __name__ == '__main__':
    unittest.main()