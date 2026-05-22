# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import os
import sys
import logging
import json
import torch
import torch_npu
import torch.nn as nn

path = os.getenv('ATB_SPEED_HOME_PATH')
sys.path.append(os.path.join(path, 'lib'))
import _libatb_torch as atb  # noqa: E402

torch_npu.npu.set_device(0)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# =============================================================================================
# 以下用作 Golden，校验精度
# =============================================================================================


class TensorName:
    hidden_states = 'hidden_states'
    norm_weight_1 = 'norm_weight_1'
    norm_weight_2 = 'norm_weight_2'
    atten_weight = 'atten_weight'
    qkv_weight = 'qkv_weight'
    qkv = 'qkv'        
    mlp_up_gate_weight = 'mlp_up_gate_weight'
    mlp_down_weight = 'mlp_down_weight'
    mlp_weight = 'mlp_weight'
    position_ids = 'position_ids'
    input_ids = 'input_ids'
    word_embed_weight = 'word_embed_weight'
    cos = 'cos'
    sin = 'sin'
    cos_table = 'cos_table'
    sin_table = 'sin_table'      
    k_cache = 'k_cache'
    v_cache = 'v_cache'
    slots_mapping = 'slots_mapping'
    seq_len = 'seq_len'
    layer_out = 'layer_out'
    norm1_out = 'norm1_out'   
    q = 'q'
    k = 'k'
    v = 'v'
    v_reshape = 'v_reshape'    
    q_embed = 'q_embed'
    q_embed_reshape = 'q_embed_reshape'
    k_embed = 'k_embed'
    k_embed_reshape = 'k_embed_reshape'
    atten_out = 'atten_out'
    atten_out_reshape = 'atten_out_reshape'
    atten_linear_out = 'atten_linear_out'
    atten_res_add_out = 'atten_res_add_out'
    norm2_out = 'norm2_out'
    up_gate_out = 'up_gate_out'
    up_out = 'up_out'
    gate_out = 'gate_out'
    swish_out = 'swish_out'
    mlp_out = 'mlp_out'
    mlp_linear_out = 'mlp_linear_out'
    final_norm_weight = 'final_norm_weight'
    lm_head_weight = 'lm_head_weight'
    model_out = 'model_out'
    norm_weight_1_layer = 'norm_weight_1_layer'
    qkv_weight_layer = 'qkv_weight_layer'
    atten_weight_layer = 'atten_weight_layer'
    norm_weight_2_layer = 'norm_weight_2_layer'
    mlp_up_gate_weight_layer = 'mlp_up_gate_weight_layer'
    mlp_down_weight_layer = 'mlp_down_weight_layer'
    mlp_weight_layer = 'mlp_weight_layer'


class OperationType:
    RmsNorm = 'RmsNorm'
    Linear = 'Linear'
    Split = 'Split'
    Rope = 'Rope'
    ReshapeAndCache = 'ReshapeAndCache'
    SelfAttention = 'SelfAttention'
    Elewise = 'Elewise'
    Activation = 'Activation'
    Gather = 'Gather'
    

class Param:
    hasBias = 'hasBias' 
    elewiseType = 'elewiseType'   
        

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
        self.op_name = 'llama_layer'

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
        bs = 1
        seqlen = inputs[TensorName.hidden_states].shape[0] // bs
        rms_norm_1 = RMSNorm(inputs[TensorName.norm_weight_1])
        norm1_out = rms_norm_1(inputs[TensorName.hidden_states])
        qkv = torch.matmul(norm1_out, inputs[TensorName.qkv_weight].t())
        # Tensor shape: [b*s, h]
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # Tensor shape: [b, s, hn, hd]
        q = q.view(1, q.shape[0], self.head_num, self.head_dim)
        k = k.view(1, k.shape[0], self.head_num, self.head_dim)
        q_embed, k_embed = self.rope(q, k, inputs[TensorName.cos][:seqlen, :], inputs[TensorName.sin][:seqlen, :])
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
        attn_linear_out = torch.matmul(attn_output, inputs[TensorName.atten_weight].t())

        res_add_out = attn_linear_out + inputs[TensorName.hidden_states]
        rms_norm_2 = RMSNorm(inputs[TensorName.norm_weight_2])
        norm1_out = rms_norm_2(res_add_out)

        # Tensor shape: [b*s, 2*h]
        up_gate = torch.matmul(norm1_out, inputs[TensorName.mlp_up_gate_weight].t())
        up, gate = torch.chunk(up_gate, 2, dim=-1)
        gate = self.swish(gate)
        swish_out = up * gate
        down = torch.matmul(swish_out, inputs[TensorName.mlp_down_weight].t())
        # Tensor shape: [b*s, h]
        mlp_linear_out = torch.matmul(down, inputs[TensorName.mlp_weight].t())

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
        h = inputs[TensorName.word_embed_weight].shape[1]
        seqlen = inputs[TensorName.input_ids].shape[0]
        hidden_states = torch.gather(inputs[TensorName.word_embed_weight],
            0, inputs[TensorName.input_ids].unsqueeze(1).expand(seqlen, h))
        cos = torch.gather(inputs[TensorName.cos_table], 0, 
                           inputs[TensorName.position_ids].unsqueeze(1).expand(seqlen, self.head_dim))
        sin = torch.gather(inputs[TensorName.sin_table], 0, 
                           inputs[TensorName.position_ids].unsqueeze(1).expand(seqlen, self.head_dim))
        layer_inputs = {'hidden_states': hidden_states, 'cos': cos, 'sin': sin}
        layer_out = hidden_states
        for i in range(self.layer_num):
            layer_inputs[TensorName.hidden_states] = layer_out
            layer_inputs[TensorName.norm_weight_1] = inputs[f'{TensorName.norm_weight_1_layer}_{i}']
            layer_inputs[TensorName.qkv_weight] = inputs[f'{TensorName.qkv_weight_layer}_{i}']
            layer_inputs[TensorName.atten_weight] = inputs[f'{TensorName.atten_weight_layer}_{i}']
            layer_inputs[TensorName.norm_weight_2] = inputs[f'{TensorName.norm_weight_2_layer}_{i}']
            layer_inputs[TensorName.mlp_up_gate_weight] = inputs[f'{TensorName.mlp_up_gate_weight_layer}_{i}']
            layer_inputs[TensorName.mlp_down_weight] = inputs[f'{TensorName.mlp_down_weight_layer}_{i}']
            layer_inputs[TensorName.mlp_weight] = inputs[f'{TensorName.mlp_weight_layer}_{i}']
            layer_out = self.layer_list[i].forward(layer_inputs)        
        final_norm = RMSNorm(inputs[TensorName.final_norm_weight])
        final_norm_out = final_norm(layer_out)
        model_out = torch.matmul(final_norm_out, inputs[TensorName.lm_head_weight].t())

        return model_out


# ==============================================================================================================
# 注意：以下模型非标准，仅作使用新版ATB组图接口的demo, 为了方便，
# 这里使用了 glm 系列的 attention mask，在prefill阶段无需mask
# ==============================================================================================================

# 0.确保ATB_SPEED_HOME_PATH环境变量存在且有效，再添加搜索路径，即可 import _libatb_torch
# 1.需要继承 atb.GraphOperation
class LlamaAtbLayer(atb.GraphOperation):
    def __init__(self, head_num, head_dim, op_name='llama_layer'):
        super().__init__(op_name)

        # 按需自定义reshape function
        def reshape_qkv(org_shape):
            return [org_shape[0], head_num, head_dim]
        
        def reshape_0_12(org_shape):
            return [org_shape[0], org_shape[1] * org_shape[2]]

        # 2.初始化需要的operation对象, 这些 BaseOperation 注册在 operation_register.cpp 文件中
        self.input_norm = atb.BaseOperation(op_type=OperationType.RmsNorm,
                                            op_param=json.dumps({'layerType': 'RMS_NORM_NORM'}), op_name='input_norm')
        self.qkv_linear = atb.BaseOperation(op_type=OperationType.Linear,
                                            op_param=json.dumps({Param.hasBias: False}), op_name='qkv_linear')
        self.qkv_split = atb.BaseOperation(op_type=OperationType.Split,
                                            op_param=json.dumps({'splitDim': 1, 'splitNum': 3}), op_name='qkv_split')
        self.rope = atb.BaseOperation(op_type=OperationType.Rope, 
                                       op_param=json.dumps({'rotaryCoeff': 2}), op_name='rope')
        self.reshape_and_cache = atb.BaseOperation(op_type="ReshapeAndCache",
                                            op_param=json.dumps({}), op_name='reshape_and_cache')
        self.attention = atb.BaseOperation(op_type=OperationType.SelfAttention,
                                           op_param=json.dumps({'headNum': head_num,
                                                                'kvHeadNum': head_num,
                                                                'calcType': 'PA_ENCODER'}),
                                           op_name='attention')
        self.atten_linear = atb.BaseOperation(op_type=OperationType.Linear,
                                            op_param=json.dumps({Param.hasBias: False}), op_name='atten_linear')
        self.atten_res_add = atb.BaseOperation(op_type=OperationType.Elewise,
                                                op_param=json.dumps({Param.elewiseType: 'ELEWISE_ADD'}),
                                                op_name='atten_res_add')
        self.post_norm = atb.BaseOperation(op_type=OperationType.RmsNorm,
                                            op_param=json.dumps({'layerType': 'RMS_NORM_NORM'}), op_name='post_norm')
        self.up_gate = atb.BaseOperation(op_type=OperationType.Linear, 
                                          op_param=json.dumps({Param.hasBias: False}), op_name='up_gate')
        self.up_gate_split = atb.BaseOperation(op_type=OperationType.Split,
                                                op_param=json.dumps({'splitDim': 1, 'splitNum': 2}),
                                                op_name='up_gate_split')
        self.swish = atb.BaseOperation(op_type=OperationType.Activation,
                                        op_param=json.dumps({'activationType': 'ACTIVATION_SWISH'}),
                                        op_name='swish')
        self.mul = atb.BaseOperation(op_type=OperationType.Elewise,
                                    op_param=json.dumps({Param.elewiseType: 'ELEWISE_MUL'}), op_name='mul')
        self.down = atb.BaseOperation(op_type=OperationType.Linear, 
                                       op_param=json.dumps({Param.hasBias: False}), op_name='down')
        self.mlp_linear = atb.BaseOperation(op_type=OperationType.Linear,
                                            op_param=json.dumps({Param.hasBias: False}), op_name='mlp_linear')
        self.mlp_res_add = atb.BaseOperation(op_type=OperationType.Elewise,
                                            op_param=json.dumps({Param.elewiseType: 'ELEWISE_ADD'}),
                                            op_name='mlp_res_add')
              
        # 3.设置输出、输出 tensor, 中间tensor可以使用时定义，不需要预先定义
        in_tensors = [
            TensorName.hidden_states,
            TensorName.norm_weight_1, TensorName.qkv_weight, 
            TensorName.cos, TensorName.sin,
            TensorName.k_cache, TensorName.v_cache, 
            TensorName.slots_mapping, TensorName.seq_len, TensorName.atten_weight,
            TensorName.norm_weight_2, TensorName.mlp_up_gate_weight, 
            TensorName.mlp_down_weight, TensorName.mlp_weight
        ]
        out_tensors = [TensorName.layer_out]
        self.add_input_output(input=in_tensors, output=out_tensors)

        # 4.定义计算过程, 每个operation节点只能被添加一次,注释表示输入输出shape变化，可以根据op的需要对op的输入进行reshape
        # [b*s,h][h]->[b*s,h]
        self.add_operation(self.input_norm, 
                           [TensorName.hidden_states, TensorName.norm_weight_1], [TensorName.norm1_out])
        # [b*s,h][3*h,h]->[b*s,3*h]
        self.add_operation(self.qkv_linear, [TensorName.norm1_out, TensorName.qkv_weight], [TensorName.qkv])
        # [b*s,3*h]->[b*s,h][b*s,h][b*s,h]
        self.add_operation(self.qkv_split, [TensorName.qkv], [TensorName.q, TensorName.k, TensorName.v])
        # [b*s,h][b*s,h][max_s,hd][max_s,hd][b*s]->[b*s,h][b*s,h]
        self.add_operation(self.rope, [TensorName.q, TensorName.k, TensorName.cos, TensorName.sin, TensorName.seq_len], 
                           [TensorName.q_embed, TensorName.k_embed])
        # [b*s,h]->[b*s,hn,hd]
        self.add_reshape(TensorName.q_embed, TensorName.q_embed_reshape, reshape_qkv)
        # [b*s,h]->[b*s,hn,hd]
        self.add_reshape(TensorName.k_embed, TensorName.k_embed_reshape, reshape_qkv)
        # [b*s,h]->[b*s,hn,hd]
        self.add_reshape(TensorName.v, TensorName.v_reshape, reshape_qkv)
        # [b*s,hn,hd][b*s,hn,hd][bn,bs,hn,hd][bn,bs,hn,hd][b*s]->[bn,bs,hn,hd][bn,bs,hn,hd]
        self.add_operation(self.reshape_and_cache,
                            [TensorName.k_embed_reshape, TensorName.v_reshape, 
                             TensorName.k_cache, TensorName.v_cache, TensorName.slots_mapping],
                            [TensorName.k_cache, TensorName.v_cache])
        # [b*s,h][b*s,h][b*s,h][max_s,max_s][b]->[b*s,h]
        self.add_operation(self.attention, [TensorName.q_embed_reshape, 
                                            TensorName.k_embed_reshape, TensorName.v_reshape, TensorName.seq_len], 
                           [TensorName.atten_out])
        # [b*s,hn,hd]->[b*s,h]
        self.add_reshape(TensorName.atten_out, TensorName.atten_out_reshape, reshape_0_12)
        # [b*s,h][h,h]->[b*s,h]
        self.add_operation(self.atten_linear, [TensorName.atten_out_reshape, TensorName.atten_weight], 
                           [TensorName.atten_linear_out])
        # [b*s,h][b*s,h]->[b*s,h]
        self.add_operation(self.atten_res_add, [TensorName.hidden_states, TensorName.atten_linear_out], 
                           [TensorName.atten_res_add_out])
        # [b*s,h][h]->[b*s,h]
        self.add_operation(self.post_norm, [TensorName.atten_res_add_out, TensorName.norm_weight_2], 
                           [TensorName.norm2_out])
        # [b*s,h][8*h,h]->[b*s,8*h]
        self.add_operation(self.up_gate, [TensorName.norm2_out, TensorName.mlp_up_gate_weight], 
                           [TensorName.up_gate_out])
        # [b*s,8*h]->[b*s,4*h][b*s,4*h]
        self.add_operation(self.up_gate_split, [TensorName.up_gate_out], [TensorName.up_out, TensorName.gate_out])
        # [b*s,4*h]->[b*s,4*h]
        self.add_operation(self.swish, [TensorName.gate_out], [TensorName.swish_out])
        # [b*s,4*h][b*s,4*h]->[b*s,4*h]
        self.add_operation(self.mul, [TensorName.up_out, TensorName.swish_out], [TensorName.swish_out])
        # [b*s,4*h][h,4*h]->[b,s,h]
        self.add_operation(self.down, [TensorName.swish_out, TensorName.mlp_down_weight], 
                           [TensorName.mlp_out])
        # [b*s,h][h,h]->[b*s,h]
        self.add_operation(self.mlp_linear, [TensorName.mlp_out, TensorName.mlp_weight], 
                           [TensorName.mlp_linear_out])
        # [b*s,h][b*s,h]->[b*s,h]
        self.add_operation(self.mlp_res_add, [TensorName.atten_linear_out, TensorName.mlp_linear_out], 
                           [TensorName.layer_out])

        # 5.图构建
        self.build()


class LlamaAtbModel(atb.GraphOperation):
    def __init__(self, layer_num, head_num, head_dim, model_name='llama_model'):
        super().__init__(model_name)

        # operation初始化
        self.word_embedding = atb.BaseOperation(op_type=OperationType.Gather, 
                                                 op_param=json.dumps({}), op_name='word_embedding')
        self.gather_cos = atb.BaseOperation(op_type=OperationType.Gather, 
                                             op_param=json.dumps({}), op_name='gather_cos')
        self.gather_sin = atb.BaseOperation(op_type=OperationType.Gather, 
                                             op_param=json.dumps({}), op_name='gather_sin')
        self.layer_list = []
        for i in range(layer_num):
            self.layer_list.append(LlamaAtbLayer(head_num, head_dim, f'{model_name}_layer_{i}'))
        self.final_norm = atb.BaseOperation(op_type=OperationType.RmsNorm,
                                            op_param=json.dumps({'layerType': 'RMS_NORM_NORM'}), op_name='final_norm')
        self.lm_head = atb.BaseOperation(op_type=OperationType.Linear, 
                                          op_param=json.dumps({Param.hasBias: False}), op_name='lm_head')

        
        # 定义输入、输出 tensors
        in_tensors = [
            TensorName.input_ids, TensorName.position_ids, TensorName.cos_table, TensorName.sin_table,
            TensorName.k_cache, TensorName.v_cache, TensorName.slots_mapping, TensorName.seq_len
        ]
        in_tensors.append(TensorName.word_embed_weight)
        for i in range(layer_num):            
            in_tensors.append(f'{TensorName.norm_weight_1_layer}_{i}')
            in_tensors.append(f'{TensorName.qkv_weight_layer}_{i}')
            in_tensors.append(f'{TensorName.atten_weight_layer}_{i}')
            in_tensors.append(f'{TensorName.norm_weight_2_layer}_{i}')
            in_tensors.append(f'{TensorName.mlp_up_gate_weight_layer}_{i}')
            in_tensors.append(f'{TensorName.mlp_down_weight_layer}_{i}')
            in_tensors.append(f'{TensorName.mlp_weight_layer}_{i}')
        in_tensors.append(TensorName.final_norm_weight)
        in_tensors.append(TensorName.lm_head_weight)
        out_tensors = [TensorName.model_out]
        self.add_input_output(input=in_tensors, output=out_tensors)

        # # 定义计算过程
        self.add_operation(self.word_embedding, 
                           [TensorName.word_embed_weight, TensorName.input_ids], [TensorName.hidden_states])
        self.add_operation(self.gather_cos, [TensorName.cos_table, TensorName.position_ids], [TensorName.cos])
        self.add_operation(self.gather_sin, [TensorName.sin_table, TensorName.position_ids], [TensorName.sin])
        for i in range(layer_num):            
            self.add_operation(self.layer_list[i],
                               [TensorName.hidden_states, f'{TensorName.norm_weight_1_layer}_{i}', 
                                f'{TensorName.qkv_weight_layer}_{i}', TensorName.cos,
                               TensorName.sin, TensorName.k_cache, TensorName.v_cache, TensorName.slots_mapping, 
                               TensorName.seq_len, f'{TensorName.atten_weight_layer}_{i}',
                               f'{TensorName.norm_weight_2_layer}_{i}', 
                               f'{TensorName.mlp_up_gate_weight_layer}_{i}',
                               f'{TensorName.mlp_down_weight_layer}_{i}', f'{TensorName.mlp_weight_layer}_{i}'],
                               [TensorName.hidden_states])
        self.add_operation(self.final_norm, 
                           [TensorName.hidden_states, TensorName.final_norm_weight], [TensorName.hidden_states])
        self.add_operation(self.lm_head, 
                           [TensorName.hidden_states, TensorName.lm_head_weight], [TensorName.model_out])
        # 图构建
        # model作为一个整图太大，需要切图，进行图流水(将model的次一级节点分解为多个子图来执行)，需要将如下配置为 false
        self.execute_as_single = False
        self.build()


# ==========================================================================================================
# TEST Layer
# ==========================================================================================================
def test_llama_atb_layer():
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
    llama_layer_weights[TensorName.norm_weight_1] = torch.rand(h).half().npu() * width - width / 2
    llama_layer_weights[TensorName.qkv_weight] = torch.rand(3 * h, h).half().npu() * width - width / 2
    llama_layer_weights[TensorName.norm_weight_2] = torch.rand(h).half().npu() * width - width / 2
    llama_layer_weights[TensorName.mlp_up_gate_weight] = torch.rand(8 * h, h).half().npu() * width - width / 2
    llama_layer_weights[TensorName.mlp_down_weight] = torch.rand(h, 4 * h).half().npu() * width - width / 2
    llama_layer_weights[TensorName.mlp_weight] = torch.rand(h, h).half().npu() * width - width / 2
    llama_layer_weights[TensorName.atten_weight] = torch.rand(h, h).half().npu() * width - width / 2

    
    # 6.输入构造，传入一个字典即可，不用按序排列，key需要和图中定义保持一致
    llama_layer_inputs = {}
    llama_layer_inputs[TensorName.hidden_states] = torch.rand(b * s, h).half().npu() * width - width / 2
    llama_layer_inputs[TensorName.cos] = torch.rand(max_s, hd).half().npu() * width - width / 2
    llama_layer_inputs[TensorName.sin] = torch.rand(max_s, hd).half().npu() * width - width / 2
    llama_layer_inputs[TensorName.k_cache] = torch.zeros(bn, bs, hn, hd).half().npu()
    llama_layer_inputs[TensorName.v_cache] = torch.zeros(bn, bs, hn, hd).half().npu()
    llama_layer_inputs[TensorName.slots_mapping] = torch.zeros(b * s, dtype=torch.int).npu()
    seqlen = torch.ones(b, dtype=torch.int) * s     # host tensor
    llama_layer_inputs[TensorName.seq_len] = seqlen.npu()    # device tensor

    llama_layer_outputs = {}
    llama_layer_outputs[TensorName.layer_out] = torch.ones(b * s, h).half().npu()

    bind_map = {}
    bind_map['seq_len'] = seqlen

    llama_atb_layer = LlamaAtbLayer(head_num=head_num, head_dim=head_dim, op_name='llama_layer')
    llama_atb_layer.set_weights(llama_layer_weights)
    # 7.模型执行(inputs、outputs、bind_map)
    llama_atb_layer.forward(llama_layer_inputs, llama_layer_outputs, bind_map)

    # 8.执行golden
    llama_torch_layer = LlamaTorchLayer(head_num=head_num, head_dim=head_dim, op_name='llama_layer')
    llama_torch_layer_out = llama_torch_layer.forward({**llama_layer_inputs, **llama_layer_weights})

    # 9.精度比较
    rt = torch.allclose(llama_torch_layer_out, llama_layer_outputs[TensorName.layer_out], rtol=1e-03, atol=1e-03)
    logger.info('\nTest Llama layer precision: %s\n', rt)


# ===============================================================================================================
# TEST Model
# ===============================================================================================================
def test_llama_atb_model():
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
    
    # 输入构造，传入一个字典即可，不用按序排列，key需要和图中定义保持一致
    llama_model_weights = {}
    llama_model_weights[TensorName.word_embed_weight] = torch.rand(vocab_size, h).half().npu() * width - width / 2
    for i in range(layer_num):
        llama_model_weights[f'{TensorName.norm_weight_1_layer}_{i}'] = torch.rand(h).half().npu() * width - width / 2
        llama_model_weights[f'{TensorName.qkv_weight_layer}_{i}'] = \
            torch.rand(3 * h, h).half().npu() * width - width / 2
        llama_model_weights[f'{TensorName.norm_weight_2_layer}_{i}'] = torch.rand(h).half().npu() * width - width / 2
        llama_model_weights[f'{TensorName.mlp_up_gate_weight_layer}_{i}'] = (torch.rand(8 * h, h).
                                                                       half().npu() * width - width / 2)
        llama_model_weights[f'{TensorName.mlp_down_weight_layer}_{i}'] = (torch.rand(h, 4 * h).
                                                                         half().npu() * width - width / 2)
        llama_model_weights[f'{TensorName.mlp_weight_layer}_{i}'] = torch.rand(h, h).half().npu() * width - width / 2
        llama_model_weights[f'{TensorName.atten_weight_layer}_{i}'] = torch.rand(h, h).half().npu() * width - width / 2
    llama_model_weights[TensorName.final_norm_weight] = torch.rand(h).half().npu() * width - width / 2
    llama_model_weights[TensorName.lm_head_weight] = torch.rand(vocab_size, h).half().npu() * width - width / 2


    llama_model_inputs = {}
    llama_model_inputs[TensorName.input_ids] = torch.arange(s).npu()
    llama_model_inputs[TensorName.position_ids] = torch.arange(s).npu()
    llama_model_inputs[TensorName.cos_table] = torch.rand(max_s, hd).half().npu() * width - width / 2
    llama_model_inputs[TensorName.sin_table] = torch.rand(max_s, hd).half().npu() * width - width / 2
    llama_model_inputs[TensorName.k_cache] = torch.zeros(bn, bs, hn, hd).half().npu()
    llama_model_inputs[TensorName.v_cache] = torch.zeros(bn, bs, hn, hd).half().npu()
    llama_model_inputs[TensorName.slots_mapping] = torch.zeros(b * s, dtype=torch.int).npu()
    seqlen = torch.ones(b, dtype=torch.int) * s     # host tensor
    llama_model_inputs[TensorName.seq_len] = seqlen.npu()    # device tensor

    llama_model_outputs = {}
    llama_model_outputs[TensorName.model_out] = torch.ones(b * s, vocab_size).half().npu()

    bind_map = {}
    bind_map[TensorName.seq_len] = seqlen

    llama_atb_model = LlamaAtbModel(layer_num=layer_num,
                                    head_num=head_num, head_dim=head_dim, model_name='llama_model')
    llama_atb_model.set_weights(llama_model_weights)
    llama_atb_model.forward(llama_model_inputs, llama_model_outputs, bind_map)

    llama_torch_model = LlamaTorchModel(layer_num=layer_num,
                                    head_num=head_num, head_dim=head_dim, model_name='llama_model')
    llama_torch_model_out = llama_torch_model.forward({**llama_model_inputs, **llama_model_weights})

    # 精度比较
    rt = torch.allclose(llama_torch_model_out, llama_model_outputs[TensorName.model_out], rtol=1e-02, atol=1e-02)
    logger.info('\nTest Llama model precision: %s\n', rt)


# 测试总入口
test_llama_atb_layer()
test_llama_atb_model()