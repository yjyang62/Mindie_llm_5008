/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#ifndef ATB_SPEED_MODELS_LLAMA_DECODER_LAYER_H
#define ATB_SPEED_MODELS_LLAMA_DECODER_LAYER_H

#include <vector>
#include "nlohmann/json.hpp"

#include "atb/atb_infer.h"
#include "atb_speed/log.h"

#include "operations/fusion/linear/linear_parallel.h"

namespace atb_speed {
namespace llama {
enum PositionEmbeddingType : uint32_t {
    ROPE = 0,
    ALIBI,
};

struct DecoderLayerParam {
    bool isEdgeHardware = true;
    bool isFA = true;
    bool isBF16 = false;
    bool isPack = true;
    bool isPrefill = true;
    bool supportSwiGLU = false;
    bool isQuant = false;
    bool supportLcoc = false;
    bool supportCompressHead = false;
    bool enableAddNorm = false;
    int quantType = 0;
    int attnBackend = 0;
    int quantGroupSize = 64;
    int numAttentionHeadsPerRank = 0;
    int hiddenSizePerAttentionHead = 0;
    int numKeyValueHeadsPerRank = 0;
    int numHiddenLayers = 28;
    float rmsNormEps = 0;
    int hiddenSize = 0;
    int seqLength = 1;
    PositionEmbeddingType positionEmbeddingType = ROPE;
    atb_speed::common::TensorParallelInfo tensorParallelInfo;
    std::vector<int> tokenOffset;
    std::vector<int> packQuantType = {};  // 两个元素，第一个元素代表QKV pack的量化类型，第二个元素代表MLP pack的量化类型
    // 七个元素，分别代表q，k，v，self attention out，gate，up，down linear的类型
    std::vector<int> linearQuantType = {};
    // 四个元素，分为表示Attn和mlp是否有bias: qkvHasBias,selfAttnHasBias,gateUpHasBias,downHasBias
    std::vector<bool> linearHasBias = {false, false, false, false};
    std::vector<int> linearTransposeType;
};
enum HasBias : uint32_t {
    // linearHasBias的四个元素
    QKV_HASBIAS = 0,
    SELFATTENTION_HASBIAS,
    GATEUP_HASBIAS,
    DOWN_HASBIAS,
};

atb::Status DecoderLayer(const DecoderLayerParam &param, atb::Operation **operation);

}  // namespace llama
}  // namespace atb_speed
#endif