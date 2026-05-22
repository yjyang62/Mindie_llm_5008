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

#ifndef ATB_SPEED_MODELS_QWEN_DECODER_LAYER_H
#define ATB_SPEED_MODELS_QWEN_DECODER_LAYER_H

#include <vector>
#include "nlohmann/json.hpp"

#include "atb/atb_infer.h"
#include "atb_speed/base/hosttensor_binder.h"
#include "atb_speed/log.h"

namespace atb_speed {
namespace qwen {
struct DecoderLayerParam {
    bool isFA = false;
    bool isPrefill = false;
    bool isBF16 = false;
    bool isPack = true;
    bool supportSwiGLU = false;
    bool supportLcoc = false;
    bool supportSpeculate = false;
    bool enableSplitFuse = false;
    bool supportLora = false;
    bool loraEnableGMM = false;
    bool enableLogN = false;
    bool enableQScale = false;
    bool enableFA3 = false;
    bool kvQuant = false;
    bool isEmbedding = false;
    std::string backend = "hccl";
    bool enableAddNorm = false;
    int rank = 0;
    int worldSize = 1;
    int quantType = 0;
    int quantGroupSize = 64;
    int numAttentionHeadsPerRank = 0;
    int hiddenSizePerAttentionHead = 0;
    int numKeyValueHeadsPerRank = 0;
    float rmsNormEps = 0;
    float normEps = 1e-6;
    float scaleDepth = 0;
    int layerId = 0;
    int numHiddenLayers = 28;
    int numAttentionHeads = 12;
    int numKeyValueHeads = 2;
    int hiddenSize = 1536;
    int seqLength = 1;
    bool isEdgeHardware = true;
    bool isQuant = false;
    bool useQKNorm = false;
    std::vector<int> seqLen;
    std::vector<int> tokenOffset;
    std::vector<int> packQuantType = {};  // 两个元素，第一个元素代表QKV pack的量化类型，第二个元素代表MLP pack的量化类型
    // 七个元素，分别代表q，k，v，self attention out，gate，up，down linear的类型
    std::vector<int> linearQuantType = {};
    std::vector<int> linearTransposeType;
};


atb::Status DecoderLayer(const DecoderLayerParam &param, atb::Operation **operation);
}  // namespace qwen
}  // namespace atb_speed
#endif