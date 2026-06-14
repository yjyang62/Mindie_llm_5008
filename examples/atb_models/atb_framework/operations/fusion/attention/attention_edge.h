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

#ifndef ATB_SPEED_MODELS_ATTENTION_H
#define ATB_SPEED_MODELS_ATTENTION_H

#include "atb/atb_infer.h"
#include "atb_speed/utils/operation_util.h"

namespace atb_speed {
namespace common {

struct AttentionParam {
    float normEps = 0; /// The epsilon used by the layer normalization layers.
    int layerId = 0;  /// The current layer Id.
    int numHiddenLayers = 0;  /// The number of hidden layers.
    int numAttentionHeads = 8;  /// The number of attention heads.
    int numKeyValueHeads = 1;  /// The number of key/value heads.
    int hiddenSize = 0;  /// The size of hidden layers.
    int seqLength = 1;  // The input sequence length.
    bool isPrefill = false;  // A flag indicating whether the  prefill phase.
    bool isGQA = false;  /// A flag indicating whether attention type is GQA.
    bool isQuant = false;  /// A flag indicating whether quantified or not.
    bool isHasQKVBias = false; /// A flag indicating whether qkv has bias or not.
    bool useQKNorm = false;
    int hiddenSizePerAttentionHead = 0;
};

/// This function helps us build an attention based on 310B, It is used only on 310B.
atb::Status AttentionEdge(const AttentionParam &param, atb::Operation **operation);

}  // namespace common
}  // namespace atb_speed
#endif