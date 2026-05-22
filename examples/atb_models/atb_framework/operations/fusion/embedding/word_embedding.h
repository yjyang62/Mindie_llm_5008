/**
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

#ifndef ATB_SPEED_MODELS_COMMON_LAYER_WORD_EMBEDDING_H
#define ATB_SPEED_MODELS_COMMON_LAYER_WORD_EMBEDDING_H

#include "atb/atb_infer.h"
#include "atb_speed/utils/operation_util.h"
#include "operations/fusion/linear/linear_parallel.h"

namespace atb_speed {
namespace common {
/// A struct defines `WordEmbedding` operation's parameters.
struct WordEmbeddingParam {
    /// Whether input tensor is unpadded.
    /// If false, input tensor shape is [batch_size, seq_len]. For request shortter than seq_len, it will be padded.
    /// If true, input tensor shape is [(seq_len_1 + seq_len_2 + ... + seq_len_n)].
    bool unpadInputs = false;
    /// Which axis to gather slices from input tensors.
    int axis = 0;
    /// A struct defined in `/fusion/linear/linear_parallel.h`. The vocabulary list will be split according to the
    /// settings of the struct; under default parameters, even if the model runs on multiple devices,
    /// the vocabulary will not be split.
    atb_speed::common::TensorParallelInfo tensorParallelInfo;
};

/// Create `WordEmbedding` graph operation.
/// \param param `WordEmbedding`'s parameters, see `WordEmbeddingParam` for more details.
/// \param operation The address pointer to the `WordEmbedding` operation.
///
/// Operation's Inputs:
/// Name                   | Dtype | Shape |
/// ---------------------- | ----- | ----- |
/// embedding_weight       | float16/float32/bfloat16/int32/uint32 | [vocab_size, hidden_size] |
/// input_ids              | int64/int32/uint32 | no constraint |
///
/// Operation's Outputs:
/// Name                   | Dtype | Shape |
/// ---------------------- | ----- | ----- |
/// output                 | float16/float32/bfloat16/int32/uint32 | [len(all_seq), hidden_size] or [bsz, seq_len, hidden_size] |
///
/// Example:
/// \code
/// enum TensorIdx: uint32_t {
///     IN_EMBEDDING_WEIGHT_ID = 0,
///     IN_INPUT_IDS_ID,
///     OUT_OUTPUT_ID,
/// };
/// std::vector<atb::Tensor> Tensors = {...};   // Prepare tensors here.
/// atb::Operation *op = nullptr;
/// atb_speed::Model::Node wordEmbeddingNode;
/// atb_speed::common::WordEmbeddingParam wordEmbeddingParam;
/// // Modify wordEmbeddingParam's attributes if needed.
/// CHECK_OPERATION_STATUS_RETURN(WordEmbedding(wordEmbeddingParam, &op));
/// wordEmbeddingNode.operation.reset(op);
/// wordEmbeddingNode.inTensors = {
///     Tensors.at(IN_EMBEDDING_WEIGHT_ID),
///     Tensors.at(IN_INPUT_IDS_ID)
/// };
/// wordEmbeddingNode.outTensors = {
///     Tensors.at(OUT_OUTPUT_ID)
/// };
/// graph.nodes.push_back(wordEmbeddingNode);  // Add node to its graph.
/// \endcode
atb::Status WordEmbedding(const WordEmbeddingParam &param, atb::Operation **operation);
}  // namespace common
}  // namespace atb_speed
#endif
