/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lightning_indexer_proto.cpp
 * \brief
 */
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "error/ops_error.h"
#include "lightning_indexer_op_input_index.h"


using namespace ge;
using namespace optiling;

namespace ops {
static ge::graphStatus InferShapeLightningIndexer(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        OPS_LOG_E("LightningIndexer", "context is nullptr!");
        return ge::GRAPH_FAILED;
    }
    const gert::Shape *queryShape = context->GetInputShape(QUERY_INDEX);
    const gert::Shape *keyShape = context->GetInputShape(KEY_INDEX);
    gert::Shape *outShape = context->GetOutputShape(LIGHTNING_INDEXER);

    auto attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL(context, attrs, return ge::GRAPH_FAILED)
    const char *inputLayoutQueryPtr = attrs->GetAttrPointer<char>(ATTR_QUERY_LAYOUT_INDEX);
    OPS_LOG_E_IF_NULL(context, inputLayoutQueryPtr, return ge::GRAPH_FAILED)
    const int64_t *seleced_count = attrs->GetInt(ATTR_SELECT_COUNT_INDEX);

    std::string inputLayoutQueryPtrStr = std::string(inputLayoutQueryPtr);
    if (inputLayoutQueryPtrStr != "TND" && inputLayoutQueryPtrStr != "BSND") {
        OPS_LOG_E(context, "The input layout query should be TND and BSND, but got %s.", inputLayoutQueryPtrStr.c_str());
        return GRAPH_FAILED;
    }

    outShape->SetDimNum(queryShape->GetDimNum());
    if (inputLayoutQueryPtrStr == "BSND") {
        outShape->SetDim(0, queryShape->GetDim(0));                   // 0:Dim B
        outShape->SetDim(1, queryShape->GetDim(1));                   // 1:Dim S
        outShape->SetDim(DIM_IDX_TWO, keyShape->GetDim(DIM_IDX_TWO)); // 2:Dim N
        outShape->SetDim(DIM_IDX_THREE, *seleced_count);              // 3:Dim K
    } else {
        outShape->SetDim(0, queryShape->GetDim(0));         // 0:Dim T
        outShape->SetDim(1, keyShape->GetDim(DIM_IDX_TWO)); // 1:Dim N
        outShape->SetDim(DIM_IDX_TWO, *seleced_count);      // 2:Dim K
    }

    OPS_LOG_D(context->GetNodeName(), "LightningIndexer InferShape end.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeLightningIndexer(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        OPS_LOG_E("LightningIndexer", "context is nullptr!");
        return ge::GRAPH_FAILED;
    }
    OPS_LOG_D(context->GetNodeName(), "Enter LightningIndexer InferDataType impl.");
    // default set q's dtype as fia's output type
    ge::DataType outputType = context->GetInputDataType(ACTUAL_SEQ_Q_INDEX);
    // attention_out, outidx:0
    context->SetOutputDataType(LIGHTNING_INDEXER, outputType);
    OPS_LOG_D(context->GetNodeName(), "LightningIndexer InferDataType end.");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(LightningIndexer)
    .InferShape(InferShapeLightningIndexer)
    .InferDataType(InferDataTypeLightningIndexer);
} // namespace ops
