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
 * \file lightning_indexer_tiling_info_parser.cpp
 * \brief
 */

#include <map>
#include <string>
#include <utility>
#include <numeric>
#include <algorithm>
#include "lightning_indexer_tiling.h"
#include "lightning_indexer_tiling_info_parser.h"

using std::map;
using std::string;
using std::pair;
using namespace ge;
// using namespace AscendC;
namespace optiling {

ge::graphStatus LiInfoParser::CheckRequiredInOutExistence() const
{
    OPS_ERR_IF(opParamInfo_.query.shape == nullptr, OPS_LOG_E(opName_, "Shape of tensor query is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.query.desc == nullptr, OPS_LOG_E(opName_, "Desc of tensor query is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.key.shape == nullptr, OPS_LOG_E(opName_, "Shape of tensor k is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.key.desc == nullptr, OPS_LOG_E(opName_, "Desc of tensor k is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.weights.shape == nullptr, OPS_LOG_E(opName_, "Shape of tensor value is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.weights.desc == nullptr, OPS_LOG_E(opName_, "Desc of tensor value is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.attenOut.shape == nullptr, OPS_LOG_E(opName_, "Shape of tensor output is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.attenOut.desc == nullptr, OPS_LOG_E(opName_, "Desc of tensor output is nullptr"),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LiInfoParser::CheckRequiredAttrExistence() const
{
    OPS_ERR_IF(opParamInfo_.layOut == nullptr, OPS_LOG_E(opName_, "attr layout_query is nullptr"),
               return ge::GRAPH_FAILED);

    OPS_ERR_IF(opParamInfo_.layOutKey == nullptr, OPS_LOG_E(opName_, "attr layout_key is nullptr"),
               return ge::GRAPH_FAILED);

    OPS_ERR_IF(opParamInfo_.selectedCount == nullptr, OPS_LOG_E(opName_, "attr selected_count is nullptr"),
               return ge::GRAPH_FAILED);

    OPS_ERR_IF(opParamInfo_.sparseMode == nullptr, OPS_LOG_E(opName_, "attr sparse_mode is nullptr"),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LiInfoParser::CheckRequiredParaExistence() const
{
    if (CheckRequiredInOutExistence() != ge::GRAPH_SUCCESS ||
        CheckRequiredAttrExistence() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LiInfoParser::GetOpName()
{
    if (context_->GetNodeName() == nullptr) {
        OPS_LOG_E("LightningIndexer", "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }
    opName_ = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LiInfoParser::GetNpuInfo()
{
    platformInfo_ = context_->GetPlatformInfo();
    OPS_ERR_IF(platformInfo_ == nullptr,
        OPS_LOG_E(opName_, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo_);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    OPS_ERR_IF(aicNum == 0 || aivNum == 0,
        OPS_LOG_E(opName_, "num of core obtained is 0."), return GRAPH_FAILED);

    socVersion_ = ascendcPlatform.GetSocVersion();
    if ((socVersion_ != platform_ascendc::SocVersion::ASCEND910B) &&
        (socVersion_ != platform_ascendc::SocVersion::ASCEND910_93)) {
        OPS_LOG_E(opName_, "SOC Version[%d] is not support.", (int32_t)socVersion_);
        return GRAPH_FAILED;
    }
    OPS_ERR_IF(context_->GetWorkspaceSizes(1) == nullptr,
               OPS_LOG_E(opName_, "workSpaceSize got from ge is nullptr"), return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->GetRawTilingData() == nullptr,
               OPS_LOG_E(context_->GetNodeName(), "RawTilingData got from GE context is nullptr."),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void LiInfoParser::GetOptionalInputParaInfo()
{
    opParamInfo_.actualSeqLengthsQ.tensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
    opParamInfo_.actualSeqLengthsQ.desc = context_->GetOptionalInputDesc(ACTUAL_SEQ_Q_INDEX);
    opParamInfo_.actualSeqLengths.tensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_K_INDEX);
    opParamInfo_.actualSeqLengths.desc = context_->GetOptionalInputDesc(ACTUAL_SEQ_K_INDEX);
    opParamInfo_.blockTable.tensor = context_->GetOptionalInputTensor(BLOCK_TABLE_INDEX);
    opParamInfo_.blockTable.desc = context_->GetOptionalInputDesc(BLOCK_TABLE_INDEX);
}

void LiInfoParser::GetInputParaInfo()
{
    opParamInfo_.query.desc = context_->GetInputDesc(QUERY_INDEX);
    opParamInfo_.query.shape = context_->GetInputShape(QUERY_INDEX);
    opParamInfo_.key.desc = context_->GetInputDesc(KEY_INDEX);
    opParamInfo_.key.shape = context_->GetInputShape(KEY_INDEX);
    opParamInfo_.weights.desc = context_->GetInputDesc(WEIGTHS_INDEX);
    opParamInfo_.weights.shape = context_->GetInputShape(WEIGTHS_INDEX);
    GetOptionalInputParaInfo();
}

void LiInfoParser::GetOutputParaInfo()
{
    opParamInfo_.attenOut.desc = context_->GetOutputDesc(LIGHTNING_INDEXER);
    opParamInfo_.attenOut.shape = context_->GetOutputShape(LIGHTNING_INDEXER);
}

ge::graphStatus LiInfoParser::GetAndCheckAttrParaInfo()
{
    auto attrs = context_->GetAttrs();
    OPS_ERR_IF(attrs == nullptr, OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "attrs got from ge is nullptr"),
               return ge::GRAPH_FAILED);

    OPS_LOG_I(context_->GetNodeName(), "GetAndCheckAttrParaInfo start");
    opParamInfo_.layOut = attrs->GetStr(ATTR_QUERY_LAYOUT_INDEX);
    opParamInfo_.layOutKey = attrs->GetStr(ATTR_KEY_LAYOUT_INDEX);
    opParamInfo_.selectedCount = attrs->GetAttrPointer<int32_t>(ATTR_SELECT_COUNT_INDEX);
    opParamInfo_.sparseMode = attrs->GetAttrPointer<int32_t>(ATTR_SPARSE_MODE_INDEX);

    if (opParamInfo_.layOut != nullptr) {
        OPS_LOG_I(context_->GetNodeName(), "layout_query is:%s", opParamInfo_.layOut);
    }
    if (opParamInfo_.layOutKey != nullptr) {
        OPS_LOG_I(context_->GetNodeName(), "layout_key is:%s", opParamInfo_.layOutKey);
    }
    if (opParamInfo_.selectedCount != nullptr) {
        OPS_LOG_I(context_->GetNodeName(), "selscted count is:%d", *opParamInfo_.selectedCount);
    }
    if (opParamInfo_.sparseMode != nullptr) {
        OPS_LOG_I(context_->GetNodeName(), "sparse mode is:%d", *opParamInfo_.sparseMode);
    }
    OPS_LOG_I(context_->GetNodeName(), "GetAndCheckAttrParaInfo end");

    OPS_ERR_IF(
        ((std::string(opParamInfo_.layOutKey) != "PA_BSND") && (std::string(opParamInfo_.layOutKey) != "PA_BBND")),
        OPS_LOG_E(opName_, "input attr layout_key only supported PA_BSND or PA_BBND."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(
        ((std::string(opParamInfo_.layOut) != "BSND") && (std::string(opParamInfo_.layOut) != "TND")),
        OPS_LOG_E(opName_, "input attr layout_query only supported BSND or TND."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(!((*opParamInfo_.selectedCount > 0) && (*opParamInfo_.selectedCount <= SELECT_LIMIT)),
               OPS_LOG_E(opName_, "input attr selected_count must > 0 and <= 2048."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(!((*opParamInfo_.sparseMode == 0) || (*opParamInfo_.sparseMode == SPARSE_MODE_THREE)),
               OPS_LOG_E(opName_, "input attr sparse_mode only supported 0 or 3."),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LiInfoParser::GetOpParaInfo()
{
    GetInputParaInfo();
    GetOutputParaInfo();
    if (ge::GRAPH_SUCCESS != GetAndCheckAttrParaInfo()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LiInfoParser::GetAndCheckInOutDataType()
{
    inputQType_ = opParamInfo_.query.desc->GetDataType();
    inputKType_ = opParamInfo_.key.desc->GetDataType();
    weightsType_ = opParamInfo_.weights.desc->GetDataType();
    outputType_ = opParamInfo_.attenOut.desc->GetDataType();

    bool inDTypeAllEqual = (inputQType_ == inputKType_) && (inputKType_ == weightsType_);
    OPS_ERR_IF(!inDTypeAllEqual,
               OPS_LOG_E(opName_, "The data types of the input query, key, and weights must be the same."),
               return ge::GRAPH_FAILED);

    OPS_ERR_IF(((inputQType_ != ge::DT_FLOAT16) && (inputQType_ != ge::DT_BF16)),
               OPS_LOG_E(opName_, "The data types of the input query, key, and weights must be float16 or bfloat16."),
               return ge::GRAPH_FAILED);

    OPS_ERR_IF(outputType_ != ge::DT_INT32,
               OPS_LOG_E(opName_, "The data types of the output selected_indices must be int32."),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LiInfoParser::GetQueryKeyAndOutLayout()
{
    // 获取query,key的Layout基准值
    const map<string, DataLayout> layoutMap = {
        {"BSND", DataLayout::BSND},
        {"TND",  DataLayout::TND }
    };

    std::string layout(opParamInfo_.layOut);
    auto it = layoutMap.find(layout);
    if (it != layoutMap.end()) {
        qLayout_ = it->second;
    }
    kLayout_ = DataLayout::BnBsND;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LiInfoParser::GetAndCheckOptionalInput()
{
    if (kLayout_ == DataLayout::BnBsND) {
        OPS_ERR_IF(opParamInfo_.blockTable.tensor == nullptr,
                   OPS_LOG_E(opName_, "key layout only supported PA_BSND, input block_table must not be null"),
                   return ge::GRAPH_FAILED);
        OPS_ERR_IF(
            opParamInfo_.actualSeqLengths.tensor == nullptr,
            OPS_LOG_E(opName_, "key layout only supported PA_BSND, input actual_seq_lengths_key must not be null"),
            return ge::GRAPH_FAILED);
        OPS_ERR_IF(opParamInfo_.blockTable.desc->GetDataType() != ge::DT_INT32,
                   OPS_LOG_E(opName_, "input block_table data type only support int32"), return ge::GRAPH_FAILED);
        OPS_ERR_IF(opParamInfo_.actualSeqLengths.desc->GetDataType() != ge::DT_INT32,
                   OPS_LOG_E(opName_, "input actual_seq_lengths_key data type only support int32"),
                   return ge::GRAPH_FAILED);
    }
    if (qLayout_ == DataLayout::TND) {
        OPS_ERR_IF(opParamInfo_.actualSeqLengthsQ.tensor == nullptr,
                   OPS_LOG_E(opName_, "when layout_query is TND, input actual_seq_lengths_query must not be null"),
                   return ge::GRAPH_FAILED);
        OPS_ERR_IF(opParamInfo_.actualSeqLengthsQ.desc->GetDataType() != ge::DT_INT32,
                   OPS_LOG_E(opName_, "input actual_seq_lengths_query data type only support int32"),
                   return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LiInfoParser::CheckShapeDim()
{
    OPS_ERR_IF((opParamInfo_.blockTable.tensor != nullptr) &&
                   (opParamInfo_.blockTable.tensor->GetStorageShape().GetDimNum() != DIM_NUM_TWO),
               OPS_LOG_E(opName_, "the dim num of block_table's shape should be 2"), return ge::GRAPH_FAILED);
    OPS_ERR_IF((kLayout_ == DataLayout::BnBsND) && (opParamInfo_.key.shape->GetStorageShape().GetDimNum() != DIM_NUM_FOUR),
               OPS_LOG_E(opName_, "the dim num of key's shape should be 4"), return ge::GRAPH_FAILED);

    uint32_t qShapeDim = opParamInfo_.query.shape->GetStorageShape().GetDimNum();
    uint32_t weightsShapeDim = opParamInfo_.weights.shape->GetStorageShape().GetDimNum();
    uint32_t outShapeDim = opParamInfo_.attenOut.shape->GetStorageShape().GetDimNum();
    uint32_t expectShapeDim = DIM_NUM_FOUR;
    if (qLayout_ == DataLayout::TND) {
        expectShapeDim = DIM_NUM_THREE;
    }
    OPS_ERR_IF(qShapeDim != expectShapeDim,
               OPS_LOG_E(opName_, "the dim num of query's shape should be %u", expectShapeDim),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(outShapeDim != expectShapeDim,
               OPS_LOG_E(opName_, "the dim num of selected_indices's shape should be %u", expectShapeDim),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(!((weightsShapeDim == (expectShapeDim - 1)) ||
                 ((weightsShapeDim == expectShapeDim) &&
                  (opParamInfo_.weights.shape->GetStorageShape().GetDim(expectShapeDim - 1) == 1))),
               OPS_LOG_E(opName_, "the dim num of weights's shape should be %u", expectShapeDim),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LiInfoParser::GetN1Size()
{
    if (qLayout_ == DataLayout::BSND) {
        n1Size_ = static_cast<uint32_t>(opParamInfo_.query.shape->GetStorageShape().GetDim(DIM_IDX_TWO));
    } else {
        // TND
        n1Size_ = static_cast<uint32_t>(opParamInfo_.query.shape->GetStorageShape().GetDim(1));
    }
    OPS_LOG_I(context_->GetNodeName(), "n1Size is %d", n1Size_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LiInfoParser::GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor,
                                                  const std::string &actualSeqLenName)
{
    size = static_cast<uint32_t>(tensor->GetShapeSize());
    if (size <= 0) {
        OPS_LOG_E(opName_, "%s's shape size is %u, it should be greater than 0.", actualSeqLenName.c_str(), size);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LiInfoParser::GetAndCheckN2Size()
{
    // PA_BSND
    n2Size_ = static_cast<uint32_t>(opParamInfo_.key.shape->GetStorageShape().GetDim(DIM_IDX_TWO));
    OPS_LOG_I(context_->GetNodeName(), "n2Size_ is %d", n2Size_);
    OPS_ERR_IF(n2Size_ != 1, OPS_LOG_E(opName_, "key shape[2] is numhead, only support 1."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LiInfoParser::GetGSize()
{
    if (n1Size_ % n2Size_ != 0) {
        OPS_LOG_E(opName_, "input query's head_num %u can not be a multiple of key's head_num %u.", n1Size_, n2Size_);
        return ge::GRAPH_FAILED;
    }
    gSize_ = n1Size_ / n2Size_;
    OPS_ERR_IF(gSize_ != 64, OPS_LOG_E(opName_, "n1 divided by n2 must equals 64."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LiInfoParser::GetBatchSize()
{
    // 获取B基准值
    // 1、非TND/NTD时, 以query的batch_size维度为基准;
    // 2、TND/NTD时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组的长度为B轴大小
    if ((qLayout_ == DataLayout::TND)) {
        return GetActualSeqLenSize(bSize_, opParamInfo_.actualSeqLengthsQ.tensor, "input actual_seq_lengths_query");
    } else { // BSND
        bSize_ = opParamInfo_.query.shape->GetStorageShape().GetDim(0);
        return ge::GRAPH_SUCCESS;
    }
}

ge::graphStatus LiInfoParser::GetHeadDim()
{
    // 以query的D维度为基准
    uint32_t dIndex = DIM_IDX_TWO;
    // 根据layout确定D维度在shape中的位置
    switch (qLayout_) {
        case DataLayout::TND:
            // TND格式: [Total, N, D] -> D是第2维(索引2)
            dIndex = DIM_IDX_TWO;
            break;
        case DataLayout::BSND:
            // BSND格式: [Batch, SeqLen, N, D] -> D是第3维(索引3)
            dIndex = DIM_IDX_THREE;
            break;
        default:
            OPS_LOG_E(opName_, "unsupported layout for getting head dim.");
            return ge::GRAPH_FAILED;
    }
    headDim_ = opParamInfo_.query.shape->GetStorageShape().GetDim(dIndex);
    OPS_ERR_IF(headDim_ != HEAD_DIM_LIMIT, OPS_LOG_E(opName_, "input query's last dim head_dim only support 128."),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LiInfoParser::GetS1Size()
{
    if (qLayout_ == DataLayout::BSND) {
        s1Size_ = opParamInfo_.query.shape->GetStorageShape().GetDim(1);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LiInfoParser::GetAndCheckBlockSize()
{
    blockSize_ = static_cast<uint32_t>(opParamInfo_.key.shape->GetStorageShape().GetDim(1));
    OPS_LOG_I(context_->GetNodeName(), "blockSize_ is %d", blockSize_);

    OPS_ERR_IF(((blockSize_ % 16 != 0) || (blockSize_ > 1024)),
               OPS_LOG_E(opName_, "input key's block_size must be a multiple of 16 and no greater than 1024."),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LiInfoParser::GetS2SizeForPageAttention()
{
    if (GetAndCheckBlockSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    maxBlockNumPerBatch_ = opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(1);
    s2Size_ = maxBlockNumPerBatch_ * blockSize_;
    OPS_LOG_I(context_->GetNodeName(), "maxBlockNumPerBatch_ is %d, blockSize_ is %d, s2Size_ is %d",
              maxBlockNumPerBatch_, blockSize_, s2Size_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LiInfoParser::GetS2Size()
{
    // 获取S2基准值
    // 1、BATCH_CONTINUOUS时, 从key的S轴获取
    // 3、PAGE_ATTENTION时, S2 = block_table.dim1 * block_size
    return GetS2SizeForPageAttention();
}

ge::graphStatus LiInfoParser::ValidateInputShapesMatch()
{
    /*
    TND:
    query [T,N1,D],
    key [BlockNum,BlockSize,N2,D],
    weight [T,N1],
    block_table [BatchSize, BatchMaxBlockNum],
    act_seq_k [BatchSize]
    act_seq_q [BatchSize],
    out [T,N2,topk]
    ----------------------
    BSND:
    query [BatchSize,S1,N1,D],
    key [BlockNum,BlockSize,N2,D],
    weight [BatchSize,S1,N1],
    block_table [BatchSize, BatchMaxBlockNum],
    act_seq_k [BatchSize]
    act_seq_q [BatchSize] 可选
    out [BatchSize,S1,N2,topk]
    */
    uint32_t queryWeightsN1Dim = 1;
    uint32_t outN2Dim = 1;
    if (qLayout_ == DataLayout::TND) {
        // -----------------------check BatchSize-------------------
        // bSize_ 来源于act_seq_q
        OPS_ERR_IF(
            (opParamInfo_.actualSeqLengths.tensor->GetShapeSize() != bSize_) ||
                (opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(0) != bSize_),
            OPS_LOG_E(
                opName_,
                "TND case input actual_seq_lengths_query, actual_seq_lengths_key, block_table dim 0 must be same."),
            return ge::GRAPH_FAILED);
        // -----------------------check T-------------------
        uint32_t qTsize = opParamInfo_.query.shape->GetStorageShape().GetDim(0);
        OPS_ERR_IF((opParamInfo_.weights.shape->GetStorageShape().GetDim(0) != qTsize) ||
                       (opParamInfo_.attenOut.shape->GetStorageShape().GetDim(0) != qTsize),
                   OPS_LOG_E(opName_, "TND case input query, weights, selected_indices dim 0 must be same."),
                   return ge::GRAPH_FAILED);
    } else {
        // -----------------------check BatchSize-------------------
        // bSize_ 来源于query
        OPS_ERR_IF((opParamInfo_.weights.shape->GetStorageShape().GetDim(0) != bSize_) ||
                       (opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(0) != bSize_) ||
                       (opParamInfo_.actualSeqLengths.tensor->GetShapeSize() != bSize_) ||
                       (opParamInfo_.attenOut.shape->GetStorageShape().GetDim(0) != bSize_),
                   OPS_LOG_E(opName_, "BSND case input query, weight, "
                                      "actual_seq_lengths_key, block_table, selected_indices dim 0 must be same."),
                   return ge::GRAPH_FAILED);
        OPS_ERR_IF((opParamInfo_.actualSeqLengthsQ.tensor != nullptr) &&
                       (opParamInfo_.actualSeqLengthsQ.tensor->GetShapeSize() != bSize_),
                   OPS_LOG_E(opName_, "BSND case input query, actual_seq_lengths_query dim 0 must be same"),
                   return ge::GRAPH_FAILED);
        // -----------------------check S1-------------------
        OPS_ERR_IF((opParamInfo_.weights.shape->GetStorageShape().GetDim(1) != s1Size_) ||
                       (opParamInfo_.attenOut.shape->GetStorageShape().GetDim(1) != s1Size_),
                   OPS_LOG_E(opName_, "BSND case input query, weight, "
                                      "selected_indices dim 1 must be same."),
                   return ge::GRAPH_FAILED);
        queryWeightsN1Dim = DIM_IDX_TWO;
        outN2Dim = DIM_IDX_TWO;
    }
    // -----------------------check N1-------------------
    OPS_ERR_IF((opParamInfo_.weights.shape->GetStorageShape().GetDim(queryWeightsN1Dim) != n1Size_),
               OPS_LOG_E(opName_, "input query, weight shape dim N1 must be same."), return ge::GRAPH_FAILED);
    // -----------------------check D-------------------
    OPS_ERR_IF((opParamInfo_.key.shape->GetStorageShape().GetDim(DIM_IDX_THREE) != headDim_),
               OPS_LOG_E(opName_, "input query, key shape last dim must be same."), return ge::GRAPH_FAILED);
    // -----------------------check N2-------------------
    OPS_ERR_IF((opParamInfo_.attenOut.shape->GetStorageShape().GetDim(outN2Dim) != n2Size_),
               OPS_LOG_E(opName_, "input query and output selected_indices shape n2 dim must be same."),
               return ge::GRAPH_FAILED);
    // -----------------------check selected_count-------------------
    OPS_ERR_IF((opParamInfo_.attenOut.shape->GetStorageShape().GetDim(outN2Dim + 1) != *opParamInfo_.selectedCount),
               OPS_LOG_E(opName_, "output selected_indices shape last dim must be same as attr selected_count."),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void LiInfoParser::GenerateInfo(LiTilingInfo &liInfo)
{
    liInfo.opName = opName_;
    liInfo.platformInfo = platformInfo_;
    liInfo.opParamInfo = opParamInfo_;
    liInfo.socVersion = socVersion_;

    liInfo.bSize = bSize_;
    liInfo.n1Size = n1Size_;
    liInfo.n2Size = n2Size_;
    liInfo.s1Size = s1Size_;
    liInfo.s2Size = s2Size_;
    liInfo.gSize = gSize_;

    liInfo.inputQType = inputQType_;
    liInfo.inputKType = inputKType_;
    liInfo.outputType = outputType_;

    liInfo.blockSize = blockSize_;
    liInfo.maxBlockNumPerBatch = maxBlockNumPerBatch_;

    liInfo.pageAttentionFlag = true;
    liInfo.sparseMode = *opParamInfo_.sparseMode;
    liInfo.selectedCount = *opParamInfo_.selectedCount;

    liInfo.inputQLayout = qLayout_;
    liInfo.inputKLayout = kLayout_;
}

ge::graphStatus LiInfoParser::ParseAndCheck(LiTilingInfo &liInfo)
{
    if (ge::GRAPH_SUCCESS != GetOpName() ||
        ge::GRAPH_SUCCESS != GetNpuInfo() ||
        ge::GRAPH_SUCCESS != GetOpParaInfo() ||
        ge::GRAPH_SUCCESS != CheckRequiredParaExistence()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetAndCheckInOutDataType() || ge::GRAPH_SUCCESS != GetQueryKeyAndOutLayout() ||
        ge::GRAPH_SUCCESS != GetAndCheckOptionalInput()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetN1Size() || ge::GRAPH_SUCCESS != GetAndCheckN2Size() || ge::GRAPH_SUCCESS != GetGSize()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetBatchSize() ||
        ge::GRAPH_SUCCESS != GetS1Size() || ge::GRAPH_SUCCESS != GetHeadDim() || ge::GRAPH_SUCCESS != GetS2Size()) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != ValidateInputShapesMatch()) {
        return ge::GRAPH_FAILED;
    }

    GenerateInfo(liInfo);

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
