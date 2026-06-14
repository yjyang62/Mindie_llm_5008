/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#include <atb/atb_infer.h>
#include "operations/fusion/embedding/word_embedding.h"
#include "operations/fusion/linear/linear_parallel.h"

namespace atb_speed {
namespace common {

bool CheckGatherParamForWordEmbedding(const atb::infer::GatherParam &param)
{
    if (param.axis != 0) {
        return false;
    }
    return true;
}

bool CheckGraphParamForWordEmbeddingCase1(const atb::GraphParam &opGraph)
{
    if (opGraph.nodes.size() != 1) {
        return false;
    }
    // 校验图算子的internalTensor个数, worldSize==1时为0
    if (opGraph.internalTensorNum != 0) {
        return false;
    }
    // 校验图算子第一个结点的intensorId, 要求为{0, 1}
    if (opGraph.nodes.at(0).inTensorIds.size() != 2 ||
        opGraph.nodes.at(0).inTensorIds.at(0) != 0 ||
        opGraph.nodes.at(0).inTensorIds.at(1) != 1) {
        return false;
    }
    // 校验图算子第一个结点的outtensorId, 要求为{2}
    if (opGraph.nodes.at(0).outTensorIds.size() != 1 ||
        opGraph.nodes.at(0).outTensorIds.at(0) != 2) {
        return false;
    }
    return true;
}

bool CheckGraphParamForWordEmbeddingCase2(const atb::GraphParam &opGraph)
{
    if (opGraph.nodes.size() != 1) {
        return false;
    }
    // 校验图算子的internalTensor个数, worldSize>1时为2
    if (opGraph.internalTensorNum != 2) {
        return false;
    }
    // 校验图算子第一个结点的intensorId, 要求为{0, 1}
    if (opGraph.nodes.at(0).inTensorIds.size() != 2 ||
        opGraph.nodes.at(0).inTensorIds.at(0) != 0 ||
        opGraph.nodes.at(0).inTensorIds.at(1) != 1) {
        return false;
    }
    // 校验图算子第一个结点的outtensorId, 要求为{3}
    if (opGraph.nodes.at(0).outTensorIds.size() != 1 ||
        opGraph.nodes.at(0).outTensorIds.at(0) != 3) {
        return false;
    }
    return true;
}

TEST(fusionEmbeddingTest, wordEmbedding)
{
    GlobalMockObject::verify(); // 清空mock
    WordEmbeddingParam param;
    atb::Operation *op = nullptr;
    MOCKER(atb::CreateOperation<atb::infer::GatherParam>).expects(once())
        .with(checkWith(CheckGatherParamForWordEmbedding), any()).will(returnValue(0));
    MOCKER(atb::CreateOperation<atb::GraphParam>).expects(once())
        .with(checkWith(CheckGraphParamForWordEmbeddingCase1), any()).will(returnValue(0));
    atb::Status ret = WordEmbedding(param, &op);
    EXPECT_EQ(ret, 0);
}

TEST(fusionEmbeddingTest, wordEmbeddingWorldSize2)
{
    GlobalMockObject::verify();
    WordEmbeddingParam param;
    atb::Operation *op = nullptr;
    param.tensorParallelInfo.worldSize = 2;
    MOCKER(AddCommunicationOp).expects(once()).will(returnValue(0));
    MOCKER(atb::CreateOperation<atb::GraphParam>).expects(once())
        .with(checkWith(CheckGraphParamForWordEmbeddingCase2), any()).will(returnValue(0));
    atb::Status ret = WordEmbedding(param, &op);
    EXPECT_EQ(ret, 0);
}
} // namespace common
} // namespace atb_speed
