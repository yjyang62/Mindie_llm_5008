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
#include <acl/acl.h>
#include "atb_speed/log.h"
#include "operations/aclnn/ops/prompt_flash_attention_operation.h"

namespace atb_speed {
namespace test {

TEST(aclnnOpsPromptFlashAttentionOperation, StepByStepTest)
{
    GlobalMockObject::verify(); // 清空mock
    aclrtSetDevice(0);
    
    // 测试基本构造函数和参数获取
    atb_speed::common::AclNNFlashAttentionParam aclNNFlashAttentionParam;
    aclNNFlashAttentionParam.needMask = false;
    aclNNFlashAttentionParam.numHeads = 8;
    aclNNFlashAttentionParam.numKeyValueHeads = 8;
    aclNNFlashAttentionParam.inputLayout = "BSND";
    
    atb_speed::common::PromptFlashAttentionOperation op("test", aclNNFlashAttentionParam);
    
    uint32_t inputNum = op.GetInputNum();
    uint32_t outputNum = op.GetOutputNum();
    
    EXPECT_GT(inputNum, 0u);
    EXPECT_GT(outputNum, 0u);
    
    // 测试形状推断
    atb::SVector<atb::TensorDesc> inputDescs;
    atb::SVector<atb::TensorDesc> outputDescs;
    inputDescs.resize(inputNum);
    outputDescs.resize(outputNum);
    
    // 简化输入配置
    for (uint32_t i = 0; i < inputNum; i++) {
        inputDescs[i].shape.dimNum = 1;
        inputDescs[i].shape.dims[0] = 1;
        inputDescs[i].dtype = ACL_FLOAT16;
        inputDescs[i].format = ACL_FORMAT_ND;
    }
    
    int ret = op.InferShape(inputDescs, outputDescs);
    EXPECT_EQ(ret, 0);
    
    // 测试stream创建
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
}

} // namespace test
} // namespace atb_speed