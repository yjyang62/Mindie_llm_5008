/**
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

#ifndef TEST_ACLNN_UTILS_H
#define TEST_ACLNN_UTILS_H

#include <gtest/gtest.h>
#include "atb/atb_infer.h"
#include "operations/aclnn/core/acl_nn_operation.h"

namespace atb_speed {
namespace test {

/**
 * @brief 创建并初始化VariantPack
 * @param op 算子实例
 * @param inputDescs 输入张量描述
 * @param outputDescs 输出张量描述
 * @return 初始化后的VariantPack
 */
inline atb::VariantPack InitVariantPack(
    const atb_speed::common::AclNNOperation& op,
    const atb::SVector<atb::TensorDesc>& inputDescs,
    const atb::SVector<atb::TensorDesc>& outputDescs)
    {
        atb::VariantPack variantPack;
        uint32_t inputNum = op.GetInputNum();
        uint32_t outputNum = op.GetOutputNum();
        
        variantPack.inTensors.resize(inputNum);
        variantPack.outTensors.resize(outputNum);
        
        for (uint32_t i = 0; i < inputNum; ++i) {
            variantPack.inTensors[i].desc = inputDescs[i];
        }
        for (uint32_t i = 0; i < outputNum; ++i) {
            variantPack.outTensors[i].desc = outputDescs[i];
        }
        
        return variantPack;
    }

// 自定义校验接口，无实现
template <typename OpType>
void CustomVerifyTensorIndices(const OpType& op, uint32_t inputNum, uint32_t outputNum) {};

/**
 * @brief 校验aclInTensors和aclOutTensors的tensorIdx
 * @param op 算子实例
 */
template <typename OpType>
inline void VerifyTensorIndices(
    const OpType& op,
    bool DefaultVerify = true) //默认情况：tensor的index与tensorIdx一致
{
    uint32_t inputNum = op.GetInputNum();
    uint32_t outputNum = op.GetOutputNum();
    
    if (DefaultVerify) {
        // 默认校验逻辑：index == tensorIdx（保持不变）
        for (int i = 0; i < static_cast<int>(inputNum); ++i) {
            EXPECT_EQ(i, op.aclnnOpCache_->aclnnVariantPack.aclInTensors.at(i)->tensorIdx);
        }
        for (int i = 0; i < static_cast<int>(outputNum); ++i) {
            EXPECT_EQ(i, op.aclnnOpCache_->aclnnVariantPack.aclOutTensors.at(i)->tensorIdx);
        }
    } else {
        // 预留分支：
        // 允许外部函数通过钩子函数扩展
        // 预留的自定义校验接口，外部可根据需要实现
        CustomVerifyTensorIndices(op, inputNum, outputNum);
    }
    
}

} // namespace test
} // namespace atb_speed

#endif // TEST_ACLNN_UTILS_H