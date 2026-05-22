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
#include <aclnn/acl_meta.h>
#include <atb/atb_infer.h>
#include "atb_speed/log.h"
#include "operations/aclnn/ops/add_rms_norm_dynamic_quant_operation.h"
#include "../../../test_utils/test_aclnn_utils.h"

namespace atb_speed {
namespace test {

TEST(aclnnOpsAddRmsNormDynamicQuantOperation, AddRmsNormDynamicQuantOperation)
{
    GlobalMockObject::verify(); // 清空mock
    aclrtSetDevice(0);
    // 创建stream, 便于后续运行测试
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    atb_speed::common::AclNNAddNormDynamicQuantMatmulParam aclNNAddNormDynamicQuantMatmulParam;
    aclNNAddNormDynamicQuantMatmulParam.epsilon = 1e-6;
    atb_speed::common::AddRmsNormDynamicQuantOperation op("test", aclNNAddNormDynamicQuantMatmulParam);
    uint32_t inputNum = op.GetInputNum();
    uint32_t outputNum = op.GetOutputNum();
    atb::SVector<atb::TensorDesc> inputDescs;
    atb::SVector<atb::TensorDesc> outputDescs;
    inputDescs.resize(inputNum);
    outputDescs.resize(outputNum);
    
    // 创建输入的TensorDesc
    int NUM0 = 0;
    int NUM1 = 1;
    int NUM2 = 2;
    int NUM8 = 8;
    inputDescs[NUM0].shape.dimNum = NUM2;
    inputDescs[NUM0].shape.dims[NUM0] = NUM2;
    inputDescs[NUM0].shape.dims[1] = NUM8;
    inputDescs[NUM0].dtype = ACL_FLOAT16;
    inputDescs[NUM0].format = ACL_FORMAT_ND;
    inputDescs[NUM1].shape.dimNum = NUM2;
    inputDescs[NUM1].shape.dims[NUM0] = NUM2;
    inputDescs[NUM1].shape.dims[NUM1] = NUM8;
    inputDescs[NUM1].dtype = ACL_FLOAT16;
    inputDescs[NUM1].format = ACL_FORMAT_ND;
    inputDescs[NUM2].shape.dimNum = NUM1;
    inputDescs[NUM2].shape.dims[NUM0] = NUM8;
    inputDescs[NUM2].dtype = ACL_FLOAT16;
    inputDescs[NUM2].format = ACL_FORMAT_ND;

    int ret = op.InferShape(inputDescs, outputDescs);
    EXPECT_EQ(ret, NUM0);
    
    // 创建aclnnVariantPack, 将上面的TensorDesc赋值给variantPack
    atb::VariantPack variantPack = atb_speed::test::InitVariantPack(op, inputDescs, outputDescs);

    // 内部aclCreateTensor已打桩, 不实际申请tensor
    ret = op.CreateAclNNVariantPack(variantPack);

    // 根据不同算子去校验aclnnVariantPack中除tensor指针外的数值, 本用例仅校验aclnnVariantPack的tensorIdx
    atb_speed::test::VerifyTensorIndices(op);
    
    // 由于算子本身已打桩, 此处返回值不做校验
    ret = op.SetAclNNWorkspaceExecutor();
    ret = op.ExecuteAclNNOp(nullptr, stream);

    // 销毁申请的stream
    aclrtDestroyStream(stream);
}

} // namespace test
} // namespace atb_speed