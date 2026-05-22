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
#include "operations/aclnn/ops/w16a16sc_operation.h"
#include "../../../test_utils/test_aclnn_utils.h"


namespace atb_speed {

namespace test {

TEST(aclnnOpsW16A16SCOperation, W16A16SCOperation)
{
    GlobalMockObject::verify(); // 清空mock
    aclrtSetDevice(0);
    // 创建stream, 便于后续运行测试
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    atb_speed::common::AclNNW16A16SCParam aclNNW16A16SCParam;
    atb_speed::common::W16A16SCOperation op("test", aclNNW16A16SCParam);
    uint32_t inputNum = op.GetInputNum();
    uint32_t outputNum = op.GetOutputNum();
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    atb::TensorDesc atbTensorDesc;
    inTensorDescs.resize(inputNum);
    outTensorDescs.resize(outputNum);

    // 创建输入
    int NUM0 = 0;
    int NUM1 = 1;
    int NUM2 = 2;
    int NUM3 = 3;
    int NUM6 = 6;
    int NUM8 = 8;
    int NUM9 = 9;
    constexpr size_t NUM4 = 4;
    inTensorDescs[NUM0].shape.dimNum = NUM2;
    inTensorDescs[NUM0].shape.dims[NUM0] = NUM2;
    inTensorDescs[NUM0].shape.dims[NUM1] = NUM8;
    inTensorDescs[NUM0].dtype = ACL_FLOAT16;
    inTensorDescs[NUM0].format = ACL_FORMAT_ND;

    inTensorDescs[NUM1].shape.dimNum = NUM1;
    inTensorDescs[NUM1].shape.dims[NUM0] = NUM9;
    inTensorDescs[NUM1].dtype = ACL_FLOAT16;
    inTensorDescs[NUM1].format = ACL_FORMAT_ND;

    inTensorDescs[NUM2].shape.dimNum = NUM1;
    inTensorDescs[NUM2].shape.dims[NUM0] = NUM6;
    inTensorDescs[NUM2].dtype = ACL_FLOAT;
    inTensorDescs[NUM2].format = ACL_FORMAT_ND;

    inTensorDescs[NUM3].shape.dimNum = NUM1;
    inTensorDescs[NUM3].shape.dims[NUM0] = NUM6;
    inTensorDescs[NUM3].dtype = ACL_INT8;
    inTensorDescs[NUM3].format = ACL_FORMAT_ND;

    atbTensorDesc.format = ACL_FORMAT_FRACTAL_NZ;
    atbTensorDesc.shape.dimNum = NUM2;
    atbTensorDesc.shape.dims[NUM0] = 32;
    atbTensorDesc.shape.dims[NUM1] = 48;

    int ret = op.InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(ret, NUM0);

    // 创建aclnnVariantPack, 将上面的TensorDesc赋值给variantPack
    atb::VariantPack variantPack = atb_speed::test::InitVariantPack(op, inTensorDescs, inTensorDescs);

    // 内部aclCreateTensor已打桩, 不实际申请tensor
    ret = op.CreateAclNNInTensorVariantPack(variantPack);
    ret = op.CreateAclNNOutTensorVariantPack(variantPack);

    // 根据不同算子去校验aclnnVariantPack中除tensor指针外的数值, 本用例仅校验aclnnVariantPack的tensorIdx
    atb_speed::test::VerifyTensorIndices(op);
    
    // 由于算子本身已打桩, 此处返回值不做校验
    ret = op.SetAclNNWorkspaceExecutor();
    ret = op.ExecuteAclNNOp(nullptr, stream);

    // 测试FRACTAL_NZ格式下的shape
    atb::Dims result = op.GetWeightStorageShape(atbTensorDesc);
    EXPECT_EQ(result.dimNum, NUM4);
    EXPECT_EQ(result.dims[0], 48 / 16);
    EXPECT_EQ(result.dims[1], 32 / 16);
    EXPECT_EQ(result.dims[2], 16);
    EXPECT_EQ(result.dims[3], 16);

    // 销毁申请的stream
    aclrtDestroyStream(stream);
}

} // namespace test
} // namespace atb_speed
