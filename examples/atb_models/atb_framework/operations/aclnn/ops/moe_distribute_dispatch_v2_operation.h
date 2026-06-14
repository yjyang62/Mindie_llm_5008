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

#ifndef ATB_SPEED_PLUGIN_ACLNN_MOE_DESTRIBUTE_DISPATCH_OPERATION_V2_H
#define ATB_SPEED_PLUGIN_ACLNN_MOE_DESTRIBUTE_DISPATCH_OPERATION_V2_H
#include "operations/aclnn/core/acl_nn_operation.h"
#include "operations/aclnn/core/acl_nn_operation_cache.h"
#include "atb_speed/utils/operation_util.h"

namespace atb_speed {
namespace common {

struct MoeDistributeDispatchV2Param {
    int32_t epRankId = 0;
    int32_t epRankSize = 1;
    int32_t tpRankId = 0;
    int32_t tpRankSize = 1;
    int32_t expertSharedType = 0;
    int32_t maxDecodeDpTokenSize = 0;
    int64_t sharedExpertRankNum = 0;
    int64_t moeExpertNum = 1;
    int64_t localMoeExpertNum = 1;
    int64_t topk = 8;
    int64_t quantMode = 2;
    int64_t globalBS = 0; // tiling里处理成BS*world_size
    int64_t expertTokenNumsType = 0;
    bool isQuant = false;
    bool isSharedExpert = false;
    bool quantSmooth = false;
    std::string tpCommName;
    std::string epCommName;
    std::string commAlg;
    std::string rankTableFile = "";
};

class MoeDistributeDispatchV2Operation : public AclNNOperation {
public:
    explicit MoeDistributeDispatchV2Operation(const std::string &name, MoeDistributeDispatchV2Param param);
    ~MoeDistributeDispatchV2Operation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                           atb::SVector<atb::TensorDesc> &outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;
    int32_t GetGlobalBS(const atb::TensorDesc &inTensorDesc) const;

private:
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;

    MoeDistributeDispatchV2Param param_;
};

}  // namespace common
}  // namespace atb_speed
#endif