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
#include <acl/acl.h>
#include <aclnn/acl_meta.h>

aclTensor *aclCreateTensor(
    const int64_t *viewDims,
    uint64_t viewDimsNum,
    aclDataType dataType,
    const int64_t *stride,
    int64_t offset,
    aclFormat format,
    const int64_t *storageDims,
    uint64_t storageDimsNum,
    void *tensorData)
{
    return reinterpret_cast<aclTensor*>(0x7ffeeb4b3a4c);  // avoid nullptr
}

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnAddRmsNormQuantV2(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
{
    return 0;
}

aclnnStatus aclnnAddRmsNormQuantV2GetWorkspaceSize(
    const aclTensor *x1,
    const aclTensor *x2,
    const aclTensor *gamma,
    const aclTensor *smoothScale1Optional,
    const aclTensor *smoothScale2Optional,
    const aclTensor *betaOptional,
    int64_t axis,
    double epsilon,
    bool divMode,
    const aclTensor *y1Out,
    const aclTensor *y2Out,
    const aclTensor *xOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    return 0;
}

aclnnStatus aclnnAddRmsNormDynamicQuantV2(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
{
    return 0;
}

aclnnStatus aclnnAddRmsNormDynamicQuantV2GetWorkspaceSize(
    const aclTensor *x1,
    const aclTensor *x2,
    const aclTensor *gamma,
    const aclTensor *smoothScale1Optional,
    const aclTensor *smoothScale2Optional,
    const aclTensor *betaOptional,
    double epsilon,
    bool outQuant1Flag,
    bool outQuant2Flag,
    const aclTensor *y1Out,
    const aclTensor *y2Out,
    const aclTensor *xOut,
    const aclTensor *scale1Out,
    const aclTensor *scale2Out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    return 0;
}

aclnnStatus aclnnAddRmsNormDynamicQuant(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
{
    return 0;
}

aclnnStatus aclnnAddRmsNormDynamicQuantGetWorkspaceSize(
    const aclTensor *x1,
    const aclTensor *x2,
    const aclTensor *gamma,
    const aclTensor *smoothScale1Optional,
    const aclTensor *smoothScale2Optional,
    double epsilon,
    const aclTensor *y1Out,
    const aclTensor *y2Out,
    const aclTensor *xOut,
    const aclTensor *scale1Out,
    const aclTensor *scale2Out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    return 0;
}

aclnnStatus aclnnDequantRopeQuantKvcacheGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *cos,
    const aclTensor *sin,
    aclTensor *kCacheRef,
    aclTensor *vCacheRef,
    const aclTensor *indices,
    const aclTensor *scaleK,
    const aclTensor *scaleV,
    const aclTensor *offsetKOptional,
    const aclTensor *offsetVOptional,
    const aclTensor *weightScaleOptional,
    const aclTensor *activationScaleOptional,
    const aclTensor *biasOptional,
    const aclIntArray *sizeSplits,
    char *quantModeOptional,
    char *layoutOptional,
    bool kvOutput,
    char *cacheModeOptional,
    const aclTensor *qOut,
    const aclTensor *kOut,
    const aclTensor *vOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    return 0;
};

aclnnStatus aclnnDequantRopeQuantKvcache(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    return 0;
};

aclnnStatus aclnnGroupedMatmulSwigluQuantGetWorkspaceSize(
    const aclTensor *x, const aclTensor *weight,
    const aclTensor *bias, const aclTensor *offset,
    const aclTensor *weightScale, const aclTensor *xScale,
    const aclTensor *groupList,
    aclTensor *output, aclTensor *outputScale,
    aclTensor *outputOffset, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    return 0;
};

aclnnStatus aclnnGroupedMatmulSwigluQuant(void* workspace, uint64_t workspaceSize,
    aclOpExecutor* executor, aclrtStream stream)
{
    return 0;
};

aclnnStatus aclnnMoeDistributeCombineGetWorkspaceSize(const aclTensor* expandX, const aclTensor* expertIds,
                                                      const aclTensor* expandIdx, const aclTensor* epSendCounts,
                                                      const aclTensor* expertScales, const aclTensor* tpSendCounts,
                                                      const char* groupEp, int64_t epWorldSize, int64_t epRankId,
                                                      int64_t moeExpertNum, const char* groupTp, int64_t tpWorldSize,
                                                      int64_t tpRankId, int64_t expertShardType,
                                                      int64_t sharedExpertRankNum, int64_t globalBs, aclTensor* x,
                                                      uint64_t* workspaceSize, aclOpExecutor** executor)
{
    return 0;
}

aclnnStatus aclnnMoeDistributeCombine(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
    aclrtStream stream)
{
    return 0;
}

aclnnStatus aclnnMoeDistributeDispatchGetWorkspaceSize(const aclTensor* x, const aclTensor* expertIds,
                                                       const aclTensor* scales, const char* groupEp,
                                                       int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum,
                                                       const char* groupTp, int64_t tpWorldSize, int64_t tpRankId,
                                                       int64_t expertShardType, int64_t sharedExpertRankNum,
                                                       int64_t quantMode, int64_t globalBs,
                                                       aclTensor* expandX, aclTensor* dynamicScales,
                                                       aclTensor* expandIdx, aclTensor* expertTokenNums,
                                                       aclTensor* epRecvCounts, aclTensor* tpRecvCounts,
                                                       uint64_t* workspaceSize, aclOpExecutor** executor)
{
    return 0;
}
aclnnStatus aclnnMoeDistributeDispatch(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                       aclrtStream stream)
{
    return 0;
}

aclnnStatus aclnnMoeDistributeCombineV2GetWorkspaceSize(const aclTensor* expandX, const aclTensor* expertIds,
    const aclTensor* assistInfo, const aclTensor* epSendCounts,
    const aclTensor* expertScales, const aclTensor* tpSendCountsOptional,
    const aclTensor* xActiveMaskOptional, const aclTensor* activationScaleOptional,
    const aclTensor* weightScaleOptional, const aclTensor* groupListOptional, const aclTensor* expandScalesOptional,
    const aclTensor* sharedExpertXOptional,
    const char* groupEp, int64_t epWorldSize,
    int64_t epRankId, int64_t moeExpertNum,
    const char* groupTp, int64_t tpWorldSize, int64_t tpRankId,
    int64_t expertShardType, int64_t sharedExpertNum, int64_t sharedExpertRankNum,
    int64_t globalBs, int64_t outDtype, int64_t commQuantMode,
    int64_t groupListType, const char* commAlg, aclTensor* xOut, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    return 0;
}

aclnnStatus aclnnMoeDistributeCombineV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                        aclrtStream stream)
{
    return 0;
}

aclnnStatus aclnnMoeDistributeDispatchV2GetWorkspaceSize(const aclTensor* x, const aclTensor* expertIds,
    const aclTensor* scalesOptional, const aclTensor* xActiveMaskOptional,
    const aclTensor* expertScalesOptional,
    const char* groupEp, int64_t epWorldSize, int64_t epRankId,
    int64_t moeExpertNum, const char* groupTp, int64_t tpWorldSize,
    int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum,
    int64_t sharedExpertRankNum, int64_t quantMode, int64_t globalBs,
    int64_t expertTokenNumsType, const char* commAlg,
    aclTensor* expandXOut, aclTensor* dynamicScalesOut,
    aclTensor* assistInfoForCombineOut, aclTensor* expertTokenNumsOut,
    aclTensor* epRecvCountsOut, aclTensor* tpRecvCountsOut, aclTensor* expandScalesOut,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    return 0;
}

aclnnStatus aclnnMoeDistributeDispatchV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                        aclrtStream stream)
{
    return 0;
}

aclnnStatus aclnnDequantSwigluQuantGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *weightScaleOptional,
    const aclTensor *activationScaleOptional,
    const aclTensor *biasOptional,
    const aclTensor *quantScaleOptional,
    const aclTensor *quantOffsetOptional,
    const aclTensor *groupIndexOptional,
    bool activateLeft,
    char *quantModeOptional,
    const aclTensor *yOut,
    const aclTensor *scaleOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    return 0;
};

aclnnStatus aclnnDequantSwigluQuant(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    return 0;
};

aclnnStatus aclnnQuantGroupedMatmulDequantGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *weight,
    const aclTensor *weightScale,
    const aclTensor *groupList,
    const aclTensor *biasOptional,
    const aclTensor *xScaleOptional,
    const aclTensor *xOffsetOptional,
    const aclTensor *smoothScaleOptional,
    char *xQuantMode,
    bool transposeWeight,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    return 0;
}

aclnnStatus aclnnQuantGroupedMatmulDequant(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    return 0;
}

aclnnStatus aclnnObfuscationSetupGetWorkspaceSize(
    const int32_t fdToClose, const int32_t dataType,
    const int32_t hiddenSize, const int32_t tpRank,
    const int32_t modelObfSeedId, const int32_t dataObfSeedId,
    const int32_t cmd, const int32_t threadNum,
    const aclTensor* fd, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    return 0;
}

aclnnStatus aclnnObfuscationSetup(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
{
    return 0;
}

aclnnStatus aclnnObfuscationCalculateGetWorkspaceSize(
    const int32_t fd, const aclTensor* x,
    const int32_t param, int32_t cmd,
    const aclTensor* y, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    return 0;
}

aclnnStatus aclnnObfuscationCalculate(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
{
    return 0;
}

aclnnStatus aclnnObfuscationSetupV2GetWorkspaceSize(
    int32_t fdToClose, int32_t dataType,
    int32_t hiddenSize, int32_t tpRank,
    int32_t modelObfSeedId, int32_t dataObfSeedId,
    int32_t cmd, int32_t threadNum,
    float obfCoefficient, aclTensor* fd,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    return 0;
}

aclnnStatus aclnnObfuscationSetupV2(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
{
    return 0;
}

aclnnStatus aclnnObfuscationCalculateV2GetWorkspaceSize(
    const int32_t fd, const aclTensor* x,
    const int32_t param, int32_t cmd,
    float obfCoefficient, aclTensor* y,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    return 0;
}

aclnnStatus aclnnObfuscationCalculateV2(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
{
    return 0;
}

aclnnStatus aclnnQuantMatmulV5GetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2,
    const aclTensor* x1Scale, const aclTensor* x2Scale, const aclTensor* yScale,
    const aclTensor* x1Offset, const aclTensor* x2Offset, const aclTensor* yOffset,
    const aclTensor* bias,
    const bool transposeX1,
    const bool transposeX2,
    const int groupSize,
    aclTensor *output, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    return 0;
}

aclnnStatus aclnnQuantMatmulV5(void* workspace, uint64_t workspaceSize,
    aclOpExecutor* executor, aclrtStream stream)
{
    return 0;
};

aclnnStatus MoeInitRoutingOperation(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
{
    return 0;
};

aclnnStatus aclnnMoeInitRoutingV2GetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *expertIdx,
    int64_t activeNum,
    int64_t expertCapacity,
    int64_t expertNum,
    int64_t dropPadMode,
    int64_t expertTokensCountOrCumsumFlag,
    bool expertTokensBeforeCapacityFlag,
    aclTensor *expandedXOut,
    aclTensor *expandedRowIdxOut,
    aclTensor *expertTokensCountOrCumsumOut,
    aclTensor *expertTokensBeforeCapacityOut,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    return 0;
};

aclnnStatus aclnnMoeInitRoutingV2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    return 0;
};

aclnnStatus MoeInitRoutingQuantOperation(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
{
    return 0;
};

aclnnStatus aclnnMoeInitRoutingQuantV2GetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *expertIdx,
    const aclTensor *scaleOptional,
    const aclTensor *offsetOptional,
    int64_t activeNum,
    int64_t expertCapacity,
    int64_t expertNum,
    int64_t dropPadMode,
    int64_t expertTokensCountOrCumsumFlag,
    bool expertTokensBeforeCapacityFlag,
    int64_t quantMode,
    const aclTensor *expandedXOut,
    const aclTensor *expandedRowIdxOut,
    const aclTensor *expertTokensCountOrCumsumOutOptional,
    const aclTensor *expertTokensBeforeCapacityOutOptional,
    const aclTensor *dynamicQuantScaleOutOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    return 0;
};

aclnnStatus aclnnMoeInitRoutingQuantV2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    return 0;
};

aclnnStatus aclnnMatmulCompressGetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* weight,
    const aclTensor* bias,
    const aclTensor* compressIndex,
    aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    return 0;
};

aclnnStatus aclnnMatmulCompress(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    return 0;
};

aclnnStatus aclnnRangeGetWorkspaceSize(
        const aclScalar* start,
        const aclScalar* end,
        const aclScalar *step,
        aclTensor *out,
        uint64_t *workspaceSize,
        aclOpExecutor **executor)
{
    return 0;
};

aclnnStatus aclnnRange(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    return 0;
};

aclnnStatus aclnnMinimumGetWorkspaceSize(
    const aclTensor *self,
    const aclTensor *other,
    aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    return 0;
};

aclnnStatus aclnnMinimum(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    return 0;
};

#ifdef __cplusplus
}
#endif