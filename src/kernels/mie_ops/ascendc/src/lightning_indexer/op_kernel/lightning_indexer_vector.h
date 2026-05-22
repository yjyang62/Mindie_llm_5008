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
 * \file lightning_indexer_vector.h
 * \brief
 */
#ifndef LIGHTNING_INDEXER_VECTOR_H
#define LIGHTNING_INDEXER_VECTOR_H

#include "lightning_indexer_vector.h"
#include "kernel_operator.h"

namespace LightningIndexerVecNS {
using namespace AscendC;

constexpr int32_t NEG_INF = 0xFF800000;
constexpr uint8_t VEC_REPEAT_MAX = 255;
constexpr uint8_t B32_VEC_ELM_NUM = 64;
constexpr uint8_t B32_BLOCK_ALIGN_NUM = 8;
constexpr uint64_t BYTE256 = 256;

template <typename T>
__aicore__ inline void CopyIn(LocalTensor<float> &mmOutUb, LocalTensor<T> &weightsUb, GlobalTensor<float> &mMoutGm,
                              GlobalTensor<T> &weightScaleGm, int64_t MMout_gmoffset, int64_t weights_gmoffset,
                              int64_t groupInner, int64_t s2Inner, int64_t mmUbStride)
{
    // 将MMout_gmoffset copy到UB上
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    DataCopyExtParams dataCopymMoutParams;
    dataCopymMoutParams.blockCount = groupInner;
    dataCopymMoutParams.blockLen = s2Inner * sizeof(float);
    dataCopymMoutParams.srcStride = 0;
    dataCopymMoutParams.dstStride = mmUbStride;
    dataCopymMoutParams.rsv = 0;
    DataCopyPad(mmOutUb, mMoutGm[MMout_gmoffset], dataCopymMoutParams, padParams);

    // 将weights_gmoffset copy到UB
    DataCopyPadExtParams<T> padTParams{false, 0, 0, 0};
    DataCopyExtParams dataCopyweightParams;
    dataCopyweightParams.blockCount = 1;
    dataCopyweightParams.blockLen = groupInner * sizeof(T);
    dataCopyweightParams.srcStride = 0;
    dataCopyweightParams.dstStride = 0;
    dataCopyweightParams.rsv = 0;
    DataCopyPad(weightsUb, weightScaleGm[weights_gmoffset], dataCopyweightParams, padTParams);
}

template <typename T>
__aicore__ inline void CopyOut(const GlobalTensor<T> &dstGm, const LocalTensor<T> &srcUb, int64_t copyCount)
{
    DataCopyParams dataCopyOutyParams;
    dataCopyOutyParams.blockCount = 1;
    dataCopyOutyParams.blockLen = copyCount * sizeof(T);
    dataCopyOutyParams.srcStride = 0;
    dataCopyOutyParams.dstStride = 0;
    DataCopyPad(dstGm, srcUb, dataCopyOutyParams);
}

template <typename T>
__aicore__ inline void DoScale(const LocalTensor<float> &reduceCacheBuf, LocalTensor<float> &mmOutUb,
                               LocalTensor<float> &weightsUb, LocalTensor<T> &weightsTUb, LocalTensor<float> &tmpBuff,
                               int64_t groupInner, int64_t s2Inner, int32_t outerGidx)
{
    // cast bfloat16_t to float
    if constexpr (!IsSameType<T, float>::value) {
        Cast(weightsUb, weightsTUb, RoundMode::CAST_NONE, groupInner);
        PipeBarrier<PIPE_V>();
    }

    // weight broadcast: [groupInner, 1] -> [groupInner, 8]
    Brcb(tmpBuff, weightsUb, CeilDiv(groupInner, 8L), {1, 8});
    PipeBarrier<PIPE_V>();

    // do scale: [groupInner, 8] * [groupInner, s2Inner]
    uint64_t countPerRepeat = BYTE256 / sizeof(float);
    uint64_t repeatTimes = s2Inner / countPerRepeat;
    for (int32_t i = 0; i < groupInner; i++) {
        if (outerGidx == 0) {
            Mul(reduceCacheBuf[i * s2Inner], mmOutUb[i * s2Inner], tmpBuff[i * 8], countPerRepeat, repeatTimes,
                {1, 1, 0, 8, 8, 0});
        } else {
            Mul(mmOutUb[i * s2Inner], mmOutUb[i * s2Inner], tmpBuff[i * 8], countPerRepeat, repeatTimes,
                {1, 1, 0, 8, 8, 0});
        }
    }

    if (outerGidx != 0) {
        PipeBarrier<PIPE_V>();
        Add(reduceCacheBuf, mmOutUb, reduceCacheBuf, groupInner * s2Inner);
    }
    PipeBarrier<PIPE_V>();
}

__aicore__ inline uint64_t FindNearestPower2(uint64_t value)
{
    if (value <= 2) {
        return value;
    } else {
        const uint64_t pow = 63 - clz(value);
        return (1 << pow);
    }
}

// dstTensor 需要初始化0
__aicore__ inline void DoReduce(const LocalTensor<float> &srcTensor, LocalTensor<float> &dstTensor, int32_t rNum,
                                int32_t aNum)
{
    if (rNum == 1) {
        Adds<float>(dstTensor, srcTensor, 0, aNum);
        PipeBarrier<PIPE_V>();
        return;
    }

    uint32_t dichotomizeAddPow = FindNearestPower2(rNum);
    uint32_t dichotomizeAddDiffSize = rNum - dichotomizeAddPow;
    if (dichotomizeAddDiffSize != 0) {
        Add(srcTensor, srcTensor, srcTensor[dichotomizeAddPow * aNum], dichotomizeAddDiffSize * aNum);
        PipeBarrier<PIPE_V>();
    }
    int32_t nowRows = dichotomizeAddPow;
    while (nowRows > 2) {
        nowRows = nowRows / 2;
        Add(srcTensor, srcTensor, srcTensor[nowRows * aNum], nowRows * aNum);
        PipeBarrier<PIPE_V>();
    }

    Add(dstTensor, srcTensor, srcTensor[aNum], aNum);
    PipeBarrier<PIPE_V>();
}

/**
  src: 传入的初始化空间
  eleNum: 需要初始化的元素个数需为64整数倍，元素将被初始化为交错排布的-inf，-1
 */
__aicore__ inline void InitSortOutBuf(const LocalTensor<float> &src, int64_t eleNum)
{
    uint64_t mask1[2] = {0x5555555555555555, 0};
    uint64_t mask0[2] = {0xaaaaaaaaaaaaaaaa, 0};
    int64_t repeatNum = eleNum / B32_VEC_ELM_NUM;
    int64_t forLoop = repeatNum / VEC_REPEAT_MAX;
    int64_t forRemain = repeatNum % VEC_REPEAT_MAX;
    for (int i = 0; i < forLoop; i++) {
        Duplicate(src.template ReinterpretCast<int32_t>(), NEG_INF, mask1, VEC_REPEAT_MAX, 1, 8);
        Duplicate(src.template ReinterpretCast<int32_t>(), -1, mask0, VEC_REPEAT_MAX, 1, 8);
    }
    if (forRemain > 0) {
        Duplicate(src.template ReinterpretCast<int32_t>()[forLoop * VEC_REPEAT_MAX * B32_VEC_ELM_NUM], NEG_INF, mask1,
                  forRemain, 1, 8);
        Duplicate(src.template ReinterpretCast<int32_t>()[forLoop * VEC_REPEAT_MAX * B32_VEC_ELM_NUM], -1, mask0,
                  forRemain, 1, 8);
    }
    PipeBarrier<PIPE_V>();
}

/**
  src: logits和索引，前logitsNum为logits，后logitsNum为索引
  tmp: 计算使用到的临时空间，大小与src一致
  logitsNum: 排序的元素个数, 暂只支持[128,256,384,512]
 */
__aicore__ inline void SortAll(LocalTensor<float> &src, LocalTensor<float> &tmp, int64_t logitsNum)
{
    int64_t sort32Repeats = logitsNum / 32;
    Sort32(tmp, src, src[logitsNum].ReinterpretCast<uint32_t>(), sort32Repeats);
    PipeBarrier<PIPE_V>();

    int64_t mrgGroups = sort32Repeats;
    int64_t mrgElements = 32;
    int64_t i = 0;
    LocalTensor<float> srcTensor;
    LocalTensor<float> dstTensor;
    while (true) {
        if (i % 2 == 0) {
            srcTensor = tmp;
            dstTensor = src;
        } else {
            srcTensor = src;
            dstTensor = tmp;
        }
        AscendC::MrgSort4Info params;
        params.elementLengths[0] = mrgElements;
        params.elementLengths[1] = mrgElements;
        params.elementLengths[2] = mrgElements;
        params.elementLengths[3] = mrgElements;
        params.ifExhaustedSuspension = false;
        params.validBit = 0b1111;

        AscendC::MrgSortSrcList<float> srcList;
        srcList.src1 = srcTensor[0];
        srcList.src2 = srcTensor[2 * mrgElements];
        srcList.src3 = srcTensor[4 * mrgElements];
        srcList.src4 = srcTensor[6 * mrgElements];
        if (mrgGroups <= 4) {
            params.repeatTimes = 1;
            if (mrgGroups == 1) {
                break;
            } else if (mrgGroups == 2) {
                params.validBit = 0b0011;
            } else if (mrgGroups == 3) {
                params.validBit = 0b0111;
            } else if (mrgGroups == 4) {
                params.validBit = 0b1111;
            }
            AscendC::MrgSort<float>(dstTensor, srcList, params);
            i += 1;
            break;
        } else {
            params.repeatTimes = mrgGroups / 4;
            AscendC::MrgSort<float>(dstTensor, srcList, params);
            i += 1;
            mrgElements = mrgElements * 4;
            mrgGroups = mrgGroups / 4;
        }
        PipeBarrier<PIPE_V>();
    }
    if (i % 2 == 0) {
        DataCopy(src, tmp, logitsNum * 2);
        PipeBarrier<PIPE_V>();
    }
}

/**
  dst: 输出全排序的结果，排布方式为value，index
  srcValue：输入的待排序浮点数
  srcIndex：浮点数的索引
  tmp: 计算使用到的临时空间，大小为srcValue+srcIndex
  logitsNum: 排序的元素个数
 */
__aicore__ inline void SortAll(LocalTensor<float> &dst, LocalTensor<float> &srcValue, LocalTensor<uint32_t> &srcIndex,
                               LocalTensor<float> &tmpTensor, int64_t logitsNum)
{
    int64_t sort32Repeats = logitsNum / 32;
    Sort<float, true>(dst, srcValue, srcIndex, tmpTensor, sort32Repeats);
    PipeBarrier<PIPE_V>();
}

/**
  mrgDst: 合并进的Tensor
  mrgSrc: 待合并的Tensor
  tmpTensor：空间为mrgDst+mrgSrc
 */
__aicore__ inline void MergeSort(const LocalTensor<float> &mrgDst, int32_t mrgDstNum, LocalTensor<float> &mrgSrc,
                                 int32_t mrgSrcNum, LocalTensor<float> &tmpTensor)
{
    AscendC::MrgSort4Info params;
    params.elementLengths[0] = mrgDstNum;
    params.elementLengths[1] = mrgSrcNum;
    params.ifExhaustedSuspension = false;
    params.validBit = 0b0011;
    params.repeatTimes = 1;

    AscendC::MrgSortSrcList<float> srcList;
    srcList.src1 = mrgDst;
    srcList.src2 = mrgSrc;

    AscendC::MrgSort<float>(tmpTensor, srcList, params);
    PipeBarrier<PIPE_V>();
    DataCopy(mrgDst, tmpTensor, mrgDstNum * 2);
    PipeBarrier<PIPE_V>();
}

/**
 * @brief 合并基础块函数
 * @param dst 归并后的输出, 大小为blockNum * basicBlockSize * 2 * sizeof(float)
 * @param src 基本块输入
 * @param blockNum 基本块的数量
 * @param basicBlockSize 基础块的大小
 * @return 无
 */
__aicore__ inline void MrgBisicBlock(const LocalTensor<float> &dst, const LocalTensor<float> &src, int64_t blockNum,
                                     int64_t basicBlockSize)
{
    // 初始化合并排序参数
    AscendC::MrgSort4Info params;
    params.elementLengths[0] = basicBlockSize;
    params.elementLengths[1] = basicBlockSize;
    params.elementLengths[2] = basicBlockSize;
    params.elementLengths[3] = basicBlockSize;
    params.ifExhaustedSuspension = false;
    // 根据块的数量设置有效位
    if (blockNum == 2) {
        params.validBit = 0b0011;
    } else if (blockNum == 3) {
        params.validBit = 0b0111;
    } else if (blockNum == 4) {
        params.validBit = 0b1111;
    } else {
        DataCopy(dst, src, basicBlockSize * 2);
        return;
    }
    // 初始化源列表
    AscendC::MrgSortSrcList<float> srcList;
    srcList.src1 = src[0];
    srcList.src2 = src[basicBlockSize * 2 * 1];
    srcList.src3 = src[basicBlockSize * 2 * 2];
    srcList.src4 = src[basicBlockSize * 2 * 3];
    // 执行合并排序
    AscendC::MrgSort<float>(dst, srcList, params);
}


/**
 * @brief 从两个队列中选择topk
 * @param dst 已经归并好的topk数据
 * @param needsMerging 需要合并的有序数据
 * @param tmp 临时空间
 * @param topk topk的元素个数
 * @param mergSize 待合并的元素个数
 * @return 无
 */
template <bool needMrg = true>
__aicore__ inline void SelectTopK(const LocalTensor<float> &dst, const LocalTensor<float> &needsMerging,
                                  const LocalTensor<float> &tmp, int64_t topk, int64_t mergSize)
{
    // 如果不需要合并，则直接复制数据
    if (!needMrg) {
        DataCopy(dst, needsMerging, mergSize * 2);
        return;
    }
    // 初始化合并排序参数
    AscendC::MrgSort4Info params;
    params.elementLengths[0] = topk;
    params.elementLengths[1] = mergSize;
    params.ifExhaustedSuspension = (topk == mergSize);
    params.validBit = 0b0011;
    // 初始化源列表
    AscendC::MrgSortSrcList<float> srcList;
    srcList.src1 = dst;
    srcList.src2 = needsMerging;
    // 执行合并排序
    AscendC::MrgSort<float>(tmp, srcList, params);
    // 将结果复制到目标张量
    DataCopy(dst, tmp, topk * 2);
}


__aicore__ inline void ExtractIndex(const LocalTensor<uint32_t> &idxULocal, const LocalTensor<uint32_t> &sortLocal,
                                    int64_t extractNum)
{
    GatherMaskParams gatherMaskParams;
    gatherMaskParams.repeatTimes = Ceil(extractNum * sizeof(float) * 2, 256);
    gatherMaskParams.src0BlockStride = 1;
    gatherMaskParams.src0RepeatStride = 8;
    gatherMaskParams.src1RepeatStride = 0;
    uint64_t rsvdCnt = 0;    // 用于保存筛选后保留下来的元素个数
    uint8_t src1Pattern = 2; // 内置固定模式
    GatherMask(idxULocal, sortLocal, src1Pattern, false, static_cast<uint32_t>(0), gatherMaskParams, rsvdCnt);
    PipeBarrier<PIPE_V>();
}

template <HardEvent event>
__aicore__ inline void SetWaitFlag(HardEvent evt)
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
    SetFlag<event>(eventId);
    WaitFlag<event>(eventId);
}

} // namespace LightningIndexerVecNS
#endif // LIGHTNING_INDEXER_VECTOR_H