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
 
#include <gtest/gtest.h>
#include "obj_pool.h"
#include "hashless_block_obj.h"

namespace mindie_llm {
class BlockObjPoolTest : public ::testing::Test {
protected:
    void SetUp() override {}

    uint64_t poolSize_{2};
    ObjPool<BlockObj> hashlessBlockObjPool_{
        ObjPool<BlockObj>(poolSize_, []() { return std::make_shared<HashLessBlockObj>(); })};
};

TEST_F(BlockObjPoolTest, ShouldMinusFreeObjNumWhenAcquireObj)
{
    auto obj = hashlessBlockObjPool_.AcquireObj();
    EXPECT_EQ(typeid(*obj), typeid(HashLessBlockObj));
    EXPECT_EQ(hashlessBlockObjPool_.GetFreeObjNum(), poolSize_ - 1);
}

TEST_F(BlockObjPoolTest, ShouldAddFreeObjNumWhenFreeObj)
{
    auto obj = hashlessBlockObjPool_.AcquireObj();
    hashlessBlockObjPool_.FreeObj(obj);
    EXPECT_EQ(hashlessBlockObjPool_.GetFreeObjNum(), poolSize_);
    EXPECT_EQ(obj, nullptr);
}

TEST_F(BlockObjPoolTest, ShouldThrowErrorWhenPoolIsFull)
{
    std::shared_ptr<BlockObj> obj = std::make_shared<HashLessBlockObj>();
    EXPECT_THROW(hashlessBlockObjPool_.FreeObj(obj), std::runtime_error);
}

TEST_F(BlockObjPoolTest, ShouldIncreaseCapcityWhenPoolIsEmpty)
{
    for (size_t i = 0; i < poolSize_; i++) {
        hashlessBlockObjPool_.AcquireObj();
    }
    hashlessBlockObjPool_.AcquireObj();
    EXPECT_EQ(hashlessBlockObjPool_.GetFreeObjNum(), poolSize_ - 1);
    uint64_t newPoolSize = poolSize_ * 2;
    EXPECT_EQ(hashlessBlockObjPool_.GetPoolSize(), newPoolSize);
}
} // namespace mindie_llm