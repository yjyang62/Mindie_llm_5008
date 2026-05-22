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
#include "atb_speed/utils/task_queue.h"

namespace atb_speed {
namespace test {

TEST(TaskQueueTest, EnqueueDequeue)
{
    GlobalMockObject::verify();
    TaskQueue queue;
    int result = 0;

    // Enqueue
    queue.Enqueue([&result]() { result = 42; });
    queue.Enqueue([&result]() { result = 42; });

    // Dequeue
    auto task1 = queue.Dequeue();
    auto task2 = queue.Dequeue();
    task1();

    EXPECT_EQ(result, 42);
}

}
}