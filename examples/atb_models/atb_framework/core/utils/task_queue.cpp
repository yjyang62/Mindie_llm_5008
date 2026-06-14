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
#include "atb_speed/utils/task_queue.h"
namespace atb_speed {
void TaskQueue::Enqueue(const Task &task)
{
    std::unique_lock<std::mutex> lock(mutex_);
    queue_.push(task);
    lock.unlock();
    cv_.notify_one();
    return;
}

Task TaskQueue::Dequeue()
{
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] {return !queue_.empty();});
    auto task = queue_.front();
    queue_.pop();
    return task;
}
} // namespace atb_speed