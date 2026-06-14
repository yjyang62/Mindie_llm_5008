/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#ifndef ATB_SPEED_TASK_QUEUE_H
#define ATB_SPEED_TASK_QUEUE_H

#include <functional>
#include <mutex>
#include <condition_variable>
#include <queue>

namespace atb_speed {

using Task = std::function<void()>;

class TaskQueue {
public:
    void Enqueue(const Task &task);
    Task Dequeue();

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<Task> queue_;
};
}  // namespace atb_speed

#endif  // ATB_SPEED_TASK_QUEUE_H
