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

#ifndef __SCENE_H__
#define __SCENE_H__

#include <string>
#include "noncopyable.h"

namespace dlt::framework {
class Scene : private NonCopyable {
public:
    static Scene &GetInstance();
    bool LoadScene();

private:
    void LoadNode(int nodeCount, std::string nodeType);
    std::string GetWorkPath(const std::string fileName) const;
};
} // namespace dlt::framework

#endif // __SCENE_H__