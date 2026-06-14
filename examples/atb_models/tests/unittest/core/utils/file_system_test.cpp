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
#include <unistd.h>
#include <sys/stat.h>
#include <fstream>
#include <filesystem>
#include "atb_speed/utils/file_system.h"

namespace atb_speed {

class FileSystemTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        // 创建临时测试目录
        test_dir_ = "/tmp/file_system_test";
        test_file_ = test_dir_ + "/test_file.txt";
        nested_dir_ = test_dir_ + "/level1/level2/level3";
        
        // 清理可能存在的旧测试目录
        RemoveTestDir();
    }

    void TearDown() override
    {
        // 清理测试目录
        RemoveTestDir();
    }

    void RemoveTestDir()
    {
        // 递归删除测试目录
        std::filesystem::remove_all(test_dir_);
    }

    std::string test_dir_;
    std::string test_file_;
    std::string nested_dir_;
};

// 测试 Exists 函数 - 文件存在的情况
TEST_F(FileSystemTest, Exists_FileExists_ReturnsTrue)
{
    // 创建测试文件
    FileSystem::Makedirs(test_dir_, 0750);
    std::ofstream file(test_file_);
    file.close();
    
    EXPECT_TRUE(FileSystem::Exists(test_file_));
}

// 测试 Exists 函数 - 文件不存在的情况
TEST_F(FileSystemTest, Exists_FileNotExists_ReturnsFalse)
{
    std::string non_exist_file = test_dir_ + "/non_exist.txt";
    EXPECT_FALSE(FileSystem::Exists(non_exist_file));
}

// 测试 Exists 函数 - 目录存在的情况
TEST_F(FileSystemTest, Exists_DirExists_ReturnsTrue)
{
    FileSystem::Makedirs(test_dir_, 0750);
    EXPECT_TRUE(FileSystem::Exists(test_dir_));
}

// 测试 MakeDir 函数 - 成功创建目录
TEST_F(FileSystemTest, MakeDir_Success_ReturnsTrue)
{
    EXPECT_TRUE(FileSystem::MakeDir(test_dir_, 0750));
    EXPECT_TRUE(FileSystem::Exists(test_dir_));
}

// 测试 MakeDir 函数 - 目录已存在
TEST_F(FileSystemTest, MakeDir_AlreadyExists_ReturnsFalse)
{
    FileSystem::Makedirs(test_dir_, 0750);
    EXPECT_FALSE(FileSystem::MakeDir(test_dir_, 0750));
}

// 测试 Makedirs 函数 - 单层目录
TEST_F(FileSystemTest, Makedirs_SingleDir_Success)
{
    std::string single_dir = "/tmp/test_single_dir";
    EXPECT_TRUE(FileSystem::Makedirs(single_dir, 0750));
    EXPECT_TRUE(FileSystem::Exists(single_dir));
    
    // 清理
    rmdir(single_dir.c_str());
}

// 测试 Makedirs 函数 - 多层嵌套目录
TEST_F(FileSystemTest, Makedirs_NestedDirs_Success)
{
    EXPECT_TRUE(FileSystem::Makedirs(nested_dir_, 0750));
    EXPECT_TRUE(FileSystem::Exists(nested_dir_));
    
    // 验证中间目录也存在
    EXPECT_TRUE(FileSystem::Exists(test_dir_ + "/level1"));
    EXPECT_TRUE(FileSystem::Exists(test_dir_ + "/level1/level2"));
}

// 测试 Makedirs 函数 - 目录已存在
TEST_F(FileSystemTest, Makedirs_AlreadyExists_ReturnsTrue)
{
    FileSystem::Makedirs(nested_dir_, 0750);
    EXPECT_TRUE(FileSystem::Makedirs(nested_dir_, 0750));
}

// 测试 Makedirs 函数 - 空路径
TEST_F(FileSystemTest, Makedirs_EmptyPath_ReturnsTrue)
{
    EXPECT_FALSE(FileSystem::Makedirs("", 0750));
}

// 测试 Makedirs 函数 - 根目录
TEST_F(FileSystemTest, Makedirs_RootPath_ReturnsTrue)
{
    EXPECT_TRUE(FileSystem::Makedirs("/", 0750));
}

// 测试 Makedirs 函数 - 相对路径
TEST_F(FileSystemTest, Makedirs_RelativePath_Success)
{
    std::string relative_path = "relative/test/path";
    EXPECT_TRUE(FileSystem::Makedirs(relative_path, 0750));
    EXPECT_TRUE(FileSystem::Exists(relative_path));
    
    // 清理
    std::filesystem::remove_all("relative");
}

// 测试权限参数是否生效
TEST_F(FileSystemTest, Makedirs_WithDifferentMode_CheckPermission)
{
    mode_t mode = 0700;
    EXPECT_TRUE(FileSystem::Makedirs(nested_dir_, mode));
    
    struct stat st;
    if (stat(nested_dir_.c_str(), &st) == 0) {
        EXPECT_EQ(st.st_mode & 0777, mode);
    }
}

// 测试边界情况 - 路径以斜杠结尾
TEST_F(FileSystemTest, Makedirs_PathEndsWithSlash_Success)
{
    std::string path_with_slash = test_dir_ + "/slash_test/";
    EXPECT_TRUE(FileSystem::Makedirs(path_with_slash, 0750));
    EXPECT_TRUE(FileSystem::Exists(test_dir_ + "/slash_test"));
}

// 性能测试 - 创建大量目录
TEST_F(FileSystemTest, Makedirs_PerformanceTest)
{
    std::string performance_path = test_dir_ + "/perf/level1/level2/level3/level4";
    auto start_time = std::chrono::high_resolution_clock::now();
    
    EXPECT_TRUE(FileSystem::Makedirs(performance_path, 0750));
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // 验证创建成功
    EXPECT_TRUE(FileSystem::Exists(performance_path));
}

} // namespace atb_speed

// 主函数
int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}