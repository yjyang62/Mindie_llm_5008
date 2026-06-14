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
 
#include "crash_handler.h"

#include <execinfo.h>
#include <signal.h>
#include <unistd.h>
#include <libgen.h>
#include <dlfcn.h>
#include <cxxabi.h>

#include <iostream>
#include <memory>
#include <thread>
#include <cstdio>
#include <string>
#include <cstring>
#include <array>
#include <vector>
#include <sstream>

namespace mindie_llm {
namespace test {

namespace {
    thread_local std::string g_context = "none";

    struct ModuleInfo {
        std::string path;      // module file path (dli_fname)
        uintptr_t base = 0;    // dli_fbase
    };

    ModuleInfo GetModuleInfo(void* addr)
    {
        ModuleInfo info;
        Dl_info dlinfo;
        if (dladdr(addr, &dlinfo) && dlinfo.dli_fname) {
            info.path = dlinfo.dli_fname;
            info.base = reinterpret_cast<uintptr_t>(dlinfo.dli_fbase);
        } else {
            // 回退读取 /proc/self/exe
            std::array<char, 256> exe_path{};
            ssize_t len = readlink("/proc/self/exe", exe_path.data(), exe_path.size() - 1);
            if (len != -1) {
                exe_path[static_cast<size_t>(len)] = '\0';
                info.path = exe_path.data();
                info.base = 0;
            }
        }
        return info;
    }

    // 使用 addr2line 解析（传入 module_path 和 相对地址）
    std::string Addr2Line(const std::string& module_path, uintptr_t relative_addr)
    {
        if (module_path.empty()) return {};

        std::ostringstream cmdoss;
        // -C demangle, -f show function, use hex address
        cmdoss << "addr2line -C -f -e \"" << module_path << "\" 0x"
               << std::hex << relative_addr << " 2>/dev/null";
        std::string cmd = cmdoss.str();

        std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), &pclose);
        if (!pipe) return {};

        std::array<char, 512> buf{};
        std::string out;
        if (fgets(buf.data(), static_cast<int>(buf.size()), pipe.get()) != nullptr) {
            out = buf.data();
            if (fgets(buf.data(), static_cast<int>(buf.size()), pipe.get()) != nullptr) {
                out += buf.data();
            }
        }
        return out;
    }

    void PrintBacktrace()
    {
        constexpr int MAX = 64;
        std::vector<void*> buffer(MAX);
        int nptrs = backtrace(buffer.data(), MAX);
        if (nptrs <= 0) return;

        char** strings = backtrace_symbols(buffer.data(), nptrs);
        std::unique_ptr<char*, void(*)(void*)> symbols_guard(strings, &free);

        std::cerr << "===== Stack Trace with Source Lines =====\n";
        for (int i = 0; i < nptrs; ++i) {
            // 原始帧输出
            if (strings && strings[i]) {
                std::cerr << i << ": " << strings[i] << "\n";
            } else {
                std::cerr << i << ": (null)\n";
            }

            // 获取模块信息（可执行或 so）并计算相对偏移
            ModuleInfo mod = GetModuleInfo(buffer[i]);
            if (!mod.path.empty()) {
                uintptr_t abs_addr = reinterpret_cast<uintptr_t>(buffer[i]);
                uintptr_t rel = (abs_addr >= mod.base) ? (abs_addr - mod.base) : abs_addr;

                std::string src = Addr2Line(mod.path, rel);
                if (!src.empty()) {
                    if (src.find("??:0") == std::string::npos && src.find("??") == std::string::npos) {
                        std::cerr << "     -> " << src;
                    } else {
                        std::cerr << "     -> <source not available>\n";
                    }
                } else {
                    std::cerr << "     -> <source not available>\n";
                }
            } else {
                std::cerr << "     -> <module info not available>\n";
            }
        }
        std::cerr << "========================================\n";
    }

    void SignalHandler(int sig)
    {
        const char* name = "UNKNOWN";
        switch (sig) {
            case SIGSEGV: name = "SIGSEGV"; break;
            case SIGABRT: name = "SIGABRT"; break;
            case SIGFPE:  name = "SIGFPE";  break;
            case SIGILL:  name = "SIGILL";  break;
            default:      name = "Unknown signal"; break;
        }
        std::cerr << "\n*** CRASH: " << name << " ***\n";
        std::cerr << "Thread: " << std::this_thread::get_id() << "\n";
        std::cerr << "Context: " << g_context << "\n";
        PrintBacktrace();
        _exit(1);
    }
} // namespace

void InitCrashHandler()
{
    static bool done = false;
    if (done) return;

    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = SignalHandler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;

    sigaction(SIGSEGV, &sa, nullptr);
    sigaction(SIGABRT, &sa, nullptr);
    sigaction(SIGFPE,  &sa, nullptr);
    sigaction(SIGILL,  &sa, nullptr);

    done = true;
}

void SetCurrentContext(const std::string& ctx) { g_context = ctx; }
std::string GetCurrentContext() { return g_context; }

} // namespace test
} // namespace mindie_llm
