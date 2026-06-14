/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
/*
Input to this function is a string containing multiple groups of nested parentheses. Your goal is to
separate those group into separate strings and return the vector of those.
Separate groups are balanced (each open brace is properly closed) and not nested within each other
Ignore any spaces in the input string.
>>> separate_paren_groups("( ) (( )) (( )( ))")
{"()", "(())", "(()())"}
*/
#include <cstdio>
#include <vector>
#include <string>

namespace {

    std::vector<std::string> separate_paren_groups(std::string paren_string)
    {
        std::vector<std::string> all_parens;
        std::string current_paren;
        int level = 0;
        char chr;
        int i;

        for (i = 0; i < paren_string.length(); i++) {
            chr = paren_string[i];
            if (chr == '(') {
                level += 1;
                current_paren += chr;
            }
            if (chr == ')') {
                level -= 1;
                current_paren += chr;
                if (level == 0) {
                    all_parens.push_back(current_paren);
                    current_paren = "";
                }
            }
        }
        return all_parens;
    }

#undef NDEBUG
#include <cassert>

    static bool g_issame(std::vector<std::string> a, std::vector<std::string> b)
    {
        if (a.size() != b.size()) return false;
        for (int i = 0; i < a.size(); i++) {
            if (a[i] != b[i]) return false;
        }
        return true;
    }
} // namespace

int main()
{
    assert(g_issame(separate_paren_groups("(()()) ((())) () ((())()())"), {"(()())", "((()))", "()", "((())()())"}));
    assert(g_issame(separate_paren_groups("() (()) ((())) (((())))"), {"()", "(())", "((()))", "(((())))"}));
    assert(g_issame(separate_paren_groups("(()(())((())))"), {"(()(())((())))"}));
    assert(g_issame(separate_paren_groups("( ) (( )) (( )( ))"), {"()", "(())", "(()())"}));
}