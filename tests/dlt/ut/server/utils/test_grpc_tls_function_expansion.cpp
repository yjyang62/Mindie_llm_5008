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
#include <libgen.h>
#include <fstream>
#include <cstdio>
#include <string>
#include <vector>
#include <memory>
#include <filesystem>
#include <openssl/x509.h>
#include <openssl/pem.h>
#define private public
#include "grpc_tls_function_expansion.h"

using namespace mindie_llm;

class GrpcTlsFunctionExpansionTest : public testing::Test {
protected:
    void SetUp() override
    {
        validCert = std::tmpnam(nullptr);
        {
            std::ofstream out(validCert);
            out << "Valid Cert Content";
        }
        validCrl = std::tmpnam(nullptr);
        {
            std::ofstream out(validCrl);
            out << "Valid CRL Content";
        }
    }

    void TearDown() override
    {
        std::remove(validCert.c_str());
        std::remove(validCrl.c_str());
        GlobalMockObject::verify();
    }

    std::string validCert;
    std::string validCrl;

    std::string GetParentDirectory()
    {
        char buffer[1024];
        try {
            return std::filesystem::current_path().string();
        } catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "Error getting current directory: " << e.what() << std::endl;
            return "";
        }

        std::unique_ptr<char, decltype(&std::free)> temp(strdup(buffer), &std::free);
        
        if (temp == nullptr) {
            std::cerr << "Memory allocation failed" << std::endl;
            return "";
        }

        char* parent = dirname(temp.get());
        std::string result(parent);
        return result;
    }
};

TEST_F(GrpcTlsFunctionExpansionTest, CheckCert_ValidCert)
{
    EXPECT_FALSE(GrpcTlsFunctionExpansion::CheckCert(validCert));
}

TEST_F(GrpcTlsFunctionExpansionTest, CheckCert_FileNotExist)
{
    EXPECT_FALSE(GrpcTlsFunctionExpansion::CheckCert("non_existent_file.pem"));
}

TEST_F(GrpcTlsFunctionExpansionTest, CheckTlsOption_InvalidCRL)
{
    MOCKER(PEM_read_X509_CRL)
        .stubs()
        .will(returnValue(reinterpret_cast<X509_CRL*>(0xCAFEBABE)));
    
    std::vector<std::string> validCa = {validCert};
    std::vector<std::string> invalidCrls = {validCrl, "invalid.crl"};
    
    EXPECT_FALSE(GrpcTlsFunctionExpansion::CheckTlsOption(
        validCa, validCert, invalidCrls));
}

TEST_F(GrpcTlsFunctionExpansionTest, CheckTlsOption_EmptyCA)
{
    std::vector<std::string> emptyCa;
    std::vector<std::string> validCrls = {validCrl};
    EXPECT_FALSE(GrpcTlsFunctionExpansion::CheckTlsOption(
        emptyCa, validCert, validCrls));
    MOCKER_CPP(&GrpcTlsFunctionExpansion::CheckCert, bool(*)(const std::string&))
        .stubs()
        .will(returnValue(true));
    EXPECT_FALSE(GrpcTlsFunctionExpansion::CheckTlsOption(
        emptyCa, validCert, validCrls));
    MOCKER_CPP(&GrpcTlsFunctionExpansion::CheckCrl, bool(*)(const std::string&))
        .stubs()
        .will(returnValue(true));
    EXPECT_TRUE(GrpcTlsFunctionExpansion::CheckTlsOption(
        emptyCa, validCert, validCrls));
}

TEST_F(GrpcTlsFunctionExpansionTest, CheckCRL)
{
    EXPECT_FALSE(GrpcTlsFunctionExpansion::CheckCrl("InvalidCa"));
    EXPECT_FALSE(GrpcTlsFunctionExpansion::CheckCrl(GetParentDirectory() + "/../../config_manager/conf/config.json"));
}

TEST_F(GrpcTlsFunctionExpansionTest, X509CertDeleterCoverage)
{
    X509* cert = X509_new();
    ASSERT_NE(cert, nullptr);

    X509CertDeleter deleter;
    deleter(cert);
}

TEST_F(GrpcTlsFunctionExpansionTest, X509CrlDeleterCoverage)
{
    X509_CRL* crl = X509_CRL_new();
    ASSERT_NE(crl, nullptr);

    X509CrlDeleter deleter;
    deleter(crl);
}