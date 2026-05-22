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
#include <sys/socket.h>
#include <netinet/in.h>
#include <libgen.h>
#include <gtest/gtest.h>
#include "mockcpp/mockcpp.hpp"
#include "http_ssl.cpp"
#define private public
#include "http_ssl.h"
#include "config_manager.h"
#include "config_manager/config_manager_impl.h"
#include "base_config_manager.h"
#include "mock_util.h"
#include "env_util.h"
#include "log.h"
#include "common_util.h"
#include "file_utils.h"
#include "memory_utils.h"

using namespace mindie_llm;

MOCKER_CPP_OVERLOAD_EQ(ServerConfig)
MOCKER_CPP_OVERLOAD_EQ(spdlog::level::level_enum)

class HttpSslTest : public testing::Test {
protected:
    void SetUp() override
    {
        serverConfig_.httpsEnabled = false;
        serverConfig_.tlsCaPath = "../../config_manager/conf/cert/";
        serverConfig_.tlsCaFile = {"ca.pem"};
        serverConfig_.tlsCert = "../../config_manager/conf/cert/server.pem";
        serverConfig_.tlsPk = "../../config_manager/conf/cert/server.key.pem";
        serverConfig_.tlsCrlPath = "../../config_manager/conf/cert/";
        serverConfig_.tlsCrlFiles = {"server_crl.pem"};
        serverConfig_.managementTlsCaFile = {"management_ca.pem"};
        serverConfig_.managementTlsCert = "../../config_manager/conf/cert/server.pem";
        serverConfig_.managementTlsPk = "../../config_manager/conf/cert/server.key.pem";
        serverConfig_.managementTlsCrlPath = "../../config_manager/conf/cert/";
        serverConfig_.managementTlsCrlFiles = {"server_crl.pem"};
        serverConfig_.inferMode = "standard";
        serverConfig_.interCommTLSEnabled = true;
        serverConfig_.interCommPort = 1121;
        serverConfig_.interCommTlsCaPath = "../../config_manager/conf/cert/";
        serverConfig_.interCommTlsCaFiles = {"ca.pem"};
        serverConfig_.interCommTlsCert = "../../config_manager/conf/cert/server.pem";
        serverConfig_.interCommPk = "../../config_manager/conf/cert/server.key.pem";
        serverConfig_.interCommTlsCrlPath = "../../config_manager/conf/cert/";
        serverConfig_.interCommTlsCrlFiles = {"server_crl.pem"};
        ctx = reinterpret_cast<X509_STORE_CTX*>(0x1);
        sslCtx = SSL_CTX_new(TLS_server_method());
        if (sslCtx == nullptr) {
            std::cout << "null ctx" << std::endl;
        }
    }

    void TearDown() override
    {
        if (sslCtx != nullptr) {
            SSL_CTX_free(sslCtx);
            sslCtx = nullptr;
        }
        GlobalMockObject::verify();
    }

    std::string GetParentDirectory()
    {
        char buffer[1024];
        try {
            return std::filesystem::current_path().string();
        } catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "Error getting current directory: " << e.what() << std::endl;
            return "";
        }
    }
    X509_CRL* crl = nullptr;
    X509_STORE_CTX* ctx = nullptr;
    X509_STORE* store = nullptr;
    ServerConfig serverConfig_;
    HttpSsl httpSsl;
    SSL_CTX* sslCtx = nullptr;
};


TEST_F(HttpSslTest, GetInstallPath_Success)
{
    std::string configPath;
    bool result = GetInstallPath(configPath);
    
    EXPECT_TRUE(result);
    
    char exePath[PATH_MAX] = {0};
    ssize_t len = readlink("/proc/self/exe", exePath, sizeof(exePath) - 1);
    ASSERT_GT(len, 0);
    exePath[len] = '\0';

    std::string expectedPath(exePath);
    size_t pos = expectedPath.find_last_of('/');
    ASSERT_NE(pos, std::string::npos);
    expectedPath = expectedPath.substr(0, pos);
    
    pos = expectedPath.find_last_of('/');
    ASSERT_NE(pos, std::string::npos);
    expectedPath = expectedPath.substr(0, pos);
    expectedPath += "/";
    
    EXPECT_EQ(configPath, expectedPath);
}

TEST_F(HttpSslTest, GetInstallPath_FAIL)
{
    MOCKER_CPP(readlink, ssize_t (*)(const char*, char*, size_t))
        .stubs()
        .will(returnValue(-1));
    std::string configPath;
    bool result = GetInstallPath(configPath);
    EXPECT_FALSE(false);
}

TEST_F(HttpSslTest, SetEnvForSecurity)
{
    std::string workDir;
    EXPECT_EQ(SetEnvForSecurity(workDir), 0);
    MOCKER_CPP(setenv, int (*)(const char*, const char*, int))
        .stubs()
        .will(returnValue(-1));
    EXPECT_EQ(SetEnvForSecurity(workDir), 1);
    MOCKER_CPP(CanonicalPath, bool (*)(std::string&))
        .stubs()
        .will(returnValue(false));
    EXPECT_EQ(SetEnvForSecurity(workDir), 1);
}

TEST_F(HttpSslTest, Start_InitTlsPath)
{
    MOCKER_CPP(GetServerConfig, const ServerConfig& (*)())
        .stubs()
        .will(returnValue(serverConfig_));
    EXPECT_EQ(httpSsl.Start(sslCtx), 0);
    EXPECT_EQ(httpSsl.InitTlsPath(serverConfig_), 0);
    EXPECT_EQ(httpSsl.Start(sslCtx, MANAGEMENT_CERT), 0);
    EXPECT_EQ(httpSsl.InitTlsPath(serverConfig_), 0);
    EXPECT_EQ(httpSsl.Start(sslCtx, METRICS_CERT), 0);
    EXPECT_EQ(httpSsl.InitTlsPath(serverConfig_), 0);
}

TEST_F(HttpSslTest, Start_Fail1)
{
    serverConfig_.httpsEnabled = true;
    MOCKER_CPP(GetServerConfig, const ServerConfig& (*)())
        .stubs()
        .will(returnValue(serverConfig_));
    MOCKER_CPP(&HttpSsl::InitWorkDir, int32_t (*)())
        .stubs()
        .will(returnValue(1));
    EXPECT_EQ(httpSsl.Start(sslCtx), 1);
}

TEST_F(HttpSslTest, Start_Fail2)
{
    serverConfig_.httpsEnabled = true;
    MOCKER_CPP(GetServerConfig, const ServerConfig& (*)())
        .stubs()
        .will(returnValue(serverConfig_));
    MOCKER_CPP(&HttpSsl::InitWorkDir, int32_t (*)())
        .stubs()
        .will(returnValue(0));
    MOCKER_CPP(&HttpSsl::InitTlsPath, int32_t (*)(ServerConfig&))
        .stubs()
        .will(returnValue(1));
    EXPECT_EQ(httpSsl.Start(sslCtx), 1);
}

TEST_F(HttpSslTest, Start_Fail3)
{
    serverConfig_.httpsEnabled = true;
    MOCKER_CPP(GetServerConfig, const ServerConfig& (*)())
        .stubs()
        .will(returnValue(serverConfig_));
    MOCKER_CPP(&HttpSsl::InitWorkDir, int32_t (*)())
        .stubs()
        .will(returnValue(0));
    MOCKER_CPP(&HttpSsl::InitTlsPath, int32_t (*)(ServerConfig&))
        .stubs()
        .will(returnValue(0));
    MOCKER_CPP(&HttpSsl::InitSSL, int32_t (*)(SSL_CTX*))
        .stubs()
        .will(returnValue(1));
    EXPECT_EQ(httpSsl.Start(sslCtx), 1);
}

TEST_F(HttpSslTest, Start_Success)
{
    serverConfig_.httpsEnabled = true;
    MOCKER_CPP(GetServerConfig, const ServerConfig& (*)())
        .stubs()
        .will(returnValue(serverConfig_));
    MOCKER_CPP(&HttpSsl::InitWorkDir, int32_t (*)())
        .stubs()
        .will(returnValue(0));
    MOCKER_CPP(&HttpSsl::InitTlsPath, int32_t (*)(ServerConfig&))
        .stubs()
        .will(returnValue(0));
    MOCKER_CPP(&HttpSsl::InitSSL, int32_t (*)(SSL_CTX*))
        .stubs()
        .will(returnValue(0));
    EXPECT_EQ(httpSsl.Start(sslCtx), 0);
}

TEST_F(HttpSslTest, InitSSL_Fail1)
{
    MOCKER_CPP(GetServerConfig, const ServerConfig& (*)())
        .stubs()
        .will(returnValue(serverConfig_));
    MOCKER_CPP(CanonicalPath, bool (*)(std::string&))
        .stubs()
        .will(returnValue(false));
    EXPECT_EQ(httpSsl.InitTlsPath(serverConfig_), 0);
    EXPECT_EQ(httpSsl.InitSSL(sslCtx), 1);
}

TEST_F(HttpSslTest, InitSSL_Fail2)
{
    MOCKER_CPP(GetServerConfig, const ServerConfig& (*)())
        .stubs()
        .will(returnValue(serverConfig_));
    EXPECT_EQ(httpSsl.InitTlsPath(serverConfig_), 0);
    EXPECT_EQ(httpSsl.InitSSL(sslCtx), 1);
}

TEST_F(HttpSslTest, InitSSL_Fail3)
{
    MOCKER_CPP(GetServerConfig, const ServerConfig& (*)())
        .stubs()
        .will(returnValue(serverConfig_));
    MOCKER_CPP(SSL_CTX_load_verify_locations, int (*)(SSL_CTX*, const char*, const char*))
        .stubs()
        .will(returnValue(1));
    EXPECT_EQ(httpSsl.InitTlsPath(serverConfig_), 0);
    EXPECT_EQ(httpSsl.InitSSL(sslCtx), 1);
}

TEST_F(HttpSslTest, InitSSL_Fail4)
{
    MOCKER_CPP(GetServerConfig, const ServerConfig& (*)())
        .stubs()
        .will(returnValue(serverConfig_));
    MOCKER_CPP(SSL_CTX_load_verify_locations, int (*)(SSL_CTX*, const char*, const char*))
        .stubs()
        .will(returnValue(1));
    MOCKER_CPP(SSL_CTX_use_certificate_file, int (*)(SSL_CTX*, const char*, int))
        .stubs()
        .will(returnValue(1));
    MOCKER_CPP(&HttpSsl::CertVerify, int (*)(X509*))
        .stubs()
        .will(returnValue(0));
    MOCKER_CPP(&HttpSsl::LoadPrivateKey, int (*)(SSL_CTX*))
        .stubs()
        .will(returnValue(0));
    EXPECT_EQ(httpSsl.InitTlsPath(serverConfig_), 0);
    EXPECT_EQ(httpSsl.InitSSL(sslCtx), 0);
}

TEST_F(HttpSslTest, LoadPrivateKey_Fail1)
{
    MOCKER_CPP(GetServerConfig, const ServerConfig& (*)())
        .stubs()
        .will(returnValue(serverConfig_));
    MOCKER_CPP(CanonicalPath, bool (*)(std::string&))
        .stubs()
        .will(returnValue(false));
    EXPECT_EQ(httpSsl.InitTlsPath(serverConfig_), 0);
    EXPECT_EQ(httpSsl.LoadPrivateKey(sslCtx), 1);
}

TEST_F(HttpSslTest, LoadPrivateKey_Fail2)
{
    MOCKER_CPP(GetServerConfig, const ServerConfig& (*)())
        .stubs()
        .will(returnValue(serverConfig_));
    MOCKER_CPP(SSL_CTX_use_PrivateKey_file, int (*)(SSL_CTX*, const char*, int type))
        .stubs()
        .will(returnValue(-1));
    EXPECT_EQ(httpSsl.LoadPrivateKey(sslCtx), 1);
    EXPECT_EQ(httpSsl.InitTlsPath(serverConfig_), 0);
    EXPECT_EQ(httpSsl.LoadPrivateKey(sslCtx), 1);
}

TEST_F(HttpSslTest, LoadPrivateKey_Fail3)
{
    MOCKER_CPP(GetServerConfig, const ServerConfig& (*)())
        .stubs()
        .will(returnValue(serverConfig_));
    MOCKER_CPP(SSL_CTX_use_PrivateKey_file, int (*)(SSL_CTX*, const char*, int type))
        .stubs()
        .will(returnValue(1));
    EXPECT_EQ(httpSsl.InitTlsPath(serverConfig_), 0);
    EXPECT_EQ(httpSsl.LoadPrivateKey(sslCtx), 1);
}

TEST_F(HttpSslTest, LoadPrivateKey_Fail4)
{
    MOCKER_CPP(GetServerConfig, const ServerConfig& (*)())
        .stubs()
        .will(returnValue(serverConfig_));
    MOCKER_CPP(SSL_CTX_use_PrivateKey_file, int (*)(SSL_CTX*, const char*, int type))
        .stubs()
        .will(returnValue(1));
    MOCKER_CPP(SSL_CTX_check_private_key, int (*)(const SSL_CTX*))
        .stubs()
        .will(returnValue(-1));
    EXPECT_EQ(httpSsl.InitTlsPath(serverConfig_), 0);
    EXPECT_EQ(httpSsl.LoadPrivateKey(sslCtx), 1);
}

TEST_F(HttpSslTest, LoadPrivateKey_Success)
{
    MOCKER_CPP(GetServerConfig, const ServerConfig& (*)())
        .stubs()
        .will(returnValue(serverConfig_));
    MOCKER_CPP(SSL_CTX_use_PrivateKey_file, int (*)(SSL_CTX*, const char*, int type))
        .stubs()
        .will(returnValue(1));
    MOCKER_CPP(SSL_CTX_check_private_key, int (*)(const SSL_CTX*))
        .stubs()
        .will(returnValue(1));
    EXPECT_EQ(httpSsl.InitTlsPath(serverConfig_), 0);
    EXPECT_EQ(httpSsl.LoadPrivateKey(sslCtx), 0);
}

TEST_F(HttpSslTest, LoadCertRevokeListFile_Fail1)
{
    MOCKER_CPP(realpath, char* (*)(const char*, char*))
        .stubs()
        .will(returnValue(nullptr));
    GlobalMockObject::verify();
    EXPECT_EQ(LoadCertRevokeListFile(""), nullptr);
    static char* resolvedPath = strdup("/../../config_manager/conf/cert");
    MOCKER_CPP(realpath, char* (*)(const char*, char*))
        .stubs()
        .will(returnValue(resolvedPath));
    BIO* fakeBio = nullptr;
    MOCKER_CPP(BIO_new, BIO* (*)(const BIO_METHOD*))
        .stubs()
        .will(returnValue(fakeBio));
    EXPECT_EQ(LoadCertRevokeListFile(""), nullptr);
}

TEST_F(HttpSslTest, LoadCertRevokeListFile_Fail2)
{
    static char* resolvedPath = strdup("/../../config_manager/conf/cert");
    MOCKER_CPP(realpath, char* (*)(const char*, char*))
        .stubs()
        .will(returnValue(resolvedPath));
    EXPECT_EQ(LoadCertRevokeListFile(""), nullptr);
}

TEST_F(HttpSslTest, LoadCertRevokeListFile_Fail3)
{
    static char* resolvedPath = strdup("/../../config_manager/conf/cert");
    MOCKER_CPP(realpath, char* (*)(const char*, char*))
        .stubs()
        .will(returnValue(resolvedPath));
    MOCKER_CPP(BIO_ctrl, long (*)(BIO*, int, long, void*))
        .stubs()
        .will(returnValue(1));
    EXPECT_EQ(LoadCertRevokeListFile(""), nullptr);
}

TEST_F(HttpSslTest, LoadCertRevokeListFile_PemReadFail)
{
    static char* resolvedPath = strdup("/valid/path");
    MOCKER_CPP(realpath, char* (*)(const char*, char*))
        .stubs()
        .will(returnValue(resolvedPath));
    MOCKER_CPP(BIO_ctrl, long (*)(BIO*, int, long, void*))
        .stubs()
        .will(returnValue(1));
    static X509_CRL* fakeCrl = reinterpret_cast<X509_CRL*>(0x2);
    MOCKER_CPP(PEM_read_bio_X509_CRL, X509_CRL* (*)(BIO*, X509_CRL**, pem_password_cb*, void*))
        .stubs()
        .will(returnValue(fakeCrl));
    MOCKER_CPP(BIO_free, int (*)(BIO*))
        .stubs()
        .will(returnValue(1));
    EXPECT_EQ(LoadCertRevokeListFile(""), fakeCrl);
}

TEST_F(HttpSslTest, CaVerifyCallback_NullParameters)
{
    EXPECT_EQ(httpSsl.CaVerifyCallback(nullptr, (void*)"arg"), 0);
    EXPECT_EQ(httpSsl.CaVerifyCallback(ctx, nullptr), 0);
}

TEST_F(HttpSslTest, CaVerifyCallback_EmptyCrlPath)
{

    MOCKER_CPP(X509_verify_cert, int (*)(X509_STORE_CTX*))
        .stubs()
        .will(returnValue(1));
    
    EXPECT_EQ(httpSsl.CaVerifyCallback(ctx, (void*)""), 1);
}

TEST_F(HttpSslTest, CaVerifyCallback_GetStoreFailed)
{
    const char* crlPath = "path1,path2";
    MOCKER_CPP(X509_STORE_CTX_get0_store, X509_STORE* (*)(const X509_STORE_CTX*))
        .stubs()
        .will(returnValue(store));
    EXPECT_EQ(httpSsl.CaVerifyCallback(ctx, (void*)crlPath), -1);
}

TEST_F(HttpSslTest, CaVerifyCallback_LoadCrlFailed)
{
    const char* crlPath = "path1,path2";
    store = reinterpret_cast<X509_STORE*>(0x3);
    MOCKER_CPP(X509_STORE_CTX_get0_store, X509_STORE* (*)(const X509_STORE_CTX*))
        .stubs()
        .will(returnValue(store));
    MOCKER_CPP(LoadCertRevokeListFile, X509_CRL* (*)(const char*))
        .stubs()
        .will(returnValue(crl));
    MOCKER_CPP(X509_STORE_CTX_set_flags, void (*)(X509_STORE_CTX*, long)).stubs();
    EXPECT_EQ(httpSsl.CaVerifyCallback(ctx, (void*)crlPath), -1);
}

TEST_F(HttpSslTest, CaVerifyCallback_AddCrlFailed)
{
    const char* crlPath = "path1,path2";
    store = reinterpret_cast<X509_STORE*>(0x3);
    MOCKER_CPP(X509_STORE_CTX_get0_store, X509_STORE* (*)(X509_STORE_CTX*))
        .stubs()
        .will(returnValue(store));
    crl = reinterpret_cast<X509_CRL*>(0x4);
    MOCKER_CPP(LoadCertRevokeListFile, X509_CRL* (*)(const char*))
        .stubs()
        .will(returnValue(crl));
    MOCKER_CPP(X509_STORE_add_crl, int (*)(X509_STORE*, X509_CRL*))
        .stubs()
        .will(returnValue(0));
    const ASN1_TIME* as = reinterpret_cast<ASN1_TIME*>(0x5);
    MOCKER_CPP(X509_CRL_get0_nextUpdate, const ASN1_TIME* (*)(const X509_CRL*))
        .stubs()
        .will(returnValue(as));
    MOCKER_CPP(X509_cmp_current_time, int (*)(const ASN1_TIME*))
        .stubs()
        .will(returnValue(0));
    MOCKER_CPP(X509_STORE_CTX_set_flags, void (*)(X509_STORE_CTX*, long)).stubs();
    MOCKER_CPP(X509_CRL_free, void (*)(X509_CRL*)).stubs();
    EXPECT_EQ(httpSsl.CaVerifyCallback(ctx, (void*)crlPath), -1);
}

TEST_F(HttpSslTest, CaVerifyCallback_CrlExpired)
{
    const char* crlPath = "path1,path2";
    store = reinterpret_cast<X509_STORE*>(0x3);
    MOCKER_CPP(X509_STORE_CTX_get0_store, X509_STORE* (*)(X509_STORE_CTX*))
        .stubs()
        .will(returnValue(store));
    crl = reinterpret_cast<X509_CRL*>(0x4);
    MOCKER_CPP(LoadCertRevokeListFile, X509_CRL* (*)(const char*))
        .stubs()
        .will(returnValue(crl));
    const ASN1_TIME* as = reinterpret_cast<ASN1_TIME*>(0x5);
    MOCKER_CPP(X509_CRL_get0_nextUpdate, const ASN1_TIME* (*)(const X509_CRL*))
        .stubs()
        .will(returnValue(as));
    MOCKER_CPP(X509_cmp_current_time, int (*)(const ASN1_TIME*))
        .stubs()
        .will(returnValue(0));
    MOCKER_CPP(X509_STORE_CTX_set_flags, void (*)(X509_STORE_CTX*, long)).stubs();
    MOCKER_CPP(X509_CRL_free, void (*)(X509_CRL*)).stubs();
    MOCKER_CPP(X509_STORE_add_crl, int (*)(X509_STORE*, X509_CRL*))
        .stubs()
        .will(returnValue(1));
    MOCKER_CPP(X509_cmp_current_time, int (*)(const ASN1_TIME*))
        .stubs()
        .will(returnValue(-1));
    MOCKER_CPP(X509_verify_cert, int (*)(X509_STORE_CTX*))
        .stubs()
        .will(returnValue(1));
    
    EXPECT_EQ(httpSsl.CaVerifyCallback(ctx, (void*)crlPath), 1);
}

TEST_F(HttpSslTest, CaVerifyCallback_VerifyFailed)
{
    const char* crlPath = "path1,path2";
    store = reinterpret_cast<X509_STORE*>(0x3);
    MOCKER_CPP(X509_STORE_CTX_get0_store, X509_STORE* (*)(X509_STORE_CTX*))
        .stubs()
        .will(returnValue(store));
    crl = reinterpret_cast<X509_CRL*>(0x4);
    MOCKER_CPP(LoadCertRevokeListFile, X509_CRL* (*)(const char*))
        .stubs()
        .will(returnValue(crl));
    MOCKER_CPP(X509_STORE_add_crl, int (*)(X509_STORE*, X509_CRL*))
        .stubs()
        .will(returnValue(1));
    const ASN1_TIME* as = reinterpret_cast<ASN1_TIME*>(0x5);
    MOCKER_CPP(X509_CRL_get0_nextUpdate, const ASN1_TIME* (*)(const X509_CRL*))
        .stubs()
        .will(returnValue(as));
    MOCKER_CPP(X509_STORE_CTX_set_flags, void (*)(X509_STORE_CTX*, long)).stubs();
    MOCKER_CPP(X509_CRL_free, void (*)(X509_CRL*)).stubs();
    MOCKER_CPP(X509_cmp_current_time, int (*)(const ASN1_TIME*))
        .stubs()
        .will(returnValue(1));
    MOCKER_CPP(X509_verify_cert, int (*)(X509_STORE_CTX*))
        .stubs()
        .will(returnValue(0));
    MOCKER_CPP(X509_STORE_CTX_get_error, int (*)(X509_STORE_CTX*))
        .stubs()
        .will(returnValue(X509_V_ERR_CERT_REVOKED));
    const char* revoked = "Certificate revoked";
    MOCKER_CPP(X509_verify_cert_error_string, const char* (*)(long))
        .stubs()
        .will(returnValue(revoked));
    EXPECT_EQ(httpSsl.CaVerifyCallback(ctx, (void*)crlPath), -1);
}

TEST_F(HttpSslTest, CaVerifyCallback_Success)
{
    const char* crlPath = "path1,path2";
    store = reinterpret_cast<X509_STORE*>(0x3);
    MOCKER_CPP(X509_STORE_CTX_get0_store, X509_STORE* (*)(X509_STORE_CTX*))
        .stubs()
        .will(returnValue(store));
    crl = reinterpret_cast<X509_CRL*>(0x4);
    MOCKER_CPP(LoadCertRevokeListFile, X509_CRL* (*)(const char*))
        .stubs()
        .will(returnValue(crl));
    MOCKER_CPP(X509_STORE_add_crl, int (*)(X509_STORE*, X509_CRL*))
        .stubs()
        .will(returnValue(1));
    const ASN1_TIME* as = reinterpret_cast<ASN1_TIME*>(0x5);
    MOCKER_CPP(X509_CRL_get0_nextUpdate, const ASN1_TIME* (*)(const X509_CRL*))
        .stubs()
        .will(returnValue(as));
    MOCKER_CPP(X509_STORE_CTX_set_flags, void (*)(X509_STORE_CTX*, long)).stubs();
    MOCKER_CPP(X509_CRL_free, void (*)(X509_CRL*)).stubs();
    MOCKER_CPP(X509_cmp_current_time, int (*)(const ASN1_TIME*))
        .stubs()
        .will(returnValue(1));
    MOCKER_CPP(X509_verify_cert, int (*)(X509_STORE_CTX*))
        .stubs()
        .will(returnValue(1));
    EXPECT_EQ(httpSsl.CaVerifyCallback(ctx, (void*)crlPath), 1);
}