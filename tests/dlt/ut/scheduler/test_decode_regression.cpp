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
#include <random>
#include "decode_regression.h"


using namespace mindie_llm;

class DecodeRegressionTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        decodeRegression_ = std::make_shared<DecodeRegression>();
    }
    
    // 生成一个double类型随机数
    static double randomGen()
    {
        static std::random_device seed;
        static std::mt19937 engine(seed());
        static std::uniform_real_distribution<> distrib(2, 10);
        return distrib(engine);
    }

    // 判断两个二维数组（矩阵）是否相等
    static void IsMatrixEqual(const std::vector<std::vector<double>>& matrixA,
                              const std::vector<std::vector<double>>& matrixB)
    {
        EXPECT_EQ(matrixA.size(), matrixB.size());
        for (size_t i = 0; i < matrixA.size(); i++) {
            EXPECT_EQ(matrixA[i].size(), matrixB[i].size());
        }
        size_t rows = matrixA.size(), cols = matrixA[0].size();
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                EXPECT_NEAR(matrixA[i][j], matrixB[i][j], 0.05);
            }
        }
    }

    std::shared_ptr<DecodeRegression> decodeRegression_;
};

// 测试转置操作的有效性
TEST_F(DecodeRegressionTest, TestTranspose)
{
    size_t rows = 5, cols = 10;
    std::vector<std::vector<double>> matrix(rows, std::vector<double>(cols));
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            matrix[i][j] = randomGen();
        }
    }

    std::vector<std::vector<double>> expectedResult(cols, std::vector<double>(rows));
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            expectedResult[j][i] = matrix[i][j];
        }
    }
    IsMatrixEqual(expectedResult, decodeRegression_->Transpose(matrix));
}

// 测试矩阵乘操作的有效性
TEST_F(DecodeRegressionTest, TestMultiply)
{
    size_t rowsA = 5, colsA = 10, rowsB = 10, colsB = 20;
    std::vector<std::vector<double>> matrixA(rowsA, std::vector<double>(colsA));
    std::vector<std::vector<double>> matrixB(rowsB, std::vector<double>(colsB));

    std::vector<std::vector<double>> expectedResult(rowsA, std::vector<double>(colsB));

    for (size_t i = 0; i < rowsA; i++) {
        for (size_t j = 0; j < colsA; j++) {
            matrixA[i][j] = randomGen();
        }
    }

    for (size_t i = 0; i < rowsB; i++) {
        for (size_t j = 0; j < colsB; j++) {
            matrixB[i][j] = randomGen();
        }
    }

    for (size_t i = 0; i < rowsA; i++) {
        for (size_t j = 0; j < colsB; j++) {
            for (size_t k = 0; k < colsA; k++) {
                expectedResult[i][j] += matrixA[i][k] * matrixB[k][j];
            }
        }
    }
    IsMatrixEqual(expectedResult, decodeRegression_->Multiply(matrixA, matrixB));
}

// 测试求逆操作的有效性
TEST_F(DecodeRegressionTest, TestInverse3by3)
{
    size_t dim = 3;
    std::vector<std::vector<double>> inputMatrix(dim, std::vector<double>(dim));
    std::vector<std::vector<double>> identityMatrix = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    for (size_t i = 0; i < dim; i++) {
        inputMatrix[i][i] = randomGen();
    }
    IsMatrixEqual(identityMatrix, decodeRegression_->Multiply(inputMatrix, decodeRegression_->Inverse3by3(inputMatrix)));
}

// 测试求行列式操作的有效性
TEST_F(DecodeRegressionTest, TestDeterminant3by3)
{
    size_t dim = 3;
    double expectedDet = 1.0;
    std::vector<std::vector<double>> inputMatrix(dim, std::vector<double>(dim));
    for (size_t i = 0; i < dim; i++) {
        inputMatrix[i][i] = randomGen();
        expectedDet *= inputMatrix[i][i];
    }
    EXPECT_NEAR(expectedDet, decodeRegression_->Determinant3by3(inputMatrix), 0.05);
}

// 添加一次数据点后，尝试回归更新一次参数
TEST_F(DecodeRegressionTest, TestTrain)
{
    size_t featureRows = 10, featureCols = 3;
    decodeRegression_->featureMatrix_ = std::vector<std::vector<double>>(featureRows, std::vector<double>(featureCols));
    decodeRegression_->attributeMatrix_ = std::vector<std::vector<double>>(featureRows, std::vector<double>(1));
    for (size_t i = 0; i < featureRows; i++) {
        for (size_t j = 0; j < featureCols; j++) {
            decodeRegression_->featureMatrix_[i][j] = static_cast<double>(i * featureCols + j);
        }
        decodeRegression_->attributeMatrix_[i][0] = static_cast<double>(i);
    }
    decodeRegression_->Train();
    IsMatrixEqual({{0.2181, 0, 29.961}}, decodeRegression_->coefficients_);
}

// 测试根据当前参数，根据tokenNum和kvBlock预测一次decode耗时
TEST_F(DecodeRegressionTest, TestPredict)
{
    int tokenNum = 1, kvBlock = 1;
    decodeRegression_->coefficients_ = {{1.0, 1.0, 1.0}};
    auto result = decodeRegression_->Predict(tokenNum, kvBlock);
    EXPECT_FLOAT_EQ(result, 3.0);
}

// 测试添加一个数据点
TEST_F(DecodeRegressionTest, TestAddDataPoint)
{
    decodeRegression_->AddDataPoint(1, 2, 3.0);
    decodeRegression_->AddDataPoint(4, 5, 6.0);

    IsMatrixEqual({{1.0, 2.0, 1.0}, {4.0, 5.0, 1.0}}, decodeRegression_->featureMatrix_);
    IsMatrixEqual({{3.0}, {6.0}}, decodeRegression_->attributeMatrix_);
}