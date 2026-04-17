#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第7天：线性代数在机器学习中的应用
线性代数学习示例
内容：线性代数在机器学习中的各种应用
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

print("=== 第7天：线性代数在机器学习中的应用 ===")

# 1. 线性回归
print("\n1. 线性回归")

# 创建示例数据
X = np.array([[1], [2], [3], [4], [5]])  # 特征
Y = np.array([2, 4, 6, 8, 10])  # 目标

print(f"特征 X:\n{X}")
print(f"目标 Y: {Y}")

# 使用numpy的线性代数求解
# 线性回归模型: Y = X * w + b
# 添加偏置项
X_with_bias = np.hstack([np.ones((X.shape[0], 1)), X])
print(f"\n添加偏置项后的 X:\n{X_with_bias}")

# 使用最小二乘法求解: w = (X^T X)^-1 X^T Y
X_T = X_with_bias.T
X_T_X = np.dot(X_T, X_with_bias)
X_T_Y = np.dot(X_T, Y)
weights = np.dot(np.linalg.inv(X_T_X), X_T_Y)

print(f"\n权重: {weights}")
print(f"偏置: {weights[0]}")
print(f"斜率: {weights[1]}")

# 预测
X_test = np.array([[6], [7]])
X_test_with_bias = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
y_pred = np.dot(X_test_with_bias, weights)
print(f"\n测试数据: {X_test.flatten()}")
print(f"预测结果: {y_pred}")

# 使用scikit-learn验证
model = LinearRegression()
model.fit(X, Y)
print(f"\nscikit-learn 模型系数: {model.coef_}")
print(f"scikit-learn 模型截距: {model.intercept_}")
print(f"scikit-learn 预测结果: {model.predict(X_test)}")

# 2. 主成分分析 (PCA)
print("\n2. 主成分分析 (PCA)")

# 加载鸢尾花数据集
iris = load_iris()
X_iris = iris.data
print(f"原始数据形状: {X_iris.shape}")
print(f"前5个样本:\n{X_iris[:5]}")

# 数据中心化
X_centered = X_iris - np.mean(X_iris, axis=0)
print(f"\n中心化后的数据:\n{X_centered[:5]}")

# 计算协方差矩阵
cov_matrix = np.cov(X_centered.T)
print(f"\n协方差矩阵形状: {cov_matrix.shape}")
print(f"协方差矩阵:\n{cov_matrix}")

# 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print(f"\n特征值: {eigenvalues}")
print(f"特征向量:\n{eigenvectors}")

# 排序特征值和特征向量
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]
print(f"\n排序后的特征值: {sorted_eigenvalues}")
print(f"排序后的特征向量:\n{sorted_eigenvectors}")

# 选择前2个主成分
k = 2
principal_components = sorted_eigenvectors[:, :k]
print(f"\n前 {k} 个主成分:\n{principal_components}")

# 降维
X_reduced = np.dot(X_centered, principal_components)
print(f"\n降维后的数据形状: {X_reduced.shape}")
print(f"降维后的数据:\n{X_reduced[:5]}")

# 使用scikit-learn验证
pca = PCA(n_components=2)
X_reduced_sklearn = pca.fit_transform(X_iris)
print(f"\nscikit-learn 降维后的数据:\n{X_reduced_sklearn[:5]}")
print(f"scikit-learn 解释方差比: {pca.explained_variance_ratio_}")

# 3. 神经网络中的矩阵运算
print("\n3. 神经网络中的矩阵运算")

# 模拟一个简单的神经网络
# 输入层: 3个神经元
# 隐藏层: 2个神经元
# 输出层: 1个神经元

# 权重矩阵
W1 = np.random.randn(3, 2)  # 输入到隐藏层
b1 = np.random.randn(2)     # 隐藏层偏置
W2 = np.random.randn(2, 1)  # 隐藏层到输出层
b2 = np.random.randn(1)     # 输出层偏置

print(f"输入到隐藏层权重 W1:\n{W1}")
print(f"隐藏层偏置 b1: {b1}")
print(f"隐藏层到输出层权重 W2:\n{W2}")
print(f"输出层偏置 b2: {b2}")

# 输入数据
X_nn = np.array([[0.1, 0.2, 0.3]])
print(f"\n输入数据: {X_nn}")

# 前向传播
hidden_layer = np.dot(X_nn, W1) + b1
hidden_layer_activation = np.maximum(0, hidden_layer)  # ReLU激活
output_layer = np.dot(hidden_layer_activation, W2) + b2

print(f"\n隐藏层输出: {hidden_layer}")
print(f"隐藏层激活后: {hidden_layer_activation}")
print(f"输出层: {output_layer}")

# 4. 奇异值分解在推荐系统中的应用
print("\n4. 奇异值分解在推荐系统中的应用")

# 创建用户-物品评分矩阵
# 行: 用户
# 列: 物品
# 值: 评分 (0表示未评分)
ratings = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4]
])

print(f"用户-物品评分矩阵:\n{ratings}")

# 执行SVD
U, S, VT = np.linalg.svd(ratings, full_matrices=False)
print(f"\nU 矩阵形状: {U.shape}")
print(f"S 向量形状: {S.shape}")
print(f"VT 矩阵形状: {VT.shape}")

# 选择前k个奇异值
k = 2
U_k = U[:, :k]
S_k = np.diag(S[:k])
VT_k = VT[:k, :]

# 重构评分矩阵
ratings_reconstructed = np.dot(np.dot(U_k, S_k), VT_k)
print(f"\n重构的评分矩阵:\n{ratings_reconstructed}")

# 预测用户对物品的评分
user_idx = 0
item_idx = 2
predicted_rating = ratings_reconstructed[user_idx, item_idx]
print(f"\n预测用户 {user_idx} 对物品 {item_idx} 的评分: {predicted_rating:.2f}")

print("\n=== 第7天学习示例结束 ===")
