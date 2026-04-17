#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第6天：奇异值分解
线性代数学习示例
内容：SVD的计算、性质和应用
"""

import numpy as np

print("=== 第6天：奇异值分解 ===")

# 1. 奇异值分解的计算
print("\n1. 奇异值分解的计算")

# 创建一个矩阵
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"矩阵 A:\n{A}")

# 执行SVD
U, S, VT = np.linalg.svd(A)
print(f"\nU 矩阵 (左奇异向量):\n{U}")
print(f"\nS 向量 (奇异值):\n{S}")
print(f"\nVT 矩阵 (右奇异向量的转置):\n{VT}")

# 构建对角矩阵 Sigma
Sigma = np.zeros(A.shape)
Sigma[:min(A.shape), :min(A.shape)] = np.diag(S)
print(f"\nSigma 矩阵:\n{Sigma}")

# 验证 A = U * Sigma * VT
A_reconstructed = np.dot(np.dot(U, Sigma), VT)
print(f"\n重构矩阵 U*Sigma*VT:\n{A_reconstructed}")
print(f"与原矩阵相等: {np.allclose(A, A_reconstructed)}")

# 2. SVD的性质
print("\n2. SVD的性质")

# U和VT都是正交矩阵
print("U是正交矩阵 (U^T U = I):")
print(np.dot(U.T, U))
print(f"正交性验证: {np.allclose(np.dot(U.T, U), np.eye(U.shape[0]))}")

print("\nVT是正交矩阵 (VT VT^T = I):")
print(np.dot(VT, VT.T))
print(f"正交性验证: {np.allclose(np.dot(VT, VT.T), np.eye(VT.shape[0]))}")

# 奇异值按降序排列
print(f"\n奇异值按降序排列: {np.array_equal(S, np.sort(S)[::-1])}")

# 3. 奇异值的意义
print("\n3. 奇异值的意义")

# 奇异值的平方等于 A^T A 的特征值
eigenvalues_ATA = np.linalg.eigvals(np.dot(A.T, A))
singular_values_squared = S**2
print(f"A^T A 的特征值: {eigenvalues_ATA}")
print(f"奇异值的平方: {singular_values_squared}")
print(f"相等: {np.allclose(np.sort(eigenvalues_ATA), np.sort(singular_values_squared))}")

# 4. 应用示例：矩阵近似
print("\n4. 应用示例：矩阵近似")

# 使用前k个奇异值进行矩阵近似
k = 2  # 保留前2个奇异值
U_k = U[:, :k]
S_k = S[:k]
VT_k = VT[:k, :]
Sigma_k = np.diag(S_k)

# 计算近似矩阵
A_approx = np.dot(np.dot(U_k, Sigma_k), VT_k)
print(f"\n原始矩阵:\n{A}")
print(f"\n使用前 {k} 个奇异值的近似矩阵:\n{A_approx}")

# 计算近似误差
error = np.linalg.norm(A - A_approx)
print(f"\n近似误差: {error}")

# 5. 应用示例：数据压缩
print("\n5. 应用示例：数据压缩")

# 计算原始矩阵的存储量
original_size = A.size
print(f"原始矩阵大小: {original_size} 元素")

# 计算压缩后的存储量
compressed_size = U_k.size + S_k.size + VT_k.size
print(f"压缩后大小: {compressed_size} 元素")
print(f"压缩率: {compressed_size / original_size:.2f}")

# 6. 应用示例：降维
print("\n6. 应用示例：降维")

# 创建一些示例数据
np.random.seed(42)
data = np.random.randn(10, 3)  # 10个样本，3个特征
print(f"原始数据形状: {data.shape}")

# 对数据进行SVD降维
U_data, S_data, VT_data = np.linalg.svd(data)

# 降维到2维
k_dim = 2
U_data_k = U_data[:, :k_dim]
S_data_k = S_data[:k_dim]
VT_data_k = VT_data[:k_dim, :]

# 计算降维后的数据
data_reduced = np.dot(data, VT_data_k.T)
print(f"降维后数据形状: {data_reduced.shape}")
print(f"降维后数据:\n{data_reduced}")

# 7. 应用示例：伪逆
print("\n7. 应用示例：伪逆")

# 计算矩阵的伪逆
A_pseudo_inverse = np.linalg.pinv(A)
print(f"矩阵的伪逆:\n{A_pseudo_inverse}")

# 使用SVD计算伪逆
Sigma_pseudo = np.zeros(A.shape).T
Sigma_pseudo[:min(A.shape), :min(A.shape)] = np.diag(1/S)
A_pseudo_inverse_svd = np.dot(np.dot(VT.T, Sigma_pseudo), U.T)
print(f"使用SVD计算的伪逆:\n{A_pseudo_inverse_svd}")
print(f"与numpy计算的伪逆相等: {np.allclose(A_pseudo_inverse, A_pseudo_inverse_svd)}")

print("\n=== 第6天学习示例结束 ===")
