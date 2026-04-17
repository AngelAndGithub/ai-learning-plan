#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第8天：线性代数复习与练习
线性代数学习示例
内容：线性代数综合练习和复习
"""

import numpy as np

print("=== 第8天：线性代数复习与练习 ===")

# 1. 向量操作练习
print("\n1. 向量操作练习")

# 创建向量
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

print(f"向量 v1: {v1}")
print(f"向量 v2: {v2}")

# 计算向量和
v_sum = v1 + v2
print(f"v1 + v2 = {v_sum}")

# 计算向量差
v_diff = v1 - v2
print(f"v1 - v2 = {v_diff}")

# 计算向量点积
v_dot = np.dot(v1, v2)
print(f"v1 · v2 = {v_dot}")

# 计算向量长度
v1_norm = np.linalg.norm(v1)
v2_norm = np.linalg.norm(v2)
print(f"||v1|| = {v1_norm}")
print(f"||v2|| = {v2_norm}")

# 计算单位向量
v1_unit = v1 / v1_norm
v2_unit = v2 / v2_norm
print(f"v1的单位向量: {v1_unit}")
print(f"v2的单位向量: {v2_unit}")

# 2. 矩阵操作练习
print("\n2. 矩阵操作练习")

# 创建矩阵
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(f"矩阵 A:\n{A}")
print(f"矩阵 B:\n{B}")

# 矩阵加法
A_plus_B = A + B
print(f"A + B:\n{A_plus_B}")

# 矩阵乘法
A_times_B = np.dot(A, B)
print(f"A · B:\n{A_times_B}")

# 矩阵转置
A_transpose = A.T
print(f"A的转置:\n{A_transpose}")

# 矩阵的逆
A_inv = np.linalg.inv(A)
print(f"A的逆:\n{A_inv}")

# 验证逆矩阵
A_times_Ainv = np.dot(A, A_inv)
print(f"A · A^{-1}:\n{A_times_Ainv}")

# 3. 行列式和特征值练习
print("\n3. 行列式和特征值练习")

# 计算行列式
det_A = np.linalg.det(A)
print(f"det(A) = {det_A}")

# 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"特征值: {eigenvalues}")
print(f"特征向量:\n{eigenvectors}")

# 4. 线性方程组求解练习
print("\n4. 线性方程组求解练习")

# 方程组: 2x + y = 5
#        x + 3y = 10
coeff_matrix = np.array([[2, 1], [1, 3]])
constants = np.array([5, 10])

print(f"系数矩阵:\n{coeff_matrix}")
print(f"常数项: {constants}")

# 方法1: 使用逆矩阵
solution1 = np.dot(np.linalg.inv(coeff_matrix), constants)
print(f"使用逆矩阵求解: x = {solution1[0]}, y = {solution1[1]}")

# 方法2: 使用numpy的solve函数
solution2 = np.linalg.solve(coeff_matrix, constants)
print(f"使用np.linalg.solve求解: x = {solution2[0]}, y = {solution2[1]}")

# 验证解
left_side1 = 2*solution1[0] + solution1[1]
left_side2 = solution1[0] + 3*solution1[1]
print(f"验证: 2x + y = {left_side1} (应该等于 5)")
print(f"验证: x + 3y = {left_side2} (应该等于 10)")

# 5. SVD练习
print("\n5. SVD练习")

# 创建一个矩阵
C = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"矩阵 C:\n{C}")

# 执行SVD
U, S, VT = np.linalg.svd(C)
print(f"U:\n{U}")
print(f"S: {S}")
print(f"VT:\n{VT}")

# 重构矩阵
Sigma = np.zeros(C.shape)
Sigma[:min(C.shape), :min(C.shape)] = np.diag(S)
C_reconstructed = np.dot(np.dot(U, Sigma), VT)
print(f"\n重构矩阵:\n{C_reconstructed}")
print(f"与原矩阵相等: {np.allclose(C, C_reconstructed)}")

# 6. 应用练习：PCA
print("\n6. 应用练习：PCA")

# 创建示例数据
np.random.seed(42)
data = np.random.randn(100, 3)  # 100个样本，3个特征
print(f"原始数据形状: {data.shape}")

# 数据中心化
data_centered = data - np.mean(data, axis=0)

# 计算协方差矩阵
cov_matrix = np.cov(data_centered.T)
print(f"协方差矩阵形状: {cov_matrix.shape}")

# 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print(f"特征值: {eigenvalues}")

# 排序并选择前2个主成分
sorted_indices = np.argsort(eigenvalues)[::-1]
top_eigenvectors = eigenvectors[:, sorted_indices[:2]]

# 降维
data_reduced = np.dot(data_centered, top_eigenvectors)
print(f"降维后数据形状: {data_reduced.shape}")
print(f"前5个降维后的数据:\n{data_reduced[:5]}")

# 7. 综合练习：神经网络前向传播
print("\n7. 综合练习：神经网络前向传播")

# 定义网络参数
input_size = 3
hidden_size = 4
output_size = 2

# 初始化权重和偏置
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size)
b1 = np.random.randn(hidden_size)
W2 = np.random.randn(hidden_size, output_size)
b2 = np.random.randn(output_size)

print(f"输入层到隐藏层权重形状: {W1.shape}")
print(f"隐藏层偏置形状: {b1.shape}")
print(f"隐藏层到输出层权重形状: {W2.shape}")
print(f"输出层偏置形状: {b2.shape}")

# 输入数据
X = np.array([[0.1, 0.2, 0.3]])
print(f"输入数据: {X}")

# 前向传播
hidden = np.dot(X, W1) + b1
hidden_activated = np.maximum(0, hidden)  # ReLU激活
output = np.dot(hidden_activated, W2) + b2

print(f"隐藏层输出: {hidden}")
print(f"隐藏层激活后: {hidden_activated}")
print(f"输出层: {output}")

print("\n=== 第8天学习示例结束 ===")
