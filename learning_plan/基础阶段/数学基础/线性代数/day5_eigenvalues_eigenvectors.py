#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第5天：特征值与特征向量
线性代数学习示例
内容：特征值和特征向量的计算、性质和应用
"""

import numpy as np

print("=== 第5天：特征值与特征向量 ===")

# 1. 特征值和特征向量的计算
print("\n1. 特征值和特征向量的计算")

# 创建一个矩阵
A = np.array([[2, 1], [1, 2]])
print(f"矩阵 A:\n{A}")

# 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"特征值: {eigenvalues}")
print(f"特征向量:\n{eigenvectors}")

# 验证 Av = λv
print("\n验证 Av = λv:")
for i in range(len(eigenvalues)):
    lambda_i = eigenvalues[i]
    v_i = eigenvectors[:, i]
    Av = np.dot(A, v_i)
    lambda_v = lambda_i * v_i
    print(f"特征值 {lambda_i}:")
    print(f"Av = {Av}")
    print(f"λv = {lambda_v}")
    print(f"相等: {np.allclose(Av, lambda_v)}")
    print()

# 2. 特征值和特征向量的性质
print("\n2. 特征值和特征向量的性质")

# 性质1：特征值的和等于矩阵的迹（对角线元素之和）
trace_A = np.trace(A)
sum_eigenvalues = np.sum(eigenvalues)
print(f"矩阵的迹: {trace_A}")
print(f"特征值的和: {sum_eigenvalues}")
print(f"相等: {np.isclose(trace_A, sum_eigenvalues)}")

# 性质2：特征值的乘积等于矩阵的行列式
det_A = np.linalg.det(A)
prod_eigenvalues = np.prod(eigenvalues)
print(f"\n矩阵的行列式: {det_A}")
print(f"特征值的乘积: {prod_eigenvalues}")
print(f"相等: {np.isclose(det_A, prod_eigenvalues)}")

# 3. 对称矩阵的特征向量
print("\n3. 对称矩阵的特征向量")

# 对称矩阵的特征向量正交
print("对称矩阵的特征向量正交性:")
v1 = eigenvectors[:, 0]
v2 = eigenvectors[:, 1]
dot_product = np.dot(v1, v2)
print(f"特征向量 v1: {v1}")
print(f"特征向量 v2: {v2}")
print(f"点积: {dot_product}")
print(f"正交: {np.isclose(dot_product, 0)}")

# 4. 对角化
print("\n4. 对角化")

# 对角化：A = PDP^-1
P = eigenvectors
D = np.diag(eigenvalues)
P_inv = np.linalg.inv(P)

# 验证 A = PDP^-1
A_reconstructed = np.dot(np.dot(P, D), P_inv)
print(f"原始矩阵 A:\n{A}")
print(f"重构矩阵 PDP^-1:\n{A_reconstructed}")
print(f"相等: {np.allclose(A, A_reconstructed)}")

# 5. 应用示例：矩阵的幂
print("\n5. 应用示例：矩阵的幂")

# 计算 A^3
A_pow3 = np.linalg.matrix_power(A, 3)
print(f"A^3:\n{A_pow3}")

# 使用特征值计算 A^3: P D^3 P^-1
D_pow3 = np.diag(eigenvalues**3)
A_pow3_eigen = np.dot(np.dot(P, D_pow3), P_inv)
print(f"使用特征值计算 A^3:\n{A_pow3_eigen}")
print(f"相等: {np.allclose(A_pow3, A_pow3_eigen)}")

# 6. 应用示例：系统动力学
print("\n6. 应用示例：系统动力学")

# 考虑一个线性系统 x(t+1) = A x(t)
x0 = np.array([1, 0])  # 初始状态

# 计算几个时间步的状态
print("系统状态演化:")
print(f"t=0: {x0}")

x1 = np.dot(A, x0)
print(f"t=1: {x1}")

x2 = np.dot(A, x1)
print(f"t=2: {x2}")

x3 = np.dot(A, x2)
print(f"t=3: {x3}")

# 使用特征分解分析系统行为
print("\n使用特征分解分析:")
print(f"主特征值: {max(eigenvalues)}")
print(f"系统将沿主特征向量方向增长")

# 7. 复数特征值
print("\n7. 复数特征值")

# 创建一个可能产生复数特征值的矩阵
B = np.array([[0, -1], [1, 0]])  # 旋转矩阵
print(f"矩阵 B:\n{B}")

eigenvalues_B, eigenvectors_B = np.linalg.eig(B)
print(f"特征值: {eigenvalues_B}")
print(f"特征向量:\n{eigenvectors_B}")

print("\n=== 第5天学习示例结束 ===")
