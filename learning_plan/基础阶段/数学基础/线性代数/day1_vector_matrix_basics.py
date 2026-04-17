#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第1天：向量与矩阵基础
线性代数学习示例
内容：向量与矩阵的基本概念、表示方法和基本运算
"""

import numpy as np

print("=== 第1天：向量与矩阵基础 ===")

# 1. 向量的基本概念和表示
print("\n1. 向量的基本概念和表示")

# 列向量
column_vector = np.array([[1], [2], [3]])
print(f"列向量:\n{column_vector}")

# 行向量
row_vector = np.array([1, 2, 3])
print(f"\n行向量:\n{row_vector}")

# 向量的维度
print(f"\n向量维度: {row_vector.shape}")

# 2. 向量的基本运算
print("\n2. 向量的基本运算")

v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# 向量加法
v_add = v1 + v2
print(f"向量加法 v1 + v2 = {v_add}")

# 向量减法
v_sub = v1 - v2
print(f"向量减法 v1 - v2 = {v_sub}")

# 向量数乘
v_scalar = 2 * v1
print(f"向量数乘 2 * v1 = {v_scalar}")

# 向量点积
v_dot = np.dot(v1, v2)
print(f"向量点积 v1 · v2 = {v_dot}")

# 向量长度（模）
v_norm = np.linalg.norm(v1)
print(f"向量长度 ||v1|| = {v_norm}")

# 3. 矩阵的基本概念和表示
print("\n3. 矩阵的基本概念和表示")

# 2x3矩阵
matrix_2x3 = np.array([[1, 2, 3], [4, 5, 6]])
print(f"2x3矩阵:\n{matrix_2x3}")

# 3x3矩阵
matrix_3x3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"\n3x3矩阵:\n{matrix_3x3}")

# 矩阵的维度
print(f"\n矩阵维度: {matrix_3x3.shape}")

# 4. 特殊矩阵
print("\n4. 特殊矩阵")

# 单位矩阵
identity_matrix = np.eye(3)
print(f"3x3单位矩阵:\n{identity_matrix}")

# 零矩阵
zero_matrix = np.zeros((2, 3))
print(f"\n2x3零矩阵:\n{zero_matrix}")

# 对角矩阵
diagonal_matrix = np.diag([1, 2, 3])
print(f"\n3x3对角矩阵:\n{diagonal_matrix}")

# 5. 矩阵的元素访问
print("\n5. 矩阵的元素访问")

print(f"矩阵元素 matrix_3x3[0, 0] = {matrix_3x3[0, 0]}")
print(f"矩阵第一行: {matrix_3x3[0, :]}")
print(f"矩阵第二列: {matrix_3x3[:, 1]}")

# 6. 向量和矩阵的应用示例
print("\n6. 应用示例")

# 线性组合
print("线性组合示例:")
a = 2
b = 3
v_combination = a * v1 + b * v2
print(f"{a}*v1 + {b}*v2 = {v_combination}")

# 矩阵表示线性方程组
print("\n线性方程组表示:")
# 方程组: 2x + 3y = 8
#        4x - 5y = -6
A = np.array([[2, 3], [4, -5]])
b = np.array([8, -6])
print(f"系数矩阵 A:\n{A}")
print(f"常数项 b: {b}")

print("\n=== 第1天学习示例结束 ===")
