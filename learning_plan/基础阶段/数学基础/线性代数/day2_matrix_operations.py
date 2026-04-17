#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第2天：矩阵运算
线性代数学习示例
内容：矩阵加法、矩阵乘法、矩阵转置等基本运算
"""

import numpy as np

print("=== 第2天：矩阵运算 ===")

# 1. 矩阵加法
print("\n1. 矩阵加法")

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 矩阵加法
C = A + B
print(f"矩阵 A:\n{A}")
print(f"矩阵 B:\n{B}")
print(f"矩阵加法 A + B:\n{C}")

# 2. 矩阵减法
print("\n2. 矩阵减法")

D = A - B
print(f"矩阵减法 A - B:\n{D}")

# 3. 矩阵标量乘法
print("\n3. 矩阵标量乘法")

E = 2 * A
print(f"标量乘法 2 * A:\n{E}")

# 4. 矩阵乘法（矩阵积）
print("\n4. 矩阵乘法（矩阵积）")

# 2x2 矩阵相乘
F = np.dot(A, B)
print(f"矩阵乘法 A · B:\n{F}")

# 2x3 和 3x2 矩阵相乘
G = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3 矩阵
H = np.array([[7, 8], [9, 10], [11, 12]])  # 3x2 矩阵
I = np.dot(G, H)
print(f"\n矩阵 G (2x3):\n{G}")
print(f"矩阵 H (3x2):\n{H}")
print(f"矩阵乘法 G · H (2x2):\n{I}")

# 5. 矩阵转置
print("\n5. 矩阵转置")

A_transpose = A.T
print(f"矩阵 A:\n{A}")
print(f"矩阵 A 的转置:\n{A_transpose}")

G_transpose = G.T
print(f"\n矩阵 G (2x3):\n{G}")
print(f"矩阵 G 的转置 (3x2):\n{G_transpose}")

# 6. 矩阵乘法的性质
print("\n6. 矩阵乘法的性质")

# 结合律：(AB)C = A(BC)
J = np.array([[1, 0], [0, 1]])  # 单位矩阵
AB = np.dot(A, B)
ABC = np.dot(AB, J)
BC = np.dot(B, J)
ABC_alt = np.dot(A, BC)
print(f"结合律验证: (AB)C = A(BC) -> {np.array_equal(ABC, ABC_alt)}")

# 分配律：A(B + C) = AB + AC
K = np.array([[1, 1], [1, 1]])
A_BC = np.dot(A, B + K)
AB_AK = np.dot(A, B) + np.dot(A, K)
print(f"分配律验证: A(B + C) = AB + AC -> {np.array_equal(A_BC, AB_AK)}")

# 矩阵乘法不满足交换律：AB != BA
AB = np.dot(A, B)
BA = np.dot(B, A)
print(f"交换律验证: AB = BA -> {np.array_equal(AB, BA)}")
print(f"AB:\n{AB}")
print(f"BA:\n{BA}")

# 7. 矩阵与向量的乘法
print("\n7. 矩阵与向量的乘法")

vector = np.array([1, 2])
result = np.dot(A, vector)
print(f"矩阵 A:\n{A}")
print(f"向量 v: {vector}")
print(f"矩阵与向量乘法 A · v: {result}")

# 8. 应用示例：线性变换
print("\n8. 应用示例：线性变换")

# 旋转矩阵 (90度)
rotation_matrix = np.array([[0, -1], [1, 0]])
point = np.array([1, 0])  # 点 (1, 0)
rotated_point = np.dot(rotation_matrix, point)
print(f"原始点: {point}")
print(f"旋转矩阵 (90度):\n{rotation_matrix}")
print(f"旋转后点: {rotated_point}")

# 缩放矩阵
scaling_matrix = np.array([[2, 0], [0, 3]])
scaled_point = np.dot(scaling_matrix, point)
print(f"\n缩放矩阵:\n{scaling_matrix}")
print(f"缩放后点: {scaled_point}")

print("\n=== 第2天学习示例结束 ===")
