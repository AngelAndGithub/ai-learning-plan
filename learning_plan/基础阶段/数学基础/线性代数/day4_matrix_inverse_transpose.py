#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第4天：矩阵的逆与转置
线性代数学习示例
内容：矩阵的逆、转置的性质和计算
"""

import numpy as np

print("=== 第4天：矩阵的逆与转置 ===")

# 1. 矩阵的逆
print("\n1. 矩阵的逆")

# 创建一个可逆矩阵
A = np.array([[1, 2], [3, 4]])
print(f"矩阵 A:\n{A}")

# 计算矩阵的逆
A_inv = np.linalg.inv(A)
print(f"矩阵 A 的逆:\n{A_inv}")

# 验证 A * A_inv = I
identity = np.dot(A, A_inv)
print(f"A * A_inv:\n{identity}")

# 验证 A_inv * A = I
identity2 = np.dot(A_inv, A)
print(f"A_inv * A:\n{identity2}")

# 2. 矩阵的转置
print("\n2. 矩阵的转置")

# 计算矩阵的转置
A_transpose = A.T
print(f"矩阵 A:\n{A}")
print(f"矩阵 A 的转置:\n{A_transpose}")

# 转置的性质：(A^T)^T = A
A_transpose_transpose = A_transpose.T
print(f"(A^T)^T = A -> {np.array_equal(A_transpose_transpose, A)}")

# 3. 转置的性质
print("\n3. 转置的性质")

B = np.array([[5, 6], [7, 8]])

# 性质1：(A + B)^T = A^T + B^T
A_plus_B_transpose = (A + B).T
A_transpose_plus_B_transpose = A.T + B.T
print(f"(A + B)^T = A^T + B^T -> {np.array_equal(A_plus_B_transpose, A_transpose_plus_B_transpose)}")

# 性质2：(kA)^T = kA^T
k = 2
kA_transpose = (k * A).T
kA_transpose_alt = k * A.T
print(f"(kA)^T = kA^T -> {np.array_equal(kA_transpose, kA_transpose_alt)}")

# 性质3：(AB)^T = B^T A^T
AB_transpose = (np.dot(A, B)).T
B_transpose_A_transpose = np.dot(B.T, A.T)
print(f"(AB)^T = B^T A^T -> {np.array_equal(AB_transpose, B_transpose_A_transpose)}")

# 4. 逆的性质
print("\n4. 逆的性质")

# 性质1：(A^-1)^-1 = A
A_inv_inv = np.linalg.inv(A_inv)
print(f"(A^-1)^-1 = A -> {np.array_equal(A_inv_inv, A)}")

# 性质2：(kA)^-1 = (1/k)A^-1  (k != 0)
k = 2
kA_inv = np.linalg.inv(k * A)
one_over_k_A_inv = (1/k) * A_inv
print(f"(kA)^-1 = (1/k)A^-1 -> {np.allclose(kA_inv, one_over_k_A_inv)}")

# 性质3：(AB)^-1 = B^-1 A^-1
B_inv = np.linalg.inv(B)
AB_inv = np.linalg.inv(np.dot(A, B))
B_inv_A_inv = np.dot(B_inv, A_inv)
print(f"(AB)^-1 = B^-1 A^-1 -> {np.allclose(AB_inv, B_inv_A_inv)}")

# 性质4：(A^T)^-1 = (A^-1)^T
A_transpose_inv = np.linalg.inv(A.T)
A_inv_transpose = A_inv.T
print(f"(A^T)^-1 = (A^-1)^T -> {np.allclose(A_transpose_inv, A_inv_transpose)}")

# 5. 特殊矩阵的逆
print("\n5. 特殊矩阵的逆")

# 对角矩阵的逆
D = np.diag([2, 3, 4])
D_inv = np.linalg.inv(D)
print(f"对角矩阵 D:\n{D}")
print(f"对角矩阵的逆 D^-1:\n{D_inv}")
print(f"对角矩阵的逆是对角线元素的倒数: {np.array_equal(D_inv, np.diag([1/2, 1/3, 1/4]))}")

# 单位矩阵的逆
I = np.eye(3)
I_inv = np.linalg.inv(I)
print(f"\n单位矩阵 I:\n{I}")
print(f"单位矩阵的逆 I^-1:\n{I_inv}")

# 6. 应用示例：解线性方程组
print("\n6. 应用示例：解线性方程组")

# 方程组: 2x + y = 5
#        x + 3y = 10
coeff_matrix = np.array([[2, 1], [1, 3]])
constants = np.array([5, 10])

# 使用矩阵的逆解方程组: x = A^-1 b
coeff_inv = np.linalg.inv(coeff_matrix)
solution = np.dot(coeff_inv, constants)
print(f"系数矩阵:\n{coeff_matrix}")
print(f"常数项: {constants}")
print(f"解: x = {solution[0]}, y = {solution[1]}")

# 验证解
left_side1 = 2*solution[0] + solution[1]
left_side2 = solution[0] + 3*solution[1]
print(f"验证: 2x + y = {left_side1} (应该等于 5)")
print(f"验证: x + 3y = {left_side2} (应该等于 10)")

# 7. 矩阵的伪逆
print("\n7. 矩阵的伪逆")

# 非方阵的伪逆
non_square = np.array([[1, 2], [3, 4], [5, 6]])
pseudo_inverse = np.linalg.pinv(non_square)
print(f"非方阵:\n{non_square}")
print(f"伪逆:\n{pseudo_inverse}")

# 验证伪逆的性质
print(f"A * A+ * A:\n{np.dot(np.dot(non_square, pseudo_inverse), non_square)}")
print(f"A+ * A * A+:\n{np.dot(np.dot(pseudo_inverse, non_square), pseudo_inverse)}")

print("\n=== 第4天学习示例结束 ===")
