#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第3天：行列式
线性代数学习示例
内容：行列式的计算和性质
"""

import numpy as np

print("=== 第3天：行列式 ===")

# 1. 2x2 矩阵的行列式
print("\n1. 2x2 矩阵的行列式")

A = np.array([[1, 2], [3, 4]])
det_A = np.linalg.det(A)
print(f"矩阵 A:\n{A}")
print(f"行列式 det(A) = {det_A}")

# 手动计算 2x2 行列式
manual_det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
print(f"手动计算行列式: {manual_det}")

# 2. 3x3 矩阵的行列式
print("\n2. 3x3 矩阵的行列式")

B = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
det_B = np.linalg.det(B)
print(f"矩阵 B:\n{B}")
print(f"行列式 det(B) = {det_B}")

# 3. 行列式的性质
print("\n3. 行列式的性质")

# 性质1：单位矩阵的行列式为1
identity = np.eye(3)
det_identity = np.linalg.det(identity)
print(f"单位矩阵的行列式: {det_identity}")

# 性质2：交换矩阵的两行，行列式变号
C = np.array([[3, 4], [1, 2]])  # 交换A的两行
det_C = np.linalg.det(C)
print(f"\n交换A的两行得到矩阵 C:\n{C}")
print(f"行列式 det(C) = {det_C}")
print(f"det(C) = -det(A) -> {np.isclose(det_C, -det_A)}")

# 性质3：矩阵的某一行乘以标量k，行列式变为k倍
D = np.array([[2*1, 2*2], [3, 4]])  # 第一行乘以2
det_D = np.linalg.det(D)
print(f"\n第一行乘以2得到矩阵 D:\n{D}")
print(f"行列式 det(D) = {det_D}")
print(f"det(D) = 2 * det(A) -> {np.isclose(det_D, 2 * det_A)}")

# 性质4：矩阵的某一行加上另一行的倍数，行列式不变
E = np.array([[1 + 2*3, 2 + 2*4], [3, 4]])  # 第一行加上第二行的2倍
det_E = np.linalg.det(E)
print(f"\n第一行加上第二行的2倍得到矩阵 E:\n{E}")
print(f"行列式 det(E) = {det_E}")
print(f"det(E) = det(A) -> {np.isclose(det_E, det_A)}")

# 4. 行列式与矩阵可逆性的关系
print("\n4. 行列式与矩阵可逆性的关系")

# 可逆矩阵（非奇异矩阵）的行列式不为零
print(f"矩阵 A 的行列式: {det_A}")
print(f"矩阵 A 是否可逆: {det_A != 0}")

# 不可逆矩阵（奇异矩阵）的行列式为零
print(f"矩阵 B 的行列式: {det_B}")
print(f"矩阵 B 是否可逆: {det_B != 0}")

# 5. 计算矩阵的逆（如果可逆）
print("\n5. 计算矩阵的逆（如果可逆）")

if det_A != 0:
    A_inv = np.linalg.inv(A)
    print(f"矩阵 A 的逆:\n{A_inv}")
    # 验证 A * A_inv = I
    identity_check = np.dot(A, A_inv)
    print(f"A * A_inv:\n{identity_check}")

# 6. 行列式的几何意义
print("\n6. 行列式的几何意义")

# 2x2 矩阵的行列式表示由列向量张成的平行四边形的面积
v1 = np.array([1, 0])
v2 = np.array([0, 1])
area_matrix = np.array([v1, v2]).T
det_area = np.linalg.det(area_matrix)
print(f"单位正方形的面积: {det_area}")

# 缩放后的面积
scaled_matrix = np.array([[2, 0], [0, 3]])
det_scaled = np.linalg.det(scaled_matrix)
print(f"缩放矩阵的行列式（面积缩放因子）: {det_scaled}")

# 7. 应用示例：解线性方程组
print("\n7. 应用示例：解线性方程组")

# 方程组: 2x + 3y = 8
#        4x - 5y = -6
coeff_matrix = np.array([[2, 3], [4, -5]])
constants = np.array([8, -6])

# 计算系数矩阵的行列式
det_coeff = np.linalg.det(coeff_matrix)
print(f"系数矩阵:\n{coeff_matrix}")
print(f"系数矩阵的行列式: {det_coeff}")

# 使用行列式解方程组（克拉默法则）
if det_coeff != 0:
    # x 的分子矩阵
    x_matrix = np.array([[8, 3], [-6, -5]])
    det_x = np.linalg.det(x_matrix)
    x = det_x / det_coeff
    
    # y 的分子矩阵
    y_matrix = np.array([[2, 8], [4, -6]])
    det_y = np.linalg.det(y_matrix)
    y = det_y / det_coeff
    
    print(f"解: x = {x}, y = {y}")
    
    # 验证解
    left_side1 = 2*x + 3*y
    left_side2 = 4*x - 5*y
    print(f"验证: 2x + 3y = {left_side1} (应该等于 8)")
    print(f"验证: 4x - 5y = {left_side2} (应该等于 -6)")

print("\n=== 第3天学习示例结束 ===")
