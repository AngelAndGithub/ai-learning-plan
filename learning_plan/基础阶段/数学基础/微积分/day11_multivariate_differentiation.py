#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第11天：多元函数求导
微积分学习示例
内容：偏导数、全微分和高阶偏导数
"""

import numpy as np

print("=== 第11天：多元函数求导 ===")

# 1. 偏导数
print("\n1. 偏导数")

# 定义一个二元函数
def f(x, y):
    return x**2 + 2*y**2 + 3*x*y

# 计算偏导数
# ∂f/∂x = 2x + 3y
def df_dx(x, y):
    return 2*x + 3*y

# ∂f/∂y = 4y + 3x
def df_dy(x, y):
    return 4*y + 3*x

# 计算在点(1, 2)处的偏导数
x0, y0 = 1, 2
print(f"函数 f(x, y) = x² + 2y² + 3xy")
print(f"在点 ({x0}, {y0}) 处的偏导数:")
print(f"∂f/∂x = {df_dx(x0, y0)}")
print(f"∂f/∂y = {df_dy(x0, y0)}")

# 数值偏导数验证
def numerical_partial_derivative(f, x, y, variable='x', h=1e-6):
    """计算数值偏导数"""
    if variable == 'x':
        return (f(x + h, y) - f(x - h, y)) / (2 * h)
    elif variable == 'y':
        return (f(x, y + h) - f(x, y - h)) / (2 * h)

print(f"\n数值偏导数验证:")
print(f"∂f/∂x (数值) = {numerical_partial_derivative(f, x0, y0, 'x')}")
print(f"∂f/∂y (数值) = {numerical_partial_derivative(f, x0, y0, 'y')}")

# 2. 全微分
print("\n2. 全微分")

# 全微分 df = (∂f/∂x)dx + (∂f/∂y)dy
def total_differential(f, x, y, dx, dy):
    return df_dx(x, y) * dx + df_dy(x, y) * dy

# 计算在点(1, 2)处，dx=0.1, dy=0.2时的全微分
dx, dy = 0.1, 0.2
df = total_differential(f, x0, y0, dx, dy)
print(f"在点 ({x0}, {y0}) 处，dx={dx}, dy={dy} 时的全微分:")
print(f"df = {df}")

# 实际函数值变化
Δf = f(x0 + dx, y0 + dy) - f(x0, y0)
print(f"实际函数值变化 Δf = {Δf}")
print(f"全微分近似误差: {abs(df - Δf)}")

# 3. 高阶偏导数
print("\n3. 高阶偏导数")

# 二阶偏导数
# ∂²f/∂x² = 2
def d2f_dx2(x, y):
    return 2

# ∂²f/∂y² = 4
def d2f_dy2(x, y):
    return 4

# ∂²f/∂x∂y = 3
def d2f_dxdy(x, y):
    return 3

# ∂²f/∂y∂x = 3
def d2f_dydx(x, y):
    return 3

print(f"二阶偏导数:")
print(f"∂²f/∂x² = {d2f_dx2(x0, y0)}")
print(f"∂²f/∂y² = {d2f_dy2(x0, y0)}")
print(f"∂²f/∂x∂y = {d2f_dxdy(x0, y0)}")
print(f"∂²f/∂y∂x = {d2f_dydx(x0, y0)}")
print(f"混合偏导数相等: {d2f_dxdy(x0, y0) == d2f_dydx(x0, y0)}")

# 4. 多元复合函数求导
print("\n4. 多元复合函数求导")

# 定义复合函数
# z = f(u, v), u = g(x, y), v = h(x, y)
def u(x, y):
    return x**2 + y**2
def v(x, y):
    return x*y
def z(u, v):
    return u**2 + v**2

# 链式法则: ∂z/∂x = ∂z/∂u * ∂u/∂x + ∂z/∂v * ∂v/∂x
def dz_dx(x, y):
    current_u = u(x, y)
    current_v = v(x, y)
    dz_du = 2 * current_u
    dz_dv = 2 * current_v
    du_dx = 2 * x
    dv_dx = y
    return dz_du * du_dx + dz_dv * dv_dx

# ∂z/∂y = ∂z/∂u * ∂u/∂y + ∂z/∂v * ∂v/∂y
def dz_dy(x, y):
    current_u = u(x, y)
    current_v = v(x, y)
    dz_du = 2 * current_u
    dz_dv = 2 * current_v
    du_dy = 2 * y
    dv_dy = x
    return dz_du * du_dy + dz_dv * dv_dy

x, y = 1, 2
print(f"复合函数 z = (x² + y²)² + (xy)²")
print(f"在点 ({x}, {y}) 处的偏导数:")
print(f"∂z/∂x = {dz_dx(x, y)}")
print(f"∂z/∂y = {dz_dy(x, y)}")

# 验证：直接计算复合函数
print(f"\n直接计算复合函数验证:")
def composite_z(x, y):
    return (x**2 + y**2)**2 + (x*y)**2

print(f"z({x}, {y}) = {composite_z(x, y)}")

# 5. 应用示例：多元函数极值
print("\n5. 应用示例：多元函数极值")

# 寻找二元函数的极值点
def g(x, y):
    return x**2 + y**2 - 2*x - 4*y + 5

# 一阶偏导数
def dg_dx(x, y):
    return 2*x - 2

def dg_dy(x, y):
    return 2*y - 4

# 二阶偏导数
def d2g_dx2(x, y):
    return 2
def d2g_dy2(x, y):
    return 2
def d2g_dxdy(x, y):
    return 0

# 求解临界点：令一阶偏导数为0
# 2x - 2 = 0 → x = 1
# 2y - 4 = 0 → y = 2
critical_x, critical_y = 1, 2
print(f"函数 g(x, y) = x² + y² - 2x - 4y + 5")
print(f"临界点: ({critical_x}, {critical_y})")
print(f"函数值: {g(critical_x, critical_y)}")

# 二阶导数判别法
D = d2g_dx2(critical_x, critical_y) * d2g_dy2(critical_x, critical_y) - (d2g_dxdy(critical_x, critical_y))**2
print(f"判别式 D = {D}")

if D > 0:
    if d2g_dx2(critical_x, critical_y) > 0:
        print("这是一个极小值点")
    else:
        print("这是一个极大值点")
elif D < 0:
    print("这是一个鞍点")
else:
    print("需要进一步分析")

# 6. 三元函数的偏导数
print("\n6. 三元函数的偏导数")

# 定义三元函数
def h(x, y, z):
    return x**2 + y**2 + z**2 + 2*x*y + 2*y*z

# 偏导数
def dh_dx(x, y, z):
    return 2*x + 2*y
def dh_dy(x, y, z):
    return 2*y + 2*x + 2*z
def dh_dz(x, y, z):
    return 2*z + 2*y

x, y, z = 1, 2, 3
print(f"函数 h(x, y, z) = x² + y² + z² + 2xy + 2yz")
print(f"在点 ({x}, {y}, {z}) 处的偏导数:")
print(f"∂h/∂x = {dh_dx(x, y, z)}")
print(f"∂h/∂y = {dh_dy(x, y, z)}")
print(f"∂h/∂z = {dh_dz(x, y, z)}")

print("\n=== 第11天学习示例结束 ===")
