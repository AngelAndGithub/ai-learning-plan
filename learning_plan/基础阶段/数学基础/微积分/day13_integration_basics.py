#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第13天：积分基础
微积分学习示例
内容：不定积分、定积分和基本积分公式
"""

import numpy as np
import scipy.integrate as integrate

print("=== 第13天：积分基础 ===")

# 1. 不定积分
print("\n1. 不定积分")

# 基本积分公式
print("基本积分公式:")
print("∫ x^n dx = (x^(n+1))/(n+1) + C, n ≠ -1")
print("∫ e^x dx = e^x + C")
print("∫ 1/x dx = ln|x| + C")
print("∫ sin(x) dx = -cos(x) + C")
print("∫ cos(x) dx = sin(x) + C")
print("∫ sec²(x) dx = tan(x) + C")

# 2. 定积分
print("\n2. 定积分")

# 定义被积函数
def f(x):
    return x**2

# 计算定积分 ∫₀¹ x² dx
result, error = integrate.quad(f, 0, 1)
print(f"定积分 ∫₀¹ x² dx = {result}")
print(f"误差估计: {error}")

# 解析解验证
analytical_result = 1/3
print(f"解析解: {analytical_result}")
print(f"误差: {abs(result - analytical_result)}")

# 3. 数值积分方法
print("\n3. 数值积分方法")

# 梯形法则
def trapezoidal_rule(f, a, b, n):
    """梯形法则计算定积分"""
    h = (b - a) / n
    sum = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        sum += f(a + i * h)
    return h * sum

# 辛普森法则
def simpson_rule(f, a, b, n):
    """辛普森法则计算定积分"""
    if n % 2 != 0:
        n += 1
    h = (b - a) / n
    sum = f(a) + f(b)
    for i in range(1, n, 2):
        sum += 4 * f(a + i * h)
    for i in range(2, n, 2):
        sum += 2 * f(a + i * h)
    return h * sum / 3

# 计算 ∫₀¹ x² dx
n = 100
trapezoidal_result = trapezoidal_rule(f, 0, 1, n)
simpson_result = simpson_rule(f, 0, 1, n)

print(f"梯形法则结果 (n={n}): {trapezoidal_result}")
print(f"辛普森法则结果 (n={n}): {simpson_result}")
print(f"解析解: {analytical_result}")

# 4. 积分的几何意义
print("\n4. 积分的几何意义")

# 计算函数 f(x) = x² 在区间 [0, 1] 下的面积
print(f"函数 f(x) = x² 在区间 [0, 1] 下的面积: {result}")

# 5. 换元积分法
print("\n5. 换元积分法")

# 示例：计算 ∫ sin(2x) dx
# 令 u = 2x, du = 2dx, dx = du/2
# ∫ sin(u) * (du/2) = (1/2)∫ sin(u) du = -1/2 cos(u) + C = -1/2 cos(2x) + C

def g(x):
    return np.sin(2*x)

# 计算定积分 ∫₀^(π/2) sin(2x) dx
result_g, error_g = integrate.quad(g, 0, np.pi/2)
print(f"定积分 ∫₀^(π/2) sin(2x) dx = {result_g}")
print(f"解析解: 1.0")

# 6. 分部积分法
print("\n6. 分部积分法")

# 示例：计算 ∫ x e^x dx
# 令 u = x, dv = e^x dx
# du = dx, v = e^x
# ∫ u dv = uv - ∫ v du = x e^x - ∫ e^x dx = x e^x - e^x + C

def h(x):
    return x * np.exp(x)

# 计算定积分 ∫₀¹ x e^x dx
result_h, error_h = integrate.quad(h, 0, 1)
print(f"定积分 ∫₀¹ x e^x dx = {result_h}")
print(f"解析解: 1.0")

# 7. 应用示例：面积计算
print("\n7. 应用示例：面积计算")

# 计算函数 f(x) = x^3 在区间 [0, 2] 下的面积
def cubic(x):
    return x**3

result_cubic, error_cubic = integrate.quad(cubic, 0, 2)
print(f"函数 f(x) = x³ 在区间 [0, 2] 下的面积: {result_cubic}")
print(f"解析解: 4.0")

# 8. 应用示例：平均值
print("\n8. 应用示例：平均值")

# 计算函数 f(x) = x² 在区间 [0, 1] 上的平均值
average_value = result / (1 - 0)
print(f"函数 f(x) = x² 在区间 [0, 1] 上的平均值: {average_value}")
print(f"解析解: 1/3 ≈ 0.3333")

# 9. 广义积分
print("\n9. 广义积分")

# 计算 ∫₀^∞ e^(-x) dx
def exponential(x):
    return np.exp(-x)

result_infinite, error_infinite = integrate.quad(exponential, 0, np.inf)
print(f"广义积分 ∫₀^∞ e^(-x) dx = {result_infinite}")
print(f"解析解: 1.0")

# 10. 多重积分
print("\n10. 多重积分")

# 计算二重积分 ∫₀¹ ∫₀¹ (x + y) dy dx
def double_integrand(y, x):  # 注意顺序：先y后x
    return x + y

result_double, error_double = integrate.dblquad(double_integrand, 0, 1, lambda x: 0, lambda x: 1)
print(f"二重积分 ∫₀¹ ∫₀¹ (x + y) dy dx = {result_double}")
print(f"解析解: 1.0")

print("\n=== 第13天学习示例结束 ===")
