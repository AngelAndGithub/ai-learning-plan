#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第9天：导数基础
微积分学习示例
内容：导数的定义、几何意义和基本导数公式
"""

import numpy as np
import matplotlib.pyplot as plt

print("=== 第9天：导数基础 ===")

# 1. 导数的定义
print("\n1. 导数的定义")

# 定义一个函数
def f(x):
    return x**2 + 2*x + 1

# 数值导数计算
def numerical_derivative(f, x, h=1e-6):
    """使用中心差分法计算数值导数"""
    return (f(x + h) - f(x - h)) / (2 * h)

# 计算在x=2处的导数
x = 2
derivative = numerical_derivative(f, x)
print(f"函数 f(x) = x² + 2x + 1")
print(f"在 x={x} 处的导数: {derivative}")

# 解析解验证
# f'(x) = 2x + 2
analytical_derivative = 2 * x + 2
print(f"解析解: {analytical_derivative}")
print(f"数值导数与解析解的误差: {abs(derivative - analytical_derivative)}")

# 2. 导数的几何意义
print("\n2. 导数的几何意义")

# 计算在x=2处的切线
x0 = 2
y0 = f(x0)
slope = analytical_derivative

# 切线方程: y = slope * (x - x0) + y0
def tangent_line(x):
    return slope * (x - x0) + y0

# 绘制函数和切线
x_values = np.linspace(0, 4, 100)
y_values = f(x_values)
tangent_values = tangent_line(x_values)

print("绘制函数 f(x) = x² + 2x + 1 和在 x=2 处的切线")
print(f"切线方程: y = {slope}(x - {x0}) + {y0}")
print(f"切线方程简化: y = {slope}x + {y0 - slope*x0}")

# 3. 基本导数公式
print("\n3. 基本导数公式")

# 常数函数的导数
def constant_function(x):
    return 5

print(f"常数函数 f(x) = 5 的导数: {numerical_derivative(constant_function, 2)}")

# 幂函数的导数
def power_function(x):
    return x**3

print(f"幂函数 f(x) = x³ 的导数在 x=2 处: {numerical_derivative(power_function, 2)}")
print(f"解析解 (3x²): {3*(2**2)}")

# 指数函数的导数
def exponential_function(x):
    return np.exp(x)

print(f"指数函数 f(x) = e^x 的导数在 x=1 处: {numerical_derivative(exponential_function, 1)}")
print(f"解析解 (e^x): {np.exp(1)}")

# 对数函数的导数
def logarithm_function(x):
    return np.log(x)

print(f"对数函数 f(x) = ln(x) 的导数在 x=2 处: {numerical_derivative(logarithm_function, 2)}")
print(f"解析解 (1/x): {1/2}")

# 三角函数的导数
def sine_function(x):
    return np.sin(x)

print(f"正弦函数 f(x) = sin(x) 的导数在 x=0 处: {numerical_derivative(sine_function, 0)}")
print(f"解析解 (cos(x)): {np.cos(0)}")

# 4. 导数与连续性的关系
print("\n4. 导数与连续性的关系")

# 连续但不可导的函数
print("连续但不可导的函数示例: f(x) = |x|")

def absolute_value(x):
    return abs(x)

# 在x=0处的导数
left_derivative = (absolute_value(0 + 1e-6) - absolute_value(0)) / 1e-6
right_derivative = (absolute_value(0) - absolute_value(0 - 1e-6)) / 1e-6
print(f"左导数: {left_derivative}")
print(f"右导数: {right_derivative}")
print(f"导数是否存在: {abs(left_derivative - right_derivative) < 1e-6}")

# 5. 应用示例：优化
print("\n5. 应用示例：优化")

# 寻找函数的最小值
def quadratic_function(x):
    return x**2 - 4*x + 5

# 导数为零的点即为极值点
def derivative_quadratic(x):
    return 2*x - 4

# 解方程 2x - 4 = 0
min_x = 2
min_y = quadratic_function(min_x)
print(f"函数 f(x) = x² - 4x + 5 的最小值在 x={min_x}, y={min_y}")

# 验证导数为零
print(f"在 x={min_x} 处的导数: {derivative_quadratic(min_x)}")

# 6. 高阶导数
print("\n6. 高阶导数")

# 计算二阶导数
def second_derivative(f, x, h=1e-6):
    """计算二阶导数"""
    return (f(x + h) - 2*f(x) + f(x - h)) / (h**2)

# 计算f(x) = x² + 2x + 1的二阶导数
second_deriv = second_derivative(f, 2)
print(f"函数 f(x) = x² + 2x + 1 的二阶导数: {second_deriv}")
print(f"解析解: 2")

print("\n=== 第9天学习示例结束 ===")
