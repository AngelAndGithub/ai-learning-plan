#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第16天：微积分综合复习
微积分学习示例
内容：微积分的综合应用和复习
"""

import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

print("=== 第16天：微积分综合复习 ===")

# 1. 导数复习
print("\n1. 导数复习")

# 定义函数
def f(x):
    return x**3 + 2*x**2 - 4*x + 1

# 解析导数
def f_prime(x):
    return 3*x**2 + 4*x - 4

# 数值导数
def numerical_derivative(func, x, h=1e-6):
    return (func(x + h) - func(x - h)) / (2 * h)

# 测试导数
x = 2
print(f"函数 f(x) = x³ + 2x² - 4x + 1")
print(f"在 x={x} 处的导数:")
print(f"解析解: {f_prime(x)}")
print(f"数值解: {numerical_derivative(f, x)}")
print(f"误差: {abs(f_prime(x) - numerical_derivative(f, x))}")

# 2. 积分复习
print("\n2. 积分复习")

# 计算定积分 ∫₀² (x³ + 2x² - 4x + 1) dx
def integrand(x):
    return x**3 + 2*x**2 - 4*x + 1

result, error = integrate.quad(integrand, 0, 2)
print(f"定积分 ∫₀² (x³ + 2x² - 4x + 1) dx = {result}")

# 解析解验证
# 原函数: F(x) = (x⁴)/4 + (2x³)/3 - 2x² + x
# F(2) - F(0) = (16/4 + 16/3 - 8 + 2) - 0 = 4 + 16/3 - 6 = 16/3 - 2 = 10/3 ≈ 3.3333
analytical_result = 10/3
print(f"解析解: {analytical_result}")
print(f"误差: {abs(result - analytical_result)}")

# 3. 梯度和方向导数
print("\n3. 梯度和方向导数")

# 定义二元函数
def g(x, y):
    return x**2 + y**2 + 2*x*y

# 计算梯度
def gradient_g(x, y):
    return np.array([2*x + 2*y, 2*y + 2*x])

# 计算在点(1, 1)处的梯度
x0, y0 = 1, 1
grad = gradient_g(x0, y0)
print(f"函数 g(x, y) = x² + y² + 2xy")
print(f"在点 ({x0}, {y0}) 处的梯度: {grad}")

# 计算方向导数
u = np.array([1, 1])
u_unit = u / np.linalg.norm(u)
directional_derivative = np.dot(grad, u_unit)
print(f"在方向 {u_unit} 上的方向导数: {directional_derivative}")

# 4. 常微分方程
print("\n4. 常微分方程")

# 示例：dy/dt = -2y + 10
# 解析解：y(t) = 5 + (y0 - 5)e^(-2t)
def dydt(y, t):
    return -2 * y + 10

# 初始条件
y0 = 0
# 时间点
t = np.linspace(0, 5, 100)

# 数值求解
sol = integrate.odeint(dydt, y0, t)

# 解析解
analytical_solution = 5 + (y0 - 5) * np.exp(-2 * t)

print(f"常微分方程 dy/dt = -2y + 10")
print(f"初始条件 y(0) = {y0}")
print(f"数值解在 t=2 处: {sol[40][0]}")
print(f"解析解在 t=2 处: {analytical_solution[40]}")
print(f"误差: {abs(sol[40][0] - analytical_solution[40])}")

# 5. 微积分在优化中的应用
print("\n5. 微积分在优化中的应用")

# 寻找函数的最小值
def objective_function(x):
    return x**4 - 4*x**3 + 2*x**2 + 4*x + 1

# 导数
def objective_derivative(x):
    return 4*x**3 - 12*x**2 + 4*x + 4

# 二阶导数
def objective_second_derivative(x):
    return 12*x**2 - 24*x + 4

# 使用梯度下降法寻找最小值
learning_rate = 0.01
x = 3  # 初始点
n_iterations = 1000

for i in range(n_iterations):
    grad = objective_derivative(x)
    x = x - learning_rate * grad
    
    if i % 100 == 0:
        print(f"迭代 {i}: x = {x}, f(x) = {objective_function(x)}")

print(f"\n最小值点: x = {x}")
print(f"最小值: f(x) = {objective_function(x)}")

# 6. 应用示例：曲线下面积
print("\n6. 应用示例：曲线下面积")

# 计算标准正态分布的累积分布函数
def normal_pdf(x, mu=0, sigma=1):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# 计算 P(X ≤ 1)
result_normal, error_normal = integrate.quad(normal_pdf, -np.inf, 1)
print(f"标准正态分布 P(X ≤ 1) = {result_normal}")
print(f"理论值: 0.8413")

# 7. 应用示例：体积计算
print("\n7. 应用示例：体积计算")

# 计算旋转体体积：由曲线 y = x², x=0 到 x=1 绕 x 轴旋转而成的体积
# 体积公式：V = π ∫₀¹ [f(x)]² dx
def volume_integrand(x):
    return (x**2)**2  # [f(x)]² = (x²)² = x⁴

volume_result, volume_error = integrate.quad(volume_integrand, 0, 1)
volume = np.pi * volume_result
print(f"旋转体体积: {volume}")
print(f"解析解: π/5 ≈ 0.6283")

# 8. 应用示例：弧长计算
print("\n8. 应用示例：弧长计算")

# 计算曲线 y = x² 从 x=0 到 x=1 的弧长
# 弧长公式：L = ∫₀¹ √(1 + [f'(x)]²) dx
def arc_length_integrand(x):
    return np.sqrt(1 + (2*x)**2)

arc_length_result, arc_length_error = integrate.quad(arc_length_integrand, 0, 1)
print(f"曲线 y = x² 从 x=0 到 x=1 的弧长: {arc_length_result}")

# 9. 应用示例：概率密度函数
print("\n9. 应用示例：概率密度函数")

# 验证概率密度函数的归一性
def uniform_pdf(x):
    return 1 if 0 <= x <= 1 else 0

result_uniform, error_uniform = integrate.quad(uniform_pdf, -np.inf, np.inf)
print(f"均匀分布 PDF 的积分: {result_uniform}")
print(f"应为 1.0")

# 10. 综合应用：机器学习中的梯度下降
print("\n10. 综合应用：机器学习中的梯度下降")

# 生成线性回归数据
np.random.seed(42)
X = np.random.randn(100, 1)
y = 3 * X + 2 + np.random.randn(100, 1)

# 添加偏置项
X_with_bias = np.hstack([np.ones((100, 1)), X])

# 定义损失函数
def mse_loss(weights, X, y):
    predictions = X.dot(weights)
    return np.mean((predictions - y) ** 2)

# 计算梯度
def compute_gradient(weights, X, y):
    predictions = X.dot(weights)
    return 2 / len(X) * X.T.dot(predictions - y)

# 梯度下降
learning_rate = 0.1
weights = np.zeros((2, 1))
n_iterations = 100

for i in range(n_iterations):
    gradient = compute_gradient(weights, X_with_bias, y)
    weights = weights - learning_rate * gradient
    
    if i % 20 == 0:
        loss = mse_loss(weights, X_with_bias, y)
        print(f"迭代 {i}: 损失 = {loss}, 权重 = {weights.flatten()}")

print(f"\n最终权重: {weights.flatten()}")
print(f"预期权重: [2, 3]")

# 11. 复习总结
print("\n11. 微积分学习总结")

print("微积分的核心概念:")
print("1. 导数：函数在某一点的变化率")
print("2. 积分：函数在区间上的累积")
print("3. 梯度：多元函数的方向导数最大值")
print("4. 微分方程：描述变量之间的变化关系")

print("\n微积分在机器学习中的应用:")
print("1. 梯度下降法：优化模型参数")
print("2. 反向传播：神经网络训练")
print("3. 损失函数：评估模型性能")
print("4. 概率模型：贝叶斯推断")

print("\n重要的微积分定理:")
print("1. 微积分基本定理：导数和积分的关系")
print("2. 中值定理：函数在区间内的平均变化率")
print("3. 泰勒展开：函数的多项式近似")
print("4. 极值定理：寻找函数的最大值和最小值")

print("\n=== 第16天学习示例结束 ===")
