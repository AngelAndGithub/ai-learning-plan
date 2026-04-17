#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第12天：梯度与方向导数
微积分学习示例
内容：梯度、方向导数和梯度下降法
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

print("=== 第12天：梯度与方向导数 ===")

# 1. 梯度
print("\n1. 梯度")

# 定义二元函数
def f(x, y):
    return x**2 + 2*y**2

# 计算梯度
# ∇f = (∂f/∂x, ∂f/∂y) = (2x, 4y)
def gradient_f(x, y):
    return np.array([2*x, 4*y])

# 计算在点(1, 1)处的梯度
x0, y0 = 1, 1
grad = gradient_f(x0, y0)
print(f"函数 f(x, y) = x² + 2y²")
print(f"在点 ({x0}, {y0}) 处的梯度: {grad}")

# 2. 方向导数
print("\n2. 方向导数")

# 定义方向向量
u = np.array([1, 1])
# 单位化方向向量
u_unit = u / np.linalg.norm(u)
print(f"方向向量: {u}")
print(f"单位方向向量: {u_unit}")

# 方向导数：D_uf = ∇f · u
# 计算方向导数
directional_derivative = np.dot(grad, u_unit)
print(f"在方向 {u_unit} 上的方向导数: {directional_derivative}")

# 验证方向导数公式
def directional_derivative_func(x, y, direction):
    """计算方向导数"""
    grad = gradient_f(x, y)
    direction_unit = direction / np.linalg.norm(direction)
    return np.dot(grad, direction_unit)

print(f"方向导数验证: {directional_derivative_func(x0, y0, u)}")

# 3. 梯度的性质
print("\n3. 梯度的性质")

# 性质1：梯度指向函数增长最快的方向
print("梯度指向函数增长最快的方向")
print(f"在点 ({x0}, {y0}) 处，梯度方向: {grad}")

# 性质2：梯度的模长是最大方向导数
max_directional_derivative = np.linalg.norm(grad)
print(f"最大方向导数 (梯度的模长): {max_directional_derivative}")

# 性质3：梯度与等值面垂直
print("梯度与等值面垂直")

# 4. 梯度下降法
print("\n4. 梯度下降法")

# 定义目标函数
def objective_function(x):
    """x是一个向量"""
    return x[0]**2 + 2*x[1]**2

def gradient_objective(x):
    """计算目标函数的梯度"""
    return np.array([2*x[0], 4*x[1]])

# 梯度下降算法
def gradient_descent(gradient, initial_x, learning_rate=0.1, max_iterations=100, tolerance=1e-6):
    """梯度下降法"""
    x = initial_x
    history = [x.copy()]
    
    for i in range(max_iterations):
        grad = gradient(x)
        # 检查梯度是否足够小
        if np.linalg.norm(grad) < tolerance:
            break
        # 沿负梯度方向更新
        x = x - learning_rate * grad
        history.append(x.copy())
    
    return x, history

# 初始点
initial_x = np.array([2, 1])
print(f"初始点: {initial_x}")
print(f"初始函数值: {objective_function(initial_x)}")

# 运行梯度下降
result, history = gradient_descent(gradient_objective, initial_x)
print(f"\n梯度下降结果: {result}")
print(f"最小函数值: {objective_function(result)}")
print(f"迭代次数: {len(history)}")

# 打印优化过程
print("\n优化过程:")
for i, x in enumerate(history[:5] + history[-5:]):
    if i < 5 or i >= len(history) - 5:
        print(f"迭代 {i}: x = {x}, f(x) = {objective_function(x)}")

# 5. 应用示例：线性回归
print("\n5. 应用示例：线性回归")

# 生成线性回归数据
np.random.seed(42)
x = np.linspace(0, 10, 100)
y = 2*x + 1 + np.random.randn(100) * 2

# 定义线性回归模型
# 模型：y = w0 + w1*x
def linear_model(weights, x):
    return weights[0] + weights[1] * x

# 定义损失函数（均方误差）
def mse_loss(weights, x, y):
    predictions = linear_model(weights, x)
    return np.mean((predictions - y)**2)

# 计算损失函数的梯度
def gradient_mse(weights, x, y):
    predictions = linear_model(weights, x)
    error = predictions - y
    dw0 = 2 * np.mean(error)
    dw1 = 2 * np.mean(error * x)
    return np.array([dw0, dw1])

# 初始权重
initial_weights = np.array([0, 0])
print(f"初始权重: {initial_weights}")
print(f"初始损失: {mse_loss(initial_weights, x, y)}")

# 运行梯度下降
result_weights, weights_history = gradient_descent(
    lambda w: gradient_mse(w, x, y),
    initial_weights,
    learning_rate=0.01,
    max_iterations=1000
)

print(f"\n优化后的权重: {result_weights}")
print(f"最小损失: {mse_loss(result_weights, x, y)}")

# 6. 等高线图和梯度
print("\n6. 等高线图和梯度")

# 绘制函数 f(x, y) = x² + 2y² 的等高线
x_vals = np.linspace(-3, 3, 100)
y_vals = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = X**2 + 2*Y**2

print("绘制函数 f(x, y) = x² + 2y² 的等高线图")
print("在点 (1, 1) 处的梯度方向")
print("梯度指向函数值增加最快的方向")

# 7. 多变量梯度下降
print("\n7. 多变量梯度下降")

# 定义三元函数
def multi_var_function(x):
    """x是一个三维向量"""
    return x[0]**2 + x[1]**2 + x[2]**2 + 2*x[0]*x[1]

def gradient_multi_var(x):
    return np.array([2*x[0] + 2*x[1], 2*x[1] + 2*x[0], 2*x[2]])

# 初始点
initial_point = np.array([1, 1, 1])
print(f"初始点: {initial_point}")
print(f"初始函数值: {multi_var_function(initial_point)}")

# 运行梯度下降
result_point, point_history = gradient_descent(
    gradient_multi_var,
    initial_point,
    learning_rate=0.1
)

print(f"\n优化结果: {result_point}")
print(f"最小函数值: {multi_var_function(result_point)}")

print("\n=== 第12天学习示例结束 ===")
