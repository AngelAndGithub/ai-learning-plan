#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第15天：微积分在机器学习中的应用
微积分学习示例
内容：微积分在机器学习中的各种应用
"""

import numpy as np
import scipy.integrate as integrate
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris

print("=== 第15天：微积分在机器学习中的应用 ===")

# 1. 梯度下降法
print("\n1. 梯度下降法")

# 定义损失函数
def loss_function(weights, X, y):
    """均方误差损失函数"""
    predictions = X.dot(weights)
    return np.mean((predictions - y) ** 2)

# 计算损失函数的梯度
def compute_gradient(weights, X, y):
    """计算梯度"""
    predictions = X.dot(weights)
    return 2 / len(X) * X.T.dot(predictions - y)

# 梯度下降算法
def gradient_descent(X, y, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
    """梯度下降法"""
    n_features = X.shape[1]
    weights = np.zeros(n_features)
    loss_history = []
    
    for i in range(max_iterations):
        gradient = compute_gradient(weights, X, y)
        # 更新权重
        weights = weights - learning_rate * gradient
        # 计算损失
        loss = loss_function(weights, X, y)
        loss_history.append(loss)
        
        # 检查收敛
        if np.linalg.norm(gradient) < tolerance:
            break
    
    return weights, loss_history

# 生成示例数据
np.random.seed(42)
X = np.random.randn(100, 1)
y = 2 * X + 1 + np.random.randn(100, 1)

# 添加偏置项
X_with_bias = np.hstack([np.ones((100, 1)), X])

# 运行梯度下降
weights, loss_history = gradient_descent(X_with_bias, y)
print(f"梯度下降优化后的权重: {weights.flatten()}")
print(f"损失值: {loss_history[-1]}")

# 与sklearn比较
model = LinearRegression()
model.fit(X, y)
print(f"sklearn 模型系数: {model.coef_.flatten()}")
print(f"sklearn 模型截距: {model.intercept_}")

# 2. 反向传播算法
print("\n2. 反向传播算法")

# 定义一个简单的神经网络
class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # 初始化权重
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.random.randn(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.random.randn(output_size)
    
    def forward(self, X):
        # 前向传播
        self.z1 = X.dot(self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU激活
        self.z2 = self.a1.dot(self.W2) + self.b2
        return self.z2
    
    def compute_loss(self, y_pred, y_true):
        # 均方误差损失
        return np.mean((y_pred - y_true) ** 2)
    
    def backward(self, X, y_pred, y_true):
        # 反向传播
        # 输出层梯度
        dz2 = 2 * (y_pred - y_true) / len(X)
        dW2 = self.a1.T.dot(dz2)
        db2 = np.sum(dz2, axis=0)
        
        # 隐藏层梯度
        da1 = dz2.dot(self.W2.T)
        dz1 = da1 * (self.a1 > 0)  # ReLU导数
        dW1 = X.T.dot(dz1)
        db1 = np.sum(dz1, axis=0)
        
        return dW1, db1, dW2, db2
    
    def update_weights(self, dW1, db1, dW2, db2, learning_rate):
        # 更新权重
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

# 生成示例数据
X_nn = np.random.randn(100, 2)
y_nn = X_nn[:, 0] + X_nn[:, 1] + np.random.randn(100) * 0.1
y_nn = y_nn.reshape(-1, 1)

# 创建神经网络
nn = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1)

# 训练神经网络
learning_rate = 0.01
n_epochs = 1000

for epoch in range(n_epochs):
    # 前向传播
    y_pred = nn.forward(X_nn)
    # 计算损失
    loss = nn.compute_loss(y_pred, y_nn)
    # 反向传播
    dW1, db1, dW2, db2 = nn.backward(X_nn, y_pred, y_nn)
    # 更新权重
    nn.update_weights(dW1, db1, dW2, db2, learning_rate)
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# 测试模型
X_test = np.array([[1, 2], [3, 4]])
y_pred_test = nn.forward(X_test)
print(f"测试输入: {X_test}")
print(f"预测输出: {y_pred_test.flatten()}")
print(f"实际输出: {X_test[:, 0] + X_test[:, 1]}")

# 3. 激活函数的导数
print("\n3. 激活函数的导数")

# Sigmoid函数及其导数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# ReLU函数及其导数
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Tanh函数及其导数
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

# 测试激活函数的导数
x = 0.5
print(f"Sigmoid({x}) = {sigmoid(x)}, 导数 = {sigmoid_derivative(x)}")
print(f"ReLU({x}) = {relu(x)}, 导数 = {relu_derivative(x)}")
print(f"Tanh({x}) = {tanh(x)}, 导数 = {tanh_derivative(x)}")

# 4. 损失函数的导数
print("\n4. 损失函数的导数")

# 均方误差损失函数及其导数
def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def mse_loss_derivative(y_pred, y_true):
    return 2 * (y_pred - y_true) / len(y_true)

# 交叉熵损失函数及其导数
def cross_entropy_loss(y_pred, y_true):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def cross_entropy_loss_derivative(y_pred, y_true):
    return (y_pred - y_true) / (y_pred * (1 - y_pred))

# 测试损失函数的导数
y_pred = np.array([0.8, 0.2, 0.6])
y_true = np.array([1, 0, 1])

print(f"均方误差损失: {mse_loss(y_pred, y_true)}")
print(f"均方误差导数: {mse_loss_derivative(y_pred, y_true)}")

# 5. 应用示例：线性回归的梯度下降
print("\n5. 应用示例：线性回归的梯度下降")

# 加载鸢尾花数据集
iris = load_iris()
X_iris = iris.data[:, :2]  # 只使用前两个特征
y_iris = iris.data[:, 2]    # 预测花瓣长度

# 添加偏置项
X_iris_with_bias = np.hstack([np.ones((X_iris.shape[0], 1)), X_iris])

# 运行梯度下降
weights_iris, loss_history_iris = gradient_descent(X_iris_with_bias, y_iris.reshape(-1, 1))
print(f"鸢尾花数据集的线性回归权重: {weights_iris.flatten()}")
print(f"最终损失: {loss_history_iris[-1]}")

# 6. 应用示例：逻辑回归
print("\n6. 应用示例：逻辑回归")

# 二分类问题
y_binary = (iris.target == 0).astype(int)

# 逻辑回归的损失函数（交叉熵）
def logistic_loss(weights, X, y):
    z = X.dot(weights)
    y_pred = sigmoid(z)
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

# 逻辑回归的梯度
def logistic_gradient(weights, X, y):
    z = X.dot(weights)
    y_pred = sigmoid(z)
    return X.T.dot(y_pred - y) / len(X)

# 逻辑回归的梯度下降
def logistic_regression_gradient_descent(X, y, learning_rate=0.1, max_iterations=1000):
    n_features = X.shape[1]
    weights = np.zeros(n_features)
    loss_history = []
    
    for i in range(max_iterations):
        gradient = logistic_gradient(weights, X, y)
        weights = weights - learning_rate * gradient
        loss = logistic_loss(weights, X, y)
        loss_history.append(loss)
    
    return weights, loss_history

# 运行逻辑回归
weights_logistic, loss_history_logistic = logistic_regression_gradient_descent(
    X_iris_with_bias, y_binary
)
print(f"逻辑回归权重: {weights_logistic}")
print(f"最终损失: {loss_history_logistic[-1]}")

# 7. 应用示例：神经网络的梯度下降
print("\n7. 应用示例：神经网络的梯度下降")

# 简单的神经网络用于分类
class ClassificationNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.random.randn(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.random.randn(output_size)
    
    def forward(self, X):
        self.z1 = X.dot(self.W1) + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1.dot(self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2
    
    def compute_loss(self, y_pred, y_true):
        return cross_entropy_loss(y_pred, y_true)
    
    def backward(self, X, y_pred, y_true):
        dz2 = y_pred - y_true
        dW2 = self.a1.T.dot(dz2)
        db2 = np.sum(dz2, axis=0)
        
        da1 = dz2.dot(self.W2.T)
        dz1 = da1 * relu_derivative(self.z1)
        dW1 = X.T.dot(dz1)
        db1 = np.sum(dz1, axis=0)
        
        return dW1, db1, dW2, db2
    
    def update_weights(self, dW1, db1, dW2, db2, learning_rate):
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

# 创建分类神经网络
class_nn = ClassificationNeuralNetwork(input_size=2, hidden_size=4, output_size=1)

# 训练神经网络
learning_rate = 0.01
n_epochs = 1000

for epoch in range(n_epochs):
    y_pred = class_nn.forward(X_iris)
    loss = class_nn.compute_loss(y_pred, y_binary.reshape(-1, 1))
    dW1, db1, dW2, db2 = class_nn.backward(X_iris, y_pred, y_binary.reshape(-1, 1))
    class_nn.update_weights(dW1, db1, dW2, db2, learning_rate)
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

print("\n=== 第15天学习示例结束 ===")
