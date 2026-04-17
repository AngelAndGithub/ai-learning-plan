#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第33天：监督学习基础
机器学习基础学习示例
内容：监督学习的基本概念、分类和回归问题
"""

print("=== 第33天：监督学习基础 ===")

# 1. 监督学习基本概念
print("\n1. 监督学习基本概念")

print("监督学习是一种机器学习方法，其中模型从标记的训练数据中学习")
print("- 训练数据包含输入特征和对应的标签")
print("- 模型学习输入到输出的映射关系")
print("- 主要应用：分类和回归")

# 2. 分类问题
print("\n2. 分类问题")

# 生成分类数据
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# 生成二分类数据
X, y = make_classification(
    n_samples=100, n_features=2, n_informative=2, 
    n_redundant=0, n_classes=2, random_state=42
)

print(f"特征形状: {X.shape}")
print(f"标签形状: {y.shape}")
print(f"前5个特征:")
print(X[:5])
print(f"前5个标签:")
print(y[:5])

# 可视化数据
print("\n可视化分类数据:")
plt.figure(figsize=(8, 6))
plt.scatter(X[y==0, 0], X[y==0, 1], label='Class 0', c='blue')
plt.scatter(X[y==1, 0], X[y==1, 1], label='Class 1', c='red')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Classification Data')
plt.legend()
plt.show()

# 3. 回归问题
print("\n3. 回归问题")

# 生成回归数据
from sklearn.datasets import make_regression

# 生成回归数据
X_reg, y_reg = make_regression(
    n_samples=100, n_features=1, noise=10, random_state=42
)

print(f"特征形状: {X_reg.shape}")
print(f"标签形状: {y_reg.shape}")
print(f"前5个特征:")
print(X_reg[:5])
print(f"前5个标签:")
print(y_reg[:5])

# 可视化数据
print("\n可视化回归数据:")
plt.figure(figsize=(8, 6))
plt.scatter(X_reg, y_reg, c='blue')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Regression Data')
plt.show()

# 4. 数据预处理
print("\n4. 数据预处理")

# 划分训练集和测试集
from sklearn.model_selection import train_test_split

# 划分分类数据
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"分类数据划分:")
print(f"训练集特征: {X_train.shape}")
print(f"测试集特征: {X_test.shape}")
print(f"训练集标签: {y_train.shape}")
print(f"测试集标签: {y_test.shape}")

# 划分回归数据
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

print(f"\n回归数据划分:")
print(f"训练集特征: {X_reg_train.shape}")
print(f"测试集特征: {X_reg_test.shape}")
print(f"训练集标签: {y_reg_train.shape}")
print(f"测试集标签: {y_reg_test.shape}")

# 特征标准化
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n特征标准化:")
print(f"标准化前训练集均值: {X_train.mean(axis=0)}")
print(f"标准化前训练集标准差: {X_train.std(axis=0)}")
print(f"标准化后训练集均值: {X_train_scaled.mean(axis=0)}")
print(f"标准化后训练集标准差: {X_train_scaled.std(axis=0)}")

# 5. 模型训练和评估
print("\n5. 模型训练和评估")

# 分类模型：逻辑回归
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 训练模型
clf = LogisticRegression(random_state=42)
clf.fit(X_train, y_train)

# 预测
y_pred = clf.predict(X_test)

# 评估
accuracy = accuracy_score(y_test, y_pred)
print(f"分类模型准确率: {accuracy:.4f}")

print("\n分类报告:")
print(classification_report(y_test, y_pred))

print("\n混淆矩阵:")
print(confusion_matrix(y_test, y_pred))

# 回归模型：线性回归
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 训练模型
reg = LinearRegression()
reg.fit(X_reg_train, y_reg_train)

# 预测
y_reg_pred = reg.predict(X_reg_test)

# 评估
mse = mean_squared_error(y_reg_test, y_reg_pred)
r2 = r2_score(y_reg_test, y_reg_pred)

print(f"\n回归模型评估:")
print(f"均方误差 (MSE): {mse:.4f}")
print(f"R² 评分: {r2:.4f}")

# 6. 模型参数
print("\n6. 模型参数")

print("逻辑回归模型参数:")
print(f"系数: {clf.coef_}")
print(f"截距: {clf.intercept_}")

print("\n线性回归模型参数:")
print(f"系数: {reg.coef_}")
print(f"截距: {reg.intercept_}")

# 7. 决策边界可视化
print("\n7. 决策边界可视化")

# 绘制逻辑回归的决策边界
plt.figure(figsize=(8, 6))

# 绘制数据点
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr', alpha=0.6)

# 绘制决策边界
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression Decision Boundary')
plt.show()

# 8. 线性回归可视化
print("\n8. 线性回归可视化")

# 绘制线性回归拟合
plt.figure(figsize=(8, 6))

# 绘制数据点
plt.scatter(X_reg_train, y_reg_train, c='blue', alpha=0.6, label='Training data')
plt.scatter(X_reg_test, y_reg_test, c='green', alpha=0.6, label='Test data')

# 绘制拟合直线
x_line = np.linspace(X_reg.min(), X_reg.max(), 100).reshape(-1, 1)
y_line = reg.predict(x_line)
plt.plot(x_line, y_line, 'r-', label='Linear regression')

plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Linear Regression')
plt.legend()
plt.show()

# 9. 过拟合和欠拟合
print("\n9. 过拟合和欠拟合")

print("过拟合：模型在训练数据上表现很好，但在测试数据上表现较差")
print("欠拟合：模型在训练数据和测试数据上表现都较差")
print("解决方法：")
print("- 过拟合：增加数据量、特征选择、正则化")
print("- 欠拟合：增加特征、使用更复杂的模型")

# 10. 练习
print("\n10. 练习")

# 练习1: 使用不同的分类模型
print("练习1: 使用不同的分类模型")

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# K近邻分类器
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"K近邻准确率: {accuracy_knn:.4f}")

# 决策树分类器
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"决策树准确率: {accuracy_dt:.4f}")

# 练习2: 使用不同的回归模型
print("\n练习2: 使用不同的回归模型")

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

# K近邻回归器
knn_reg = KNeighborsRegressor(n_neighbors=5)
knn_reg.fit(X_reg_train, y_reg_train)
y_reg_pred_knn = knn_reg.predict(X_reg_test)
mse_knn = mean_squared_error(y_reg_test, y_reg_pred_knn)
r2_knn = r2_score(y_reg_test, y_reg_pred_knn)
print(f"K近邻回归 MSE: {mse_knn:.4f}, R²: {r2_knn:.4f}")

# 决策树回归器
dt_reg = DecisionTreeRegressor(random_state=42)
dt_reg.fit(X_reg_train, y_reg_train)
y_reg_pred_dt = dt_reg.predict(X_reg_test)
mse_dt = mean_squared_error(y_reg_test, y_reg_pred_dt)
r2_dt = r2_score(y_reg_test, y_reg_pred_dt)
print(f"决策树回归 MSE: {mse_dt:.4f}, R²: {r2_dt:.4f}")

print("\n=== 第33天学习示例结束 ===")
