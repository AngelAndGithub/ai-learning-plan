#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第38天：k近邻算法
机器学习基础学习示例
内容：k近邻算法的原理、参数选择和应用
"""

print("=== 第38天：k近邻算法 ===")

# 1. k近邻算法基本原理
print("\n1. k近邻算法基本原理")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, load_iris, make_regression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

print("k近邻（KNN）是一种基于实例的监督学习算法")
print("- 对新样本，根据其k个最近邻的类别进行投票或平均")
print("- 是非参数算法，不做任何假设")
print("- 计算复杂度较高，需要存储所有训练数据")

# 2. KNN分类
print("\n2. KNN分类")

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 创建并训练KNN分类器
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# 预测
y_pred = knn.predict(X_test_scaled)

# 评估
accuracy = accuracy_score(y_test, y_pred)
print(f"KNN分类器评估:")
print(f"准确率: {accuracy:.4f}")

print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

print("\n混淆矩阵:")
print(confusion_matrix(y_test, y_pred))

# 3. 不同k值的影响
print("\n3. 不同k值的影响")

# 尝试不同的k值
k_values = [1, 3, 5, 7, 9, 11, 13, 15]
train_accuracies = []
test_accuracies = []

for k in k_values:
    # 创建并训练KNN分类器
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    
    # 计算训练集准确率
    y_train_pred = knn.predict(X_train_scaled)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_accuracies.append(train_accuracy)
    
    # 计算测试集准确率
    y_test_pred = knn.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_accuracies.append(test_accuracy)
    
    print(f"k={k}: 训练集准确率 = {train_accuracy:.4f}, 测试集准确率 = {test_accuracy:.4f}")

# 绘制准确率曲线
plt.figure(figsize=(8, 6))
plt.plot(k_values, train_accuracies, label='训练集准确率')
plt.plot(k_values, test_accuracies, label='测试集准确率')
plt.xlabel('k值')
plt.ylabel('准确率')
plt.title('不同k值对KNN分类器性能的影响')
plt.legend()
plt.show()

# 4. 距离度量
print("\n4. 距离度量")

print("KNN中常用的距离度量:")
print("1. 欧氏距离（euclidean）: √(Σ(xi-yi)²)")
print("2. 曼哈顿距离（manhattan）: Σ|xi-yi|")
print("3. 切比雪夫距离（chebyshev）: max|xi-yi|")
print("4. 闵可夫斯基距离（minkowski）: (Σ|xi-yi|^p)^(1/p)")

# 尝试不同的距离度量
distance_metrics = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']

for metric in distance_metrics:
    # 创建并训练KNN分类器
    knn = KNeighborsClassifier(n_neighbors=5, metric=metric)
    knn.fit(X_train_scaled, y_train)
    
    # 预测
    y_pred = knn.predict(X_test_scaled)
    
    # 评估
    accuracy = accuracy_score(y_test, y_pred)
    print(f"距离度量 {metric}: 准确率 = {accuracy:.4f}")

# 5. KNN回归
print("\n5. KNN回归")

# 生成回归数据
np.random.seed(42)
x = np.linspace(0, 10, 100).reshape(-1, 1)
y = np.sin(x) + np.random.randn(100, 1) * 0.1

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# 创建并训练KNN回归器
knn_reg = KNeighborsRegressor(n_neighbors=5)
knn_reg.fit(X_train, y_train.ravel())

# 预测
y_pred = knn_reg.predict(X_test)

# 评估
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"KNN回归器评估:")
print(f"均方误差 (MSE): {mse:.4f}")
print(f"R² 评分: {r2:.4f}")

# 可视化回归结果
plt.figure(figsize=(8, 6))
plt.scatter(X_train, y_train, label='训练数据', alpha=0.6)
plt.scatter(X_test, y_test, label='测试数据', alpha=0.6, color='green')

# 绘制预测曲线
x_line = np.linspace(0, 10, 1000).reshape(-1, 1)
y_pred_line = knn_reg.predict(x_line)
plt.plot(x_line, y_pred_line, 'r-', label='KNN回归')
plt.plot(x_line, np.sin(x_line), 'b--', label='真实函数')

plt.xlabel('X')
plt.ylabel('y')
plt.title('KNN回归')
plt.legend()
plt.show()

# 6. 不同k值对回归的影响
print("\n6. 不同k值对回归的影响")

# 尝试不同的k值
k_values = [1, 3, 5, 7, 9, 11, 13, 15]
train_mses = []
test_mses = []

for k in k_values:
    # 创建并训练KNN回归器
    knn_reg = KNeighborsRegressor(n_neighbors=k)
    knn_reg.fit(X_train, y_train.ravel())
    
    # 计算训练集MSE
    y_train_pred = knn_reg.predict(X_train)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_mses.append(train_mse)
    
    # 计算测试集MSE
    y_test_pred = knn_reg.predict(X_test)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_mses.append(test_mse)
    
    print(f"k={k}: 训练集MSE = {train_mse:.4f}, 测试集MSE = {test_mse:.4f}")

# 绘制MSE曲线
plt.figure(figsize=(8, 6))
plt.plot(k_values, train_mses, label='训练集MSE')
plt.plot(k_values, test_mses, label='测试集MSE')
plt.xlabel('k值')
plt.ylabel('MSE')
plt.title('不同k值对KNN回归器性能的影响')
plt.legend()
plt.show()

# 7. 加权KNN
print("\n7. 加权KNN")

# 尝试不同的权重策略
weights = ['uniform', 'distance']

for weight in weights:
    # 创建并训练KNN分类器
    knn = KNeighborsClassifier(n_neighbors=5, weights=weight)
    knn.fit(X_train_scaled, y_train)
    
    # 预测
    y_pred = knn.predict(X_test_scaled)
    
    # 评估
    accuracy = accuracy_score(y_test, y_pred)
    print(f"权重策略 {weight}: 准确率 = {accuracy:.4f}")

# 8. 交叉验证
print("\n8. 交叉验证")

from sklearn.model_selection import cross_val_score

# 对KNN分类器进行交叉验证
knn = KNeighborsClassifier(n_neighbors=5)
cv_scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy')

print(f"5折交叉验证准确率:")
print(cv_scores)
print(f"平均准确率: {cv_scores.mean():.4f}")
print(f"标准差: {cv_scores.std():.4f}")

# 9. 超参数调优
print("\n9. 超参数调优")

from sklearn.model_selection import GridSearchCV

# 定义参数网格
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'chebyshev']
}

# 网格搜索
grid_search = GridSearchCV(
    KNeighborsClassifier(),
    param_grid,
    cv=5,
    scoring='accuracy'
)

grid_search.fit(X_train_scaled, y_train)

print(f"最佳参数:")
print(grid_search.best_params_)
print(f"最佳交叉验证准确率: {grid_search.best_score_:.4f}")

# 使用最佳参数训练模型
best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"测试集准确率: {accuracy:.4f}")

# 10. 练习
print("\n10. 练习")

# 练习1: 决策边界可视化
print("练习1: 决策边界可视化")

# 生成二分类数据
X, y = make_classification(
    n_samples=200, n_features=2, n_informative=2, 
    n_redundant=0, n_classes=2, random_state=42
)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 尝试不同的k值
k_values = [1, 3, 5, 10]

plt.figure(figsize=(12, 8))

for i, k in enumerate(k_values, 1):
    # 创建并训练KNN分类器
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    
    # 预测
    y_pred = knn.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"k={k}: 准确率 = {accuracy:.4f}")
    
    # 绘制决策边界
    plt.subplot(2, 2, i)
    
    # 绘制数据点
    plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap='bwr', alpha=0.6)
    
    # 绘制决策边界
    x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
    y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
    
    plt.title(f'k={k}\nAccuracy: {accuracy:.4f}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()

# 练习2: 不同距离度量对回归的影响
print("\n练习2: 不同距离度量对回归的影响")

# 生成回归数据
np.random.seed(42)
x = np.linspace(0, 10, 100).reshape(-1, 1)
y = np.sin(x) + np.random.randn(100, 1) * 0.1

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# 尝试不同的距离度量
distance_metrics = ['euclidean', 'manhattan', 'chebyshev']

plt.figure(figsize=(12, 8))

for i, metric in enumerate(distance_metrics, 1):
    # 创建并训练KNN回归器
    knn_reg = KNeighborsRegressor(n_neighbors=5, metric=metric)
    knn_reg.fit(X_train, y_train.ravel())
    
    # 预测
    y_pred = knn_reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"距离度量 {metric}: MSE = {mse:.4f}, R² = {r2:.4f}")
    
    # 绘制预测曲线
    plt.subplot(3, 1, i)
    plt.scatter(X_train, y_train, label='训练数据', alpha=0.6)
    plt.scatter(X_test, y_test, label='测试数据', alpha=0.6, color='green')
    
    x_line = np.linspace(0, 10, 1000).reshape(-1, 1)
    y_pred_line = knn_reg.predict(x_line)
    plt.plot(x_line, y_pred_line, 'r-', label=f'KNN回归 ({metric})')
    plt.plot(x_line, np.sin(x_line), 'b--', label='真实函数')
    
    plt.title(f'KNN回归 ({metric})\nMSE: {mse:.4f}, R²: {r2:.4f}')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()

plt.tight_layout()
plt.show()

print("\n=== 第38天学习示例结束 ===")
