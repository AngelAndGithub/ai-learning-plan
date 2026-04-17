#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第37天：支持向量机
机器学习基础学习示例
内容：支持向量机的原理、核函数和应用
"""

print("=== 第37天：支持向量机 ===")

# 1. 支持向量机基本原理
print("\n1. 支持向量机基本原理")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score

print("支持向量机（SVM）是一种强大的监督学习算法")
print("- 寻找最优超平面来分隔不同类别的数据")
print("- 最大化支持向量到超平面的距离（间隔）")
print("- 可以使用核函数处理非线性问题")

# 2. 线性SVM分类
print("\n2. 线性SVM分类")

# 生成线性可分的数据
X, y = make_classification(
    n_samples=100, n_features=2, n_informative=2, 
    n_redundant=0, n_classes=2, class_sep=2, random_state=42
)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 创建并训练线性SVM
svm_linear = SVC(kernel='linear', random_state=42)
svm_linear.fit(X_train, y_train)

# 预测
y_pred = svm_linear.predict(X_test)

# 评估
accuracy = accuracy_score(y_test, y_pred)
print(f"线性SVM分类器评估:")
print(f"准确率: {accuracy:.4f}")

print("\n分类报告:")
print(classification_report(y_test, y_pred))

# 3. 决策边界可视化
print("\n3. 决策边界可视化")

plt.figure(figsize=(8, 6))

# 绘制数据点
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr', alpha=0.6)

# 绘制决策边界
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))
Z = svm_linear.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')

# 绘制支持向量
plt.scatter(
    svm_linear.support_vectors_[:, 0],
    svm_linear.support_vectors_[:, 1],
    s=100, facecolors='none', edgecolors='black', label='Support Vectors'
)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Linear SVM Decision Boundary')
plt.legend()
plt.show()

# 4. 非线性SVM分类
print("\n4. 非线性SVM分类")

# 生成非线性可分的数据
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=100, noise=0.1, random_state=42)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 创建并训练RBF核SVM
svm_rbf = SVC(kernel='rbf', random_state=42)
svm_rbf.fit(X_train, y_train)

# 预测
y_pred = svm_rbf.predict(X_test)

# 评估
accuracy = accuracy_score(y_test, y_pred)
print(f"RBF核SVM分类器评估:")
print(f"准确率: {accuracy:.4f}")

# 决策边界可视化
plt.figure(figsize=(8, 6))

# 绘制数据点
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr', alpha=0.6)

# 绘制决策边界
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))
Z = svm_rbf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')

# 绘制支持向量
plt.scatter(
    svm_rbf.support_vectors_[:, 0],
    svm_rbf.support_vectors_[:, 1],
    s=100, facecolors='none', edgecolors='black', label='Support Vectors'
)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('RBF Kernel SVM Decision Boundary')
plt.legend()
plt.show()

# 5. 不同核函数的比较
print("\n5. 不同核函数的比较")

# 尝试不同的核函数
kernels = ['linear', 'poly', 'rbf', 'sigmoid']

plt.figure(figsize=(12, 8))

for i, kernel in enumerate(kernels, 1):
    # 创建并训练SVM
    svm = SVC(kernel=kernel, random_state=42)
    svm.fit(X_train, y_train)
    
    # 预测
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"核函数 {kernel}: 准确率 = {accuracy:.4f}")
    
    # 绘制决策边界
    plt.subplot(2, 2, i)
    
    # 绘制数据点
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr', alpha=0.6)
    
    # 绘制决策边界
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
    
    # 绘制支持向量
    plt.scatter(
        svm.support_vectors_[:, 0],
        svm.support_vectors_[:, 1],
        s=50, facecolors='none', edgecolors='black'
    )
    
    plt.title(f'Kernel: {kernel}\nAccuracy: {accuracy:.4f}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()

# 6. SVM回归
print("\n6. SVM回归")

# 生成回归数据
np.random.seed(42)
x = np.linspace(0, 10, 100).reshape(-1, 1)
y = np.sin(x) + np.random.randn(100, 1) * 0.1

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# 创建并训练SVM回归器
svr = SVR(kernel='rbf')
svr.fit(X_train, y_train.ravel())

# 预测
y_pred = svr.predict(X_test)

# 评估
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"SVM回归器评估:")
print(f"均方误差 (MSE): {mse:.4f}")
print(f"R² 评分: {r2:.4f}")

# 可视化回归结果
plt.figure(figsize=(8, 6))
plt.scatter(X_train, y_train, label='训练数据', alpha=0.6)
plt.scatter(X_test, y_test, label='测试数据', alpha=0.6, color='green')

# 绘制预测曲线
x_line = np.linspace(0, 10, 1000).reshape(-1, 1)
y_pred_line = svr.predict(x_line)
plt.plot(x_line, y_pred_line, 'r-', label='SVM回归')
plt.plot(x_line, np.sin(x_line), 'b--', label='真实函数')

plt.xlabel('X')
plt.ylabel('y')
plt.title('SVM回归')
plt.legend()
plt.show()

# 7. 超参数调优
print("\n7. 超参数调优")

from sklearn.model_selection import GridSearchCV

# 定义参数网格
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'poly', 'sigmoid']
}

# 网格搜索
grid_search = GridSearchCV(
    SVC(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy'
)

# 使用鸢尾花数据集进行网格搜索
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

grid_search.fit(X_train, y_train)

print(f"最佳参数:")
print(grid_search.best_params_)
print(f"最佳交叉验证准确率: {grid_search.best_score_:.4f}")

# 使用最佳参数训练模型
best_svm = grid_search.best_estimator_
y_pred = best_svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"测试集准确率: {accuracy:.4f}")

# 8. 处理不平衡数据
print("\n8. 处理不平衡数据")

from sklearn.datasets import make_classification

# 生成不平衡数据
X_imbalanced, y_imbalanced = make_classification(
    n_samples=1000, n_features=2, n_informative=2, 
    n_redundant=0, n_classes=2, weights=[0.9, 0.1], random_state=42
)

print(f"类别分布:")
print(f"类别0: {np.sum(y_imbalanced == 0)}")
print(f"类别1: {np.sum(y_imbalanced == 1)}")

# 划分训练集和测试集
X_train_imb, X_test_imb, y_train_imb, y_test_imb = train_test_split(
    X_imbalanced, y_imbalanced, test_size=0.2, random_state=42
)

# 不使用权重的SVM
svm_no_weight = SVC(kernel='rbf', random_state=42)
svm_no_weight.fit(X_train_imb, y_train_imb)
y_pred_no_weight = svm_no_weight.predict(X_test_imb)

# 使用权重的SVM
svm_with_weight = SVC(
    kernel='rbf', 
    class_weight='balanced', 
    random_state=42
)
svm_with_weight.fit(X_train_imb, y_train_imb)
y_pred_with_weight = svm_with_weight.predict(X_test_imb)

print("\n不使用权重的SVM:")
print(classification_report(y_test_imb, y_pred_no_weight))

print("\n使用权重的SVM:")
print(classification_report(y_test_imb, y_pred_with_weight))

# 9. 多分类问题
print("\n9. 多分类问题")

# 使用鸢尾花数据集
iris = load_iris()
X, y = iris.data, iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 创建并训练SVM分类器
svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train, y_train)

# 预测
y_pred = svm.predict(X_test)

# 评估
accuracy = accuracy_score(y_test, y_pred)
print(f"多分类SVM评估:")
print(f"准确率: {accuracy:.4f}")

print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 10. 练习
print("\n10. 练习")

# 练习1: 不同C值的影响
print("练习1: 不同C值的影响")

# 生成数据
X, y = make_classification(
    n_samples=100, n_features=2, n_informative=2, 
    n_redundant=0, n_classes=2, class_sep=1, random_state=42
)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 尝试不同的C值
C_values = [0.001, 0.01, 0.1, 1, 10, 100]

plt.figure(figsize=(12, 8))

for i, C in enumerate(C_values, 1):
    # 创建并训练SVM
    svm = SVC(kernel='linear', C=C, random_state=42)
    svm.fit(X_train, y_train)
    
    # 预测
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"C={C}: 准确率 = {accuracy:.4f}")
    
    # 绘制决策边界
    plt.subplot(2, 3, i)
    
    # 绘制数据点
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr', alpha=0.6)
    
    # 绘制决策边界
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
    
    # 绘制支持向量
    plt.scatter(
        svm.support_vectors_[:, 0],
        svm.support_vectors_[:, 1],
        s=50, facecolors='none', edgecolors='black'
    )
    
    plt.title(f'C={C}\nAccuracy: {accuracy:.4f}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()

# 练习2: 不同gamma值的影响
print("\n练习2: 不同gamma值的影响")

# 尝试不同的gamma值
gamma_values = [0.001, 0.01, 0.1, 1, 10, 100]

plt.figure(figsize=(12, 8))

for i, gamma in enumerate(gamma_values, 1):
    # 创建并训练SVM
    svm = SVC(kernel='rbf', gamma=gamma, random_state=42)
    svm.fit(X_train, y_train)
    
    # 预测
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"gamma={gamma}: 准确率 = {accuracy:.4f}")
    
    # 绘制决策边界
    plt.subplot(2, 3, i)
    
    # 绘制数据点
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr', alpha=0.6)
    
    # 绘制决策边界
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
    
    plt.title(f'gamma={gamma}\nAccuracy: {accuracy:.4f}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()

print("\n=== 第37天学习示例结束 ===")
