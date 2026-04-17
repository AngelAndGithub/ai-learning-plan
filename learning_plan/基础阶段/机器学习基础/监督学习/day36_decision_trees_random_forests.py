#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第36天：决策树和随机森林
机器学习基础学习示例
内容：决策树的原理、随机森林的应用和模型评估
"""

print("=== 第36天：决策树和随机森林 ===")

# 1. 决策树基本原理
print("\n1. 决策树基本原理")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score

print("决策树是一种基于树结构的监督学习算法")
print("- 从根节点开始，根据特征值进行划分")
print("- 每个内部节点表示一个特征测试")
print("- 每个叶节点表示一个类别或预测值")
print("- 常用的划分标准：Gini impurity, Entropy, Mean Squared Error")

# 2. 分类决策树
print("\n2. 分类决策树")

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 创建并训练决策树分类器
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# 预测
y_pred = dt_classifier.predict(X_test)

# 评估
accuracy = accuracy_score(y_test, y_pred)
print(f"决策树分类器评估:")
print(f"准确率: {accuracy:.4f}")

print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

print("\n混淆矩阵:")
print(confusion_matrix(y_test, y_pred))

# 3. 回归决策树
print("\n3. 回归决策树")

# 生成回归数据
np.random.seed(42)
x = np.linspace(0, 10, 100).reshape(-1, 1)
y = x**2 + np.random.randn(100, 1) * 5

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# 创建并训练决策树回归器
dt_regressor = DecisionTreeRegressor(random_state=42)
dt_regressor.fit(X_train, y_train)

# 预测
y_pred = dt_regressor.predict(X_test)

# 评估
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"决策树回归器评估:")
print(f"均方误差 (MSE): {mse:.4f}")
print(f"R² 评分: {r2:.4f}")

# 可视化回归结果
plt.figure(figsize=(8, 6))
plt.scatter(X_train, y_train, label='训练数据', alpha=0.6)
plt.scatter(X_test, y_test, label='测试数据', alpha=0.6, color='green')

# 绘制预测曲线
x_line = np.linspace(0, 10, 1000).reshape(-1, 1)
y_pred_line = dt_regressor.predict(x_line)
plt.plot(x_line, y_pred_line, 'r-', label='决策树回归')

plt.xlabel('X')
plt.ylabel('y')
plt.title('决策树回归')
plt.legend()
plt.show()

# 4. 决策树可视化
print("\n4. 决策树可视化")

from sklearn.tree import plot_tree

# 可视化决策树
plt.figure(figsize=(12, 8))
plot_tree(dt_classifier, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title('决策树可视化')
plt.show()

# 5. 随机森林
print("\n5. 随机森林")

print("随机森林是一种集成学习方法，由多个决策树组成")
print("- 每个决策树使用不同的训练子集")
print("- 每个决策树使用不同的特征子集")
print("- 最终预测通过投票或平均获得")

# 创建并训练随机森林分类器
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# 预测
y_pred = rf_classifier.predict(X_test)

# 评估
accuracy = accuracy_score(y_test, y_pred)
print(f"随机森林分类器评估:")
print(f"准确率: {accuracy:.4f}")

# 6. 特征重要性
print("\n6. 特征重要性")

# 决策树的特征重要性
print("决策树特征重要性:")
for feature, importance in zip(iris.feature_names, dt_classifier.feature_importances_):
    print(f"{feature}: {importance:.4f}")

# 随机森林的特征重要性
print("\n随机森林特征重要性:")
for feature, importance in zip(iris.feature_names, rf_classifier.feature_importances_):
    print(f"{feature}: {importance:.4f}")

# 绘制特征重要性
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.bar(iris.feature_names, dt_classifier.feature_importances_)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Decision Tree Feature Importance')

plt.subplot(1, 2, 2)
plt.bar(iris.feature_names, rf_classifier.feature_importances_)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Random Forest Feature Importance')

plt.tight_layout()
plt.show()

# 7. 超参数调优
print("\n7. 超参数调优")

from sklearn.model_selection import GridSearchCV

# 定义参数网格
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 网格搜索
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy'
)

# 使用鸢尾花数据集进行网格搜索
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

grid_search.fit(X_train, y_train)

print(f"最佳参数:")
print(grid_search.best_params_)
print(f"最佳交叉验证准确率: {grid_search.best_score_:.4f}")

# 使用最佳参数训练模型
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"测试集准确率: {accuracy:.4f}")

# 8. 过拟合问题
print("\n8. 过拟合问题")

print("决策树容易过拟合，特别是当树深度较深时")
print("解决方法:")
print("- 设置最大深度")
print("- 设置最小样本分割数")
print("- 设置最小叶节点样本数")
print("- 剪枝")
print("- 使用随机森林")

# 9. 随机森林回归
print("\n9. 随机森林回归")

# 生成回归数据
np.random.seed(42)
x = np.linspace(0, 10, 100).reshape(-1, 1)
y = x**2 + np.random.randn(100, 1) * 5

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# 创建并训练随机森林回归器
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train.ravel())

# 预测
y_pred = rf_regressor.predict(X_test)

# 评估
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"随机森林回归器评估:")
print(f"均方误差 (MSE): {mse:.4f}")
print(f"R² 评分: {r2:.4f}")

# 可视化回归结果
plt.figure(figsize=(8, 6))
plt.scatter(X_train, y_train, label='训练数据', alpha=0.6)
plt.scatter(X_test, y_test, label='测试数据', alpha=0.6, color='green')

# 绘制预测曲线
x_line = np.linspace(0, 10, 1000).reshape(-1, 1)
y_pred_line = rf_regressor.predict(x_line)
plt.plot(x_line, y_pred_line, 'r-', label='随机森林回归')

plt.xlabel('X')
plt.ylabel('y')
plt.title('随机森林回归')
plt.legend()
plt.show()

# 10. 练习
print("\n10. 练习")

# 练习1: 不同深度的决策树
print("练习1: 不同深度的决策树")

# 生成二分类数据
X, y = make_classification(
    n_samples=1000, n_features=2, n_informative=2, 
    n_redundant=0, n_classes=2, random_state=42
)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 尝试不同的最大深度
depths = [1, 2, 3, 5, 10, None]

plt.figure(figsize=(12, 8))

for i, max_depth in enumerate(depths, 1):
    # 创建并训练决策树
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    dt.fit(X_train, y_train)
    
    # 预测
    y_pred = dt.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"最大深度 {max_depth}: 准确率 = {accuracy:.4f}")
    
    # 绘制决策边界
    plt.subplot(2, 3, i)
    
    # 绘制数据点
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr', alpha=0.6)
    
    # 绘制决策边界
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = dt.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
    
    plt.title(f'Max Depth: {max_depth}\nAccuracy: {accuracy:.4f}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()

# 练习2: 不同数量的树的随机森林
print("\n练习2: 不同数量的树的随机森林")

# 尝试不同的树数量
n_estimators = [10, 50, 100, 200, 500]

accuracies = []
for n in n_estimators:
    rf = RandomForestClassifier(n_estimators=n, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    print(f"树数量 {n}: 准确率 = {accuracy:.4f}")

# 绘制准确率曲线
plt.figure(figsize=(8, 6))
plt.plot(n_estimators, accuracies, marker='o')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.title('Effect of Number of Trees on Random Forest Accuracy')
plt.show()

print("\n=== 第36天学习示例结束 ===")
