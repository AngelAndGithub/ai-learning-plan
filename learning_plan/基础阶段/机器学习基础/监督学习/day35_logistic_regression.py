#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第35天：逻辑回归
机器学习基础学习示例
内容：逻辑回归的原理、多分类和模型评估
"""

print("=== 第35天：逻辑回归 ===")

# 1. 逻辑回归基本原理
print("\n1. 逻辑回归基本原理")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

print("逻辑回归是一种用于分类问题的线性模型")
print("- 使用sigmoid函数将线性组合映射到[0,1]区间")
print("- 预测概率大于0.5时为正类，否则为负类")

# 2. 二分类问题
print("\n2. 二分类问题")

# 生成二分类数据
X, y = make_classification(
    n_samples=1000, n_features=2, n_informative=2, 
    n_redundant=0, n_classes=2, random_state=42
)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 创建并训练模型
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# 评估
accuracy = accuracy_score(y_test, y_pred)
print(f"二分类逻辑回归评估:")
print(f"准确率: {accuracy:.4f}")

print("\n分类报告:")
print(classification_report(y_test, y_pred))

print("\n混淆矩阵:")
print(confusion_matrix(y_test, y_pred))

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
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')

# 绘制决策边界（概率为0.5的等高线）
Z_proba = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z_proba = Z_proba.reshape(xx.shape)
plt.contour(xx, yy, Z_proba, levels=[0.5], colors='black')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression Decision Boundary')
plt.show()

# 4. 多分类问题
print("\n4. 多分类问题")

from sklearn.datasets import load_iris

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target

print(f"特征数量: {X.shape[1]}")
print(f"样本数量: {X.shape[0]}")
print(f"类别数量: {len(np.unique(y))}")
print(f"类别名称: {iris.target_names}")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 创建并训练模型（使用ovr策略）
model = LogisticRegression(multi_class='ovr', random_state=42)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估
accuracy = accuracy_score(y_test, y_pred)
print(f"多分类逻辑回归评估:")
print(f"准确率: {accuracy:.4f}")

print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

print("\n混淆矩阵:")
print(confusion_matrix(y_test, y_pred))

# 5. 正则化
print("\n5. 正则化")

from sklearn.linear_model import LogisticRegressionCV

# 使用带交叉验证的逻辑回归
model_cv = LogisticRegressionCV(
    Cs=[0.001, 0.01, 0.1, 1, 10, 100],
    cv=5,
    penalty='l2',
    random_state=42
)
model_cv.fit(X_train, y_train)

# 预测
y_pred_cv = model_cv.predict(X_test)

# 评估
accuracy_cv = accuracy_score(y_test, y_pred_cv)
print(f"带交叉验证的逻辑回归评估:")
print(f"准确率: {accuracy_cv:.4f}")
print(f"最佳C值: {model_cv.C_[0]:.4f}")

# 6. 模型评估指标
print("\n6. 模型评估指标")

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve

# 二分类评估指标
X_binary, y_binary = make_classification(
    n_samples=1000, n_features=2, n_informative=2, 
    n_redundant=0, n_classes=2, random_state=42
)

X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
    X_binary, y_binary, test_size=0.2, random_state=42
)

model_bin = LogisticRegression(random_state=42)
model_bin.fit(X_train_bin, y_train_bin)

# 预测
y_pred_bin = model_bin.predict(X_test_bin)
y_pred_proba_bin = model_bin.predict_proba(X_test_bin)[:, 1]

# 计算评估指标
precision = precision_score(y_test_bin, y_pred_bin)
recall = recall_score(y_test_bin, y_pred_bin)
f1 = f1_score(y_test_bin, y_pred_bin)
roc_auc = roc_auc_score(y_test_bin, y_pred_proba_bin)

print(f"二分类评估指标:")
print(f"精确率: {precision:.4f}")
print(f"召回率: {recall:.4f}")
print(f"F1分数: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

# 绘制ROC曲线
fpr, tpr, thresholds = roc_curve(y_test_bin, y_pred_proba_bin)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# 7. 特征重要性
print("\n7. 特征重要性")

# 查看逻辑回归的系数
print("逻辑回归系数:")
for feature, coef in zip(iris.feature_names, model.coef_[0]):
    print(f"{feature}: {coef:.4f}")

# 绘制特征重要性
plt.figure(figsize=(8, 6))
plt.bar(iris.feature_names, np.abs(model.coef_[0]))
plt.xlabel('Feature')
plt.ylabel('Absolute Coefficient')
plt.title('Feature Importance in Logistic Regression')
plt.show()

# 8. 超参数调优
print("\n8. 超参数调优")

from sklearn.model_selection import GridSearchCV

# 定义参数网格
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']  # 用于l1正则化
}

# 网格搜索
grid_search = GridSearchCV(
    LogisticRegression(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy'
)

# 注意：使用二分类数据进行网格搜索
grid_search.fit(X_train_bin, y_train_bin)

print(f"最佳参数:")
print(grid_search.best_params_)
print(f"最佳交叉验证准确率: {grid_search.best_score_:.4f}")

# 9. 处理不平衡数据
print("\n9. 处理不平衡数据")

from sklearn.datasets import make_classification

# 生成不平衡数据
X_imbalanced, y_imbalanced = make_classification(
    n_samples=1000, n_features=2, n_informative=2, 
    n_redundant=0, n_classes=2, weights=[0.9, 0.1], random_state=42
)

print(f"类别分布:")
print(f"类别0: {np.sum(y_imbalanced == 0)}")
print(f"类别1: {np.sum(y_imbalanced == 1)}")
print(f"不平衡比例: {np.sum(y_imbalanced == 0) / np.sum(y_imbalanced == 1):.2f}:1")

# 划分训练集和测试集
X_train_imb, X_test_imb, y_train_imb, y_test_imb = train_test_split(
    X_imbalanced, y_imbalanced, test_size=0.2, random_state=42
)

# 不使用权重的模型
model_no_weight = LogisticRegression(random_state=42)
model_no_weight.fit(X_train_imb, y_train_imb)
y_pred_no_weight = model_no_weight.predict(X_test_imb)

# 使用权重的模型
model_with_weight = LogisticRegression(
    class_weight='balanced', random_state=42
)
model_with_weight.fit(X_train_imb, y_train_imb)
y_pred_with_weight = model_with_weight.predict(X_test_imb)

print("\n不使用权重的模型:")
print(classification_report(y_test_imb, y_pred_no_weight))

print("\n使用权重的模型:")
print(classification_report(y_test_imb, y_pred_with_weight))

# 10. 练习
print("\n10. 练习")

# 练习1: 使用不同的求解器
print("练习1: 使用不同的求解器")

# 尝试不同的求解器
solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']

for solver in solvers:
    try:
        model = LogisticRegression(solver=solver, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"求解器 {solver}: 准确率 = {accuracy:.4f}")
    except Exception as e:
        print(f"求解器 {solver}: 错误 - {e}")

# 练习2: 不同正则化强度的影响
print("\n练习2: 不同正则化强度的影响")

# 尝试不同的C值
C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

accuracies = []
for C in C_values:
    model = LogisticRegression(C=C, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    print(f"C={C}: 准确率 = {accuracy:.4f}")

# 绘制准确率曲线
plt.figure(figsize=(8, 6))
plt.plot(C_values, accuracies, marker='o')
plt.xscale('log')
plt.xlabel('C (正则化强度的倒数)')
plt.ylabel('Accuracy')
plt.title('Effect of Regularization Strength on Accuracy')
plt.show()

print("\n=== 第35天学习示例结束 ===")
