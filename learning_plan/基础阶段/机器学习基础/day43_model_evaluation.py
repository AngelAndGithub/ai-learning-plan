#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第43天：模型评估
机器学习基础学习示例
内容：模型评估的基本概念、评估指标、交叉验证
"""

print("=== 第43天：模型评估 ===")

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.datasets import load_iris, load_wine, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# 1. 模型评估概述
print("\n1. 模型评估概述")

print("模型评估是机器学习的重要环节，用于评估模型的性能")
print("- 目的：了解模型在新数据上的表现")
print("- 方法：训练集和测试集分离、交叉验证")
print("- 指标：准确率、精确率、召回率、F1分数等")

# 2. 数据集准备
print("\n2. 数据集准备")

# 加载iris数据集
iris = load_iris()
X, y = iris.data, iris.target
print(f"Iris数据集形状: {X.shape}, {y.shape}")

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"训练集形状: {X_train.shape}, {y_train.shape}")
print(f"测试集形状: {X_test.shape}, {y_test.shape}")

# 3. 分类模型评估指标
print("\n3. 分类模型评估指标")

# 训练模型
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 3.1 准确率（Accuracy）
print("\n3.1 准确率（Accuracy）")
accuracy = accuracy_score(y_test, y_pred)
print(f"准确率: {accuracy:.4f}")

# 3.2 精确率（Precision）
print("\n3.2 精确率（Precision）")
precision = precision_score(y_test, y_pred, average='weighted')
print(f"精确率: {precision:.4f}")

# 3.3 召回率（Recall）
print("\n3.3 召回率（Recall）")
recall = recall_score(y_test, y_pred, average='weighted')
print(f"召回率: {recall:.4f}")

# 3.4 F1分数（F1-Score）
print("\n3.4 F1分数（F1-Score）")
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"F1分数: {f1:.4f}")

# 3.5 混淆矩阵（Confusion Matrix）
print("\n3.5 混淆矩阵（Confusion Matrix）")
cm = confusion_matrix(y_test, y_pred)
print("混淆矩阵:")
print(cm)

# 3.6 分类报告（Classification Report）
print("\n3.6 分类报告（Classification Report）")
report = classification_report(y_test, y_pred, target_names=iris.target_names)
print("分类报告:")
print(report)

# 4. 回归模型评估指标
print("\n4. 回归模型评估指标")

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 加载糖尿病数据集
diabetes = load_diabetes()
X_reg, y_reg = diabetes.data, diabetes.target
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

# 训练线性回归模型
reg_model = LinearRegression()
reg_model.fit(X_reg_train, y_reg_train)
y_reg_pred = reg_model.predict(X_reg_test)

# 4.1 均方误差（MSE）
print("\n4.1 均方误差（MSE）")
mse = mean_squared_error(y_reg_test, y_reg_pred)
print(f"均方误差: {mse:.4f}")

# 4.2 均方根误差（RMSE）
print("\n4.2 均方根误差（RMSE）")
rmse = np.sqrt(mse)
print(f"均方根误差: {rmse:.4f}")

# 4.3 平均绝对误差（MAE）
print("\n4.3 平均绝对误差（MAE）")
mae = mean_absolute_error(y_reg_test, y_reg_pred)
print(f"平均绝对误差: {mae:.4f}")

# 4.4 R²分数
print("\n4.4 R²分数")
r2 = r2_score(y_reg_test, y_reg_pred)
print(f"R²分数: {r2:.4f}")

# 5. 交叉验证
print("\n5. 交叉验证")

print("交叉验证是一种更可靠的模型评估方法")
print("- 目的：减少过拟合，提高模型评估的可靠性")
print("- 方法：K折交叉验证、分层K折交叉验证")

# 5.1 K折交叉验证
print("\n5.1 K折交叉验证")
kfold = KFold(n_splits=5, random_state=42, shuffle=True)
scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
print(f"K折交叉验证准确率: {scores}")
print(f"平均准确率: {scores.mean():.4f}")
print(f"标准差: {scores.std():.4f}")

# 5.2 分层K折交叉验证
print("\n5.2 分层K折交叉验证")
stratified_kfold = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
stratified_scores = cross_val_score(model, X, y, cv=stratified_kfold, scoring='accuracy')
print(f"分层K折交叉验证准确率: {stratified_scores}")
print(f"平均准确率: {stratified_scores.mean():.4f}")
print(f"标准差: {stratified_scores.std():.4f}")

# 6. 模型比较
print("\n6. 模型比较")

# 比较不同模型的性能
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'SVM': SVC(random_state=42)
}

print("不同模型的交叉验证性能:")
for model_name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"{model_name}: 平均准确率 = {scores.mean():.4f}, 标准差 = {scores.std():.4f}")

# 7. 过拟合和欠拟合
print("\n7. 过拟合和欠拟合")

print("过拟合和欠拟合是模型评估中的常见问题")
print("- 过拟合：模型在训练集上表现良好，但在测试集上表现差")
print("- 欠拟合：模型在训练集和测试集上表现都差")

# 生成合成数据集
X_synth, y_synth = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_synth_train, X_synth_test, y_synth_train, y_synth_test = train_test_split(X_synth, y_synth, test_size=0.3, random_state=42)

# 测试不同深度的决策树
print("\n测试不同深度的决策树:")
depths = [1, 2, 5, 10, 20]
for depth in depths:
    tree_model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree_model.fit(X_synth_train, y_synth_train)
    train_score = tree_model.score(X_synth_train, y_synth_train)
    test_score = tree_model.score(X_synth_test, y_synth_test)
    print(f"深度={depth}: 训练准确率 = {train_score:.4f}, 测试准确率 = {test_score:.4f}")

# 8. 验证曲线和学习曲线
print("\n8. 验证曲线和学习曲线")

from sklearn.model_selection import validation_curve, learning_curve
import matplotlib.pyplot as plt

# 验证曲线
def plot_validation_curve():
    """绘制验证曲线"""
    param_range = [1, 2, 5, 10, 20]
    train_scores, test_scores = validation_curve(
        DecisionTreeClassifier(random_state=42),
        X_synth, y_synth,
        param_name="max_depth",
        param_range=param_range,
        cv=5,
        scoring="accuracy"
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.title("Validation Curve for Decision Tree")
    plt.xlabel("Max Depth")
    plt.ylabel("Accuracy")
    plt.ylim(0.7, 1.0)
    plt.plot(param_range, train_mean, 'o-', label="Training score")
    plt.plot(param_range, test_mean, 'o-', label="Cross-validation score")
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.1)
    plt.legend(loc="best")
    plt.savefig("validation_curve.png")
    print("验证曲线已保存为 validation_curve.png")

# 学习曲线
def plot_learning_curve():
    """绘制学习曲线"""
    train_sizes, train_scores, test_scores = learning_curve(
        DecisionTreeClassifier(max_depth=5, random_state=42),
        X_synth, y_synth,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5,
        scoring="accuracy"
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.title("Learning Curve for Decision Tree")
    plt.xlabel("Training Examples")
    plt.ylabel("Accuracy")
    plt.ylim(0.7, 1.0)
    plt.plot(train_sizes, train_mean, 'o-', label="Training score")
    plt.plot(train_sizes, test_mean, 'o-', label="Cross-validation score")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
    plt.legend(loc="best")
    plt.savefig("learning_curve.png")
    print("学习曲线已保存为 learning_curve.png")

# 绘制验证曲线和学习曲线
plot_validation_curve()
plot_learning_curve()

# 9. 模型评估的最佳实践
print("\n9. 模型评估的最佳实践")

print("模型评估的最佳实践:")
print("- 使用分层抽样确保训练集和测试集的类别分布一致")
print("- 使用交叉验证而不是单一的训练-测试分割")
print("- 选择合适的评估指标，考虑业务需求")
print("- 比较多个模型，选择性能最好的")
print("- 分析模型的错误，了解模型的弱点")
print("- 使用验证曲线和学习曲线分析模型的性能")

# 10. 练习
print("\n10. 练习")

# 练习1: 实现自定义评估指标
print("练习1: 实现自定义评估指标")
print("- 实现一个自定义的评估指标，如平衡准确率")
print("- 测试自定义评估指标的性能")

# 练习2: 模型评估综合实践
print("\n练习2: 模型评估综合实践")
print("- 选择一个数据集，训练多个模型")
print("- 使用交叉验证评估模型性能")
print("- 绘制验证曲线和学习曲线")
print("- 选择最佳模型并分析其性能")

# 练习3: 不平衡数据集的评估
print("\n练习3: 不平衡数据集的评估")
print("- 生成一个不平衡的分类数据集")
print("- 测试不同评估指标的表现")
print("- 尝试使用不同的采样方法处理不平衡数据")

# 练习4: 超参数调优
print("\n练习4: 超参数调优")
print("- 使用网格搜索或随机搜索调优模型超参数")
print("- 评估调优后的模型性能")

# 练习5: 模型解释
print("\n练习5: 模型解释")
print("- 使用SHAP或LIME解释模型的预测")
print("- 分析模型的特征重要性")

print("\n=== 第43天学习示例结束 ===")
