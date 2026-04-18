#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第45天：模型选择
机器学习基础学习示例
内容：模型选择的基本概念、网格搜索、随机搜索、贝叶斯优化
"""

print("=== 第45天：模型选择 ===")

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from scipy.stats import uniform, randint
import time

# 1. 模型选择概述
print("\n1. 模型选择概述")

print("模型选择是机器学习中的重要环节")
print("- 目的：选择最适合特定问题的模型和超参数")
print("- 内容：算法选择、超参数调优")
print("- 方法：网格搜索、随机搜索、贝叶斯优化")

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

# 3. 算法选择
print("\n3. 算法选择")

print("选择合适的算法是模型选择的第一步")

# 测试不同算法的性能
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'SVM': SVC(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

print("不同算法的交叉验证性能:")
for model_name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"{model_name}: 平均准确率 = {scores.mean():.4f}, 标准差 = {scores.std():.4f}")

# 4. 超参数调优
print("\n4. 超参数调优")

print("超参数调优是模型选择的重要环节")
print("- 超参数：在模型训练前设置的参数，如学习率、正则化强度等")
print("- 调优方法：网格搜索、随机搜索、贝叶斯优化")

# 4.1 网格搜索
print("\n4.1 网格搜索")

print("网格搜索是一种暴力搜索方法，尝试所有可能的超参数组合")

# 定义超参数网格
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

# 创建网格搜索对象
grid_search = GridSearchCV(
    LogisticRegression(max_iter=1000, random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# 开始搜索
start_time = time.time()
grid_search.fit(X_train, y_train)
end_time = time.time()

print(f"网格搜索耗时: {end_time - start_time:.2f} 秒")
print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")

# 使用最佳参数进行预测
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"测试集准确率: {accuracy:.4f}")

# 4.2 随机搜索
print("\n4.2 随机搜索")

print("随机搜索是一种随机采样超参数组合的方法，比网格搜索更高效")

# 定义超参数分布
param_dist = {
    'C': uniform(0.001, 100),
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

# 创建随机搜索对象
random_search = RandomizedSearchCV(
    LogisticRegression(max_iter=1000, random_state=42),
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

# 开始搜索
start_time = time.time()
random_search.fit(X_train, y_train)
end_time = time.time()

print(f"随机搜索耗时: {end_time - start_time:.2f} 秒")
print(f"最佳参数: {random_search.best_params_}")
print(f"最佳交叉验证分数: {random_search.best_score_:.4f}")

# 使用最佳参数进行预测
best_model_random = random_search.best_estimator_
y_pred_random = best_model_random.predict(X_test)
accuracy_random = accuracy_score(y_test, y_pred_random)
print(f"测试集准确率: {accuracy_random:.4f}")

# 4.3 贝叶斯优化
print("\n4.3 贝叶斯优化")

print("贝叶斯优化是一种基于概率模型的超参数调优方法，比随机搜索更高效")

# 安装bayesian-optimization库（如果需要）
try:
    from bayes_opt import BayesianOptimization
except ImportError:
    print("请安装bayesian-optimization库: pip install bayesian-optimization")
else:
    # 定义目标函数
def target_function(C, penalty):
    penalty = 'l1' if penalty < 0.5 else 'l2'
    model = LogisticRegression(
        C=C,
        penalty=penalty,
        solver='liblinear',
        max_iter=1000,
        random_state=42
    )
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    return scores.mean()

# 创建贝叶斯优化对象
optimizer = BayesianOptimization(
    f=target_function,
    pbounds={
        'C': (0.001, 100),
        'penalty': (0, 1)
    },
    random_state=42
)

# 开始优化
start_time = time.time()
optimizer.maximize(init_points=5, n_iter=15)
end_time = time.time()

print(f"贝叶斯优化耗时: {end_time - start_time:.2f} 秒")
print(f"最佳参数: {optimizer.max['params']}")
print(f"最佳交叉验证分数: {optimizer.max['target']:.4f}")

# 5. 模型选择的最佳实践
print("\n5. 模型选择的最佳实践")

print("模型选择的最佳实践:")
print("- 从简单模型开始，逐步尝试复杂模型")
print("- 使用交叉验证评估模型性能")
print("- 结合领域知识选择合适的算法")
print("- 考虑模型的计算成本和可解释性")
print("- 对于大规模数据集，使用随机搜索或贝叶斯优化")
print("- 记录实验结果，便于比较不同模型")

# 6. 模型选择的评估
print("\n6. 模型选择的评估")

print("评估模型选择的效果")

# 比较不同调优方法的性能
print("不同调优方法的性能比较:")
print(f"网格搜索: 最佳分数 = {grid_search.best_score_:.4f}, 测试准确率 = {accuracy:.4f}")
print(f"随机搜索: 最佳分数 = {random_search.best_score_:.4f}, 测试准确率 = {accuracy_random:.4f}")
if 'optimizer' in locals():
    print(f"贝叶斯优化: 最佳分数 = {optimizer.max['target']:.4f}")

# 7. 回归模型的超参数调优
print("\n7. 回归模型的超参数调优")

from sklearn.datasets import load_diabetes

# 加载糖尿病数据集
diabetes = load_diabetes()
X_reg, y_reg = diabetes.data, diabetes.target
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

# 定义Ridge回归的超参数网格
param_grid_ridge = {
    'alpha': [0.001, 0.01, 0.1, 1, 10, 100]
}

# 创建网格搜索对象
grid_search_ridge = GridSearchCV(
    Ridge(random_state=42),
    param_grid=param_grid_ridge,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

# 开始搜索
grid_search_ridge.fit(X_reg_train, y_reg_train)

print(f"Ridge回归最佳参数: {grid_search_ridge.best_params_}")
print(f"最佳交叉验证分数: {-grid_search_ridge.best_score_:.4f}")

# 使用最佳参数进行预测
best_ridge = grid_search_ridge.best_estimator_
y_reg_pred = best_ridge.predict(X_reg_test)
mse = mean_squared_error(y_reg_test, y_reg_pred)
print(f"测试集均方误差: {mse:.4f}")

# 8. 集成模型的超参数调优
print("\n8. 集成模型的超参数调优")

# 定义随机森林的超参数分布
param_dist_rf = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(3, 10),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 5)
}

# 创建随机搜索对象
random_search_rf = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist_rf,
    n_iter=20,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

# 开始搜索
random_search_rf.fit(X_train, y_train)

print(f"随机森林最佳参数: {random_search_rf.best_params_}")
print(f"最佳交叉验证分数: {random_search_rf.best_score_:.4f}")

# 使用最佳参数进行预测
best_rf = random_search_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"测试集准确率: {accuracy_rf:.4f}")

# 9. 模型选择的自动化
print("\n9. 模型选择的自动化")

print("使用自动化工具进行模型选择")

# 安装auto-sklearn库（如果需要）
try:
    from autosklearn.classification import AutoSklearnClassifier
except ImportError:
    print("请安装auto-sklearn库: pip install auto-sklearn")
else:
    # 创建AutoSklearn分类器
    automl = AutoSklearnClassifier(
        time_left_for_this_task=300,
        per_run_time_limit=30,
        n_jobs=-1,
        random_state=42
    )

    # 开始拟合
    start_time = time.time()
    automl.fit(X_train, y_train)
    end_time = time.time()

    print(f"AutoSklearn拟合耗时: {end_time - start_time:.2f} 秒")
    print(f"AutoSklearn最佳模型: {automl.show_models()}")

    # 使用最佳模型进行预测
    y_pred_automl = automl.predict(X_test)
    accuracy_automl = accuracy_score(y_test, y_pred_automl)
    print(f"测试集准确率: {accuracy_automl:.4f}")

# 10. 练习
print("\n10. 练习")

# 练习1: 实现自定义超参数调优
print("练习1: 实现自定义超参数调优")
print("- 实现一个自定义的超参数调优函数")
print("- 测试不同调优策略的效果")

# 练习2: 模型选择综合实践
print("\n练习2: 模型选择综合实践")
print("- 选择一个数据集，进行完整的模型选择")
print("- 包括算法选择和超参数调优")
print("- 评估不同模型的性能")

# 练习3: 模型选择的时间复杂度分析
print("\n练习3: 模型选择的时间复杂度分析")
print("- 分析不同超参数调优方法的时间复杂度")
print("- 测试不同规模数据集的调优时间")

# 练习4: 模型选择的可视化
print("\n练习4: 模型选择的可视化")
print("- 可视化不同超参数对模型性能的影响")
print("- 绘制超参数调优的学习曲线")

# 练习5: 模型选择的交叉验证策略
print("\n练习5: 模型选择的交叉验证策略")
print("- 比较不同交叉验证策略的效果")
print("- 测试分层交叉验证和留一交叉验证")

print("\n=== 第45天学习示例结束 ===")
