#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第47天：集成方法
机器学习基础学习示例
内容：集成方法的基本概念、Bagging、Boosting、Stacking
"""

print("=== 第47天：集成方法 ===")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_classification, load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, 
                              AdaBoostClassifier, GradientBoostingClassifier, 
                              BaggingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# 1. 集成方法概述
print("\n1. 集成方法概述")

print("集成方法是一种通过组合多个基学习器来提高模型性能的方法")
print("- 目的：减少方差、偏差，提高模型的泛化能力")
print("- 类型：Bagging、Boosting、Stacking")
print("- 优点：通常比单个模型表现更好，更稳定")

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

# 3. Bagging方法
print("\n3. Bagging方法")

print("Bagging（Bootstrap Aggregating）是一种通过自助采样构建多个基学习器的方法")
print("- 步骤：1. 自助采样 2. 训练基学习器 3. 投票或平均")
print("- 代表算法：随机森林、Extra Trees")

# 3.1 随机森林
print("\n3.1 随机森林")

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"随机森林准确率: {accuracy_rf:.4f}")

# 特征重要性
feature_importances = rf_model.feature_importances_
print(f"特征重要性: {feature_importances}")
print(f"特征重要性排序: {np.argsort(feature_importances)[::-1]}")

# 3.2 Extra Trees
print("\n3.2 Extra Trees")

extra_trees_model = ExtraTreesClassifier(n_estimators=100, random_state=42)
extra_trees_model.fit(X_train, y_train)
y_pred_extra = extra_trees_model.predict(X_test)
accuracy_extra = accuracy_score(y_test, y_pred_extra)
print(f"Extra Trees准确率: {accuracy_extra:.4f}")

# 3.3 BaggingClassifier
print("\n3.3 BaggingClassifier")

bagging_model = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=100,
    random_state=42
)
bagging_model.fit(X_train, y_train)
y_pred_bagging = bagging_model.predict(X_test)
accuracy_bagging = accuracy_score(y_test, y_pred_bagging)
print(f"BaggingClassifier准确率: {accuracy_bagging:.4f}")

# 4. Boosting方法
print("\n4. Boosting方法")

print("Boosting是一种通过迭代训练基学习器，逐步提高模型性能的方法")
print("- 步骤：1. 训练基学习器 2. 调整样本权重 3. 组合基学习器")
print("- 代表算法：AdaBoost、Gradient Boosting")

# 4.1 AdaBoost
print("\n4.1 AdaBoost")

adaboost_model = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100,
    random_state=42
)
adaboost_model.fit(X_train, y_train)
y_pred_adaboost = adaboost_model.predict(X_test)
accuracy_adaboost = accuracy_score(y_test, y_pred_adaboost)
print(f"AdaBoost准确率: {accuracy_adaboost:.4f}")

# 4.2 Gradient Boosting
print("\n4.2 Gradient Boosting")

gradient_boosting_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gradient_boosting_model.fit(X_train, y_train)
y_pred_gb = gradient_boosting_model.predict(X_test)
accuracy_gb = accuracy_score(y_test, y_pred_gb)
print(f"Gradient Boosting准确率: {accuracy_gb:.4f}")

# 5. Stacking方法
print("\n5. Stacking方法")

print("Stacking是一种通过组合多个基学习器的预测结果来构建元学习器的方法")
print("- 步骤：1. 训练多个基学习器 2. 使用基学习器的预测结果训练元学习器")
print("- 优点：可以结合不同类型模型的优势")

# 定义基学习器
base_learners = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('svc', SVC(probability=True, random_state=42)),
    ('knn', KNeighborsClassifier())
]

# 定义元学习器
meta_learner = LogisticRegression()

# 创建VotingClassifier（软投票）
voting_model = VotingClassifier(
    estimators=base_learners,
    voting='soft'
)

voting_model.fit(X_train, y_train)
y_pred_voting = voting_model.predict(X_test)
accuracy_voting = accuracy_score(y_test, y_pred_voting)
print(f"VotingClassifier准确率: {accuracy_voting:.4f}")

# 6. 集成方法的性能比较
print("\n6. 集成方法的性能比较")

# 定义所有模型
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=42),
    'Bagging': BaggingClassifier(
        base_estimator=DecisionTreeClassifier(),
        n_estimators=100,
        random_state=42
    ),
    'AdaBoost': AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=100,
        random_state=42
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    ),
    'Voting': voting_model
}

# 评估所有模型
print("不同集成方法的性能比较:")
for model_name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"{model_name}: 平均准确率 = {scores.mean():.4f}, 标准差 = {scores.std():.4f}")

# 7. 集成方法的调优
print("\n7. 集成方法的调优")

print("集成方法的调优参数:")
print("- n_estimators: 基学习器的数量")
print("- max_depth: 决策树的最大深度")
print("- learning_rate: 学习率（Boosting方法）")
print("- max_features: 随机森林中每次分裂考虑的特征数量")

# 8. 集成方法的应用场景
print("\n8. 集成方法的应用场景")

print("不同集成方法的应用场景:")
print("- 随机森林：处理高维数据，特征重要性分析")
print("- AdaBoost：处理二分类问题，对噪声敏感")
print("- Gradient Boosting：处理回归和分类问题，性能优异")
print("- Stacking：结合不同模型的优势，提高预测性能")

# 9. 集成方法的可视化
print("\n9. 集成方法的可视化")

# 生成合成数据集
X_synth, y_synth = make_classification(
    n_samples=1000, n_features=20, n_classes=2, 
    n_informative=15, n_redundant=5, random_state=42
)
X_synth_train, X_synth_test, y_synth_train, y_synth_test = train_test_split(
    X_synth, y_synth, test_size=0.3, random_state=42
)

# 测试不同集成方法的性能
synth_models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Voting': VotingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),
            ('svc', SVC(probability=True, random_state=42))
        ],
        voting='soft'
    )
}

# 评估合成数据集上的性能
print("合成数据集上的性能比较:")
for model_name, model in synth_models.items():
    model.fit(X_synth_train, y_synth_train)
    y_synth_pred = model.predict(X_synth_test)
    accuracy = accuracy_score(y_synth_test, y_synth_pred)
    print(f"{model_name}: 准确率 = {accuracy:.4f}")

# 10. 集成方法的优缺点
print("\n10. 集成方法的优缺点")

print("集成方法的优点:")
print("- 提高模型的预测性能")
print("- 减少过拟合的风险")
print("- 增强模型的稳定性")
print("- 可以处理不同类型的数据")

print("\n集成方法的缺点:")
print("- 计算成本高")
print("- 模型解释性差")
print("- 训练时间长")
print("- 调优参数复杂")

# 11. 练习
print("\n11. 练习")

# 练习1: 实现自定义Bagging
print("练习1: 实现自定义Bagging")
print("- 实现一个简单的Bagging算法")
print("- 测试自定义Bagging的性能")

# 练习2: 集成方法的参数调优
print("\n练习2: 集成方法的参数调优")
print("- 使用网格搜索对随机森林进行参数调优")
print("- 分析不同参数对模型性能的影响")

# 练习3: 集成方法的特征重要性
print("\n练习3: 集成方法的特征重要性")
print("- 分析随机森林和梯度提升的特征重要性")
print("- 可视化特征重要性")

# 练习4: 堆叠集成
print("\n练习4: 堆叠集成")
print("- 实现一个简单的堆叠集成模型")
print("- 测试堆叠集成的性能")

# 练习5: 集成方法的时间复杂度分析
print("\n练习5: 集成方法的时间复杂度分析")
print("- 分析不同集成方法的时间复杂度")
print("- 测试不同规模数据集的训练时间")

print("\n=== 第47天学习示例结束 ===")
