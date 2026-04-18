#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第51天：机器学习基础复习
机器学习基础学习示例
内容：机器学习基础概念复习、常见算法回顾、最佳实践总结
"""

print("=== 第51天：机器学习基础复习 ===")

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report

# 1. 机器学习基础概念复习
print("\n1. 机器学习基础概念复习")

print("机器学习的基本概念:")
print("- 监督学习：有标签数据，包括分类和回归")
print("- 无监督学习：无标签数据，包括聚类和降维")
print("- 半监督学习：少量标签数据和大量无标签数据")
print("- 强化学习：通过与环境交互学习最佳策略")
print("- 过拟合：模型过度拟合训练数据，泛化能力差")
print("- 欠拟合：模型无法捕捉数据的基本模式")
print("- 偏差-方差权衡：平衡模型的偏差和方差")
print("- 交叉验证：评估模型性能的方法")
print("- 特征工程：提高模型性能的重要环节")

# 2. 常见算法回顾
print("\n2. 常见算法回顾")

# 2.1 线性模型
print("\n2.1 线性模型")

print("线性模型的特点:")
print("- 线性回归：用于回归任务，假设特征与目标之间存在线性关系")
print("- 逻辑回归：用于分类任务，使用sigmoid函数将输出映射到[0,1]")
print("- Ridge回归：L2正则化，防止过拟合")
print("- Lasso回归：L1正则化，产生稀疏解")

# 2.2 树模型
print("\n2.2 树模型")

print("树模型的特点:")
print("- 决策树：易于理解，可处理非线性关系")
print("- 随机森林：集成方法，降低过拟合风险")
print("- 梯度提升：迭代训练，提高模型性能")

# 2.3 支持向量机
print("\n2.3 支持向量机")

print("支持向量机的特点:")
print("- 分类和回归任务都适用")
print("- 核技巧：处理非线性问题")
print("- 边际最大化：提高模型泛化能力")

# 2.4 聚类算法
print("\n2.4 聚类算法")

print("聚类算法的特点:")
print("- K-means：基于距离的聚类方法")
print("- 层次聚类：构建聚类层次结构")
print("- DBSCAN：基于密度的聚类方法")

# 2.5 降维方法
print("\n2.5 降维方法")

print("降维方法的特点:")
print("- PCA：线性降维，保留数据的主要方差")
print("- t-SNE：非线性降维，适合数据可视化")
print("- LDA：监督式降维，考虑类别信息")

# 3. 机器学习工作流程复习
print("\n3. 机器学习工作流程复习")

print("机器学习的完整工作流程:")
print("1. 问题定义：明确任务目标和评估指标")
print("2. 数据收集：获取相关数据")
print("3. 数据探索：了解数据的基本情况")
print("4. 数据预处理：处理缺失值、异常值、特征编码等")
print("5. 特征工程：特征选择、提取、构建")
print("6. 模型选择：选择合适的算法")
print("7. 模型训练：使用训练数据训练模型")
print("8. 模型评估：评估模型性能")
print("9. 模型调优：调整超参数")
print("10. 模型部署：将模型应用到实际环境")

# 4. 实践练习：分类任务
print("\n4. 实践练习：分类任务")

# 加载iris数据集
iris = load_iris()
X, y = iris.data, iris.target
print(f"Iris数据集形状: {X.shape}, {y.shape}")

# 数据预处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
print(f"训练集形状: {X_train.shape}, {y_train.shape}")
print(f"测试集形状: {X_test.shape}, {y_test.shape}")

# 训练多个分类模型
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42),
    'KNN': KNeighborsClassifier()
}

print("\n分类模型性能比较:")
for name, clf in classifiers.items():
    # 交叉验证
    scores = cross_val_score(clf, X_scaled, y, cv=5, scoring='accuracy')
    # 训练和预测
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name}: 交叉验证准确率 = {scores.mean():.4f}, 测试准确率 = {accuracy:.4f}")

# 5. 实践练习：回归任务
print("\n5. 实践练习：回归任务")

# 加载糖尿病数据集
diabetes = load_diabetes()
X_reg, y_reg = diabetes.data, diabetes.target
print(f"糖尿病数据集形状: {X_reg.shape}, {y_reg.shape}")

# 数据预处理
scaler = StandardScaler()
X_reg_scaled = scaler.fit_transform(X_reg)

# 分割训练集和测试集
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg_scaled, y_reg, test_size=0.3, random_state=42)
print(f"训练集形状: {X_reg_train.shape}, {y_reg_train.shape}")
print(f"测试集形状: {X_reg_test.shape}, {y_reg_test.shape}")

# 训练多个回归模型
regressors = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(random_state=42),
    'Lasso Regression': Lasso(random_state=42),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}

print("\n回归模型性能比较:")
for name, reg in regressors.items():
    # 交叉验证
    scores = cross_val_score(reg, X_reg_scaled, y_reg, cv=5, scoring='r2')
    # 训练和预测
    reg.fit(X_reg_train, y_reg_train)
    y_reg_pred = reg.predict(X_reg_test)
    r2 = r2_score(y_reg_test, y_reg_pred)
    mse = mean_squared_error(y_reg_test, y_reg_pred)
    print(f"{name}: 交叉验证R² = {scores.mean():.4f}, 测试R² = {r2:.4f}, 测试MSE = {mse:.4f}")

# 6. 机器学习最佳实践
print("\n6. 机器学习最佳实践")

print("数据处理最佳实践:")
print("- 数据质量：确保数据的准确性和完整性")
print("- 特征工程：选择和构建有意义的特征")
print("- 数据标准化：使特征具有相同的尺度")
print("- 处理不平衡数据：使用采样或加权方法")

print("\n模型训练最佳实践:")
print("- 交叉验证：使用交叉验证评估模型性能")
print("- 超参数调优：使用网格搜索或随机搜索")
print("- 模型集成：结合多个模型的预测")
print("- 早停：防止过拟合")

print("\n模型评估最佳实践:")
print("- 选择合适的评估指标：根据任务类型选择")
print("- 考虑业务需求：模型性能与业务目标对齐")
print("- 模型解释性：理解模型的决策过程")
print("- 监控模型性能：定期评估模型在新数据上的表现")

# 7. 常见问题与解决方案
print("\n7. 常见问题与解决方案")

print("常见问题及解决方案:")
print("- 过拟合：使用正则化、增加数据、减少模型复杂度")
print("- 欠拟合：增加模型复杂度、添加特征、减少正则化")
print("- 数据不平衡：使用采样方法、代价敏感学习、集成方法")
print("- 特征选择：使用过滤法、包装法、嵌入法")
print("- 模型选择：根据数据特点和任务需求选择合适的算法")

# 8. 机器学习工具和库
print("\n8. 机器学习工具和库")

print("常用的机器学习工具和库:")
print("- scikit-learn：Python中最流行的机器学习库")
print("- TensorFlow：深度学习框架")
print("- PyTorch：深度学习框架")
print("- XGBoost：梯度提升库")
print("- LightGBM：梯度提升库，速度更快")
print("- CatBoost：梯度提升库，处理类别特征效果好")
print("- pandas：数据处理库")
print("- NumPy：数值计算库")
print("- Matplotlib/Seaborn：数据可视化库")

# 9. 机器学习资源推荐
print("\n9. 机器学习资源推荐")

print("推荐的机器学习资源:")
print("- 书籍：《机器学习实战》、《统计学习方法》、《深度学习》")
print("- 在线课程：Coursera上的机器学习课程、fast.ai课程")
print("- 网站：Kaggle、GitHub、Machine Learning Mastery")
print("- 论文：arXiv上的最新研究论文")
print("- 社区：Stack Overflow、Reddit的Machine Learning子版块")

# 10. 练习
print("\n10. 练习")

# 练习1: 模型比较
print("练习1: 模型比较")
print("- 选择一个数据集，比较不同算法的性能")
print("- 分析不同算法的优缺点")

# 练习2: 特征工程
print("\n练习2: 特征工程")
print("- 选择一个数据集，进行特征工程")
print("- 评估特征工程对模型性能的影响")

# 练习3: 模型调优
print("\n练习3: 模型调优")
print("- 选择一个模型，使用网格搜索进行调优")
print("- 分析不同超参数对模型性能的影响")

# 练习4: 集成学习
print("\n练习4: 集成学习")
print("- 实现一个集成学习模型")
print("- 比较集成模型与单一模型的性能")

# 练习5: 完整项目
print("\n练习5: 完整项目")
print("- 完成一个完整的机器学习项目")
print("- 包括数据探索、预处理、模型训练、评估和部署")

print("\n=== 第51天学习示例结束 ===")
