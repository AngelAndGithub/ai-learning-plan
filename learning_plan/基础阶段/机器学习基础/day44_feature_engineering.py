#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第44天：特征工程
机器学习基础学习示例
内容：特征工程的基本概念、特征选择、特征提取、特征转换
"""

print("=== 第44天：特征工程 ===")

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, chi2, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 1. 特征工程概述
print("\n1. 特征工程概述")

print("特征工程是机器学习中非常重要的环节")
print("- 目的：提高模型的性能和泛化能力")
print("- 内容：特征选择、特征提取、特征转换、特征构建")
print("- 重要性：数据和特征决定了机器学习的上限，模型和算法只是逼近这个上限")

# 2. 数据预处理
print("\n2. 数据预处理")

# 加载iris数据集
iris = load_iris()
X, y = iris.data, iris.target
print(f"Iris数据集形状: {X.shape}, {y.shape}")

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"训练集形状: {X_train.shape}, {y_train.shape}")
print(f"测试集形状: {X_test.shape}, {y_test.shape}")

# 3. 特征缩放
print("\n3. 特征缩放")

print("特征缩放是特征工程的重要步骤")
print("- 目的：使不同特征的尺度一致，提高模型的收敛速度和性能")
print("- 方法：标准化、归一化")

# 3.1 标准化
print("\n3.1 标准化")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"标准化前训练集均值: {X_train.mean(axis=0)}")
print(f"标准化前训练集标准差: {X_train.std(axis=0)}")
print(f"标准化后训练集均值: {X_train_scaled.mean(axis=0)}")
print(f"标准化后训练集标准差: {X_train_scaled.std(axis=0)}")

# 3.2 归一化
print("\n3.2 归一化")
minmax_scaler = MinMaxScaler()
X_train_normalized = minmax_scaler.fit_transform(X_train)
X_test_normalized = minmax_scaler.transform(X_test)
print(f"归一化前训练集最小值: {X_train.min(axis=0)}")
print(f"归一化前训练集最大值: {X_train.max(axis=0)}")
print(f"归一化后训练集最小值: {X_train_normalized.min(axis=0)}")
print(f"归一化后训练集最大值: {X_train_normalized.max(axis=0)}")

# 4. 特征选择
print("\n4. 特征选择")

print("特征选择是选择对模型预测最有用的特征")
print("- 目的：减少特征维度，提高模型性能，减少过拟合")
print("- 方法：过滤法、包装法、嵌入法")

# 4.1 过滤法
print("\n4.1 过滤法")

# 使用SelectKBest选择最好的k个特征
selector = SelectKBest(f_classif, k=2)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
print(f"选择前特征数量: {X_train.shape[1]}")
print(f"选择后特征数量: {X_train_selected.shape[1]}")
print(f"选择的特征索引: {selector.get_support(indices=True)}")

# 4.2 包装法
print("\n4.2 包装法")

# 使用RFE（递归特征消除）选择特征
estimator = LogisticRegression(max_iter=1000, random_state=42)
rfe = RFE(estimator, n_features_to_select=2)
X_train_rfe = rfe.fit_transform(X_train, y_train)
X_test_rfe = rfe.transform(X_test)
print(f"RFE选择的特征索引: {rfe.get_support(indices=True)}")
print(f"特征排名: {rfe.ranking_}")

# 4.3 嵌入法
print("\n4.3 嵌入法")

# 使用决策树的特征重要性
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
importances = model.feature_importances_
print(f"特征重要性: {importances}")
print(f"特征重要性排序: {np.argsort(importances)[::-1]}")

# 5. 特征提取
print("\n5. 特征提取")

print("特征提取是从原始数据中提取有意义的特征")
print("- 目的：减少特征维度，提取更有意义的特征")
print("- 方法：主成分分析（PCA）、t-SNE")

# 5.1 主成分分析（PCA）
print("\n5.1 主成分分析（PCA）")
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
print(f"PCA前特征数量: {X_train.shape[1]}")
print(f"PCA后特征数量: {X_train_pca.shape[1]}")
print(f"解释方差比例: {pca.explained_variance_ratio_}")
print(f"累计解释方差比例: {np.cumsum(pca.explained_variance_ratio_)}")

# 可视化PCA结果
plt.figure(figsize=(10, 6))
for i in range(len(np.unique(y_train))):
    plt.scatter(X_train_pca[y_train == i, 0], X_train_pca[y_train == i, 1], label=iris.target_names[i])
plt.title('PCA结果可视化')
plt.xlabel('主成分1')
plt.ylabel('主成分2')
plt.legend()
plt.savefig('pca_visualization.png')
print("PCA结果已保存为 pca_visualization.png")

# 5.2 t-SNE
print("\n5.2 t-SNE")
tsne = TSNE(n_components=2, random_state=42)
X_train_tsne = tsne.fit_transform(X_train)

# 可视化t-SNE结果
plt.figure(figsize=(10, 6))
for i in range(len(np.unique(y_train))):
    plt.scatter(X_train_tsne[y_train == i, 0], X_train_tsne[y_train == i, 1], label=iris.target_names[i])
plt.title('t-SNE结果可视化')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.legend()
plt.savefig('tsne_visualization.png')
print("t-SNE结果已保存为 tsne_visualization.png")

# 6. 特征转换
print("\n6. 特征转换")

print("特征转换是将原始特征转换为更适合模型的形式")
print("- 目的：提高模型的性能和泛化能力")
print("- 方法：独热编码、标签编码、多项式特征")

# 6.1 独热编码
print("\n6.1 独热编码")

# 创建一个包含分类特征的数据集
data = pd.DataFrame({
    'color': ['red', 'blue', 'green', 'red', 'blue'],
    'size': ['S', 'M', 'L', 'M', 'S'],
    'price': [10, 20, 30, 25, 15]
})
print("原始数据:")
print(data)

# 独热编码
encoder = OneHotEncoder()
categorical_features = data[['color', 'size']]
encoded_features = encoder.fit_transform(categorical_features).toarray()
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['color', 'size']))
final_df = pd.concat([encoded_df, data['price']], axis=1)
print("\n独热编码后的数据:")
print(final_df)

# 6.2 标签编码
print("\n6.2 标签编码")

label_encoder = LabelEncoder()
data['color_encoded'] = label_encoder.fit_transform(data['color'])
data['size_encoded'] = label_encoder.fit_transform(data['size'])
print("标签编码后的数据:")
print(data)

# 6.3 多项式特征
print("\n6.3 多项式特征")

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_train[:, :2])  # 使用前两个特征
print(f"原始特征形状: {X_train[:, :2].shape}")
print(f"多项式特征形状: {X_poly.shape}")
print(f"多项式特征名称: {poly.get_feature_names_out(['feature1', 'feature2'])}")

# 7. 特征构建
print("\n7. 特征构建")

print("特征构建是根据领域知识创建新的特征")
print("- 目的：捕获数据中的非线性关系")
print("- 方法：基于业务知识的特征构建、交互特征")

# 示例：基于iris数据集构建新特征
iris_df = pd.DataFrame(X, columns=iris.feature_names)
iris_df['sepal_area'] = iris_df['sepal length (cm)'] * iris_df['sepal width (cm)']
iris_df['petal_area'] = iris_df['petal length (cm)'] * iris_df['petal width (cm)']
iris_df['sepal_to_petal_ratio'] = (iris_df['sepal length (cm)'] + iris_df['sepal width (cm)']) / (iris_df['petal length (cm)'] + iris_df['petal width (cm)'])
print("\n构建新特征后的数据:")
print(iris_df.head())

# 8. 特征工程的最佳实践
print("\n8. 特征工程的最佳实践")

print("特征工程的最佳实践:")
print("- 了解业务领域，基于业务知识构建特征")
print("- 对特征进行探索性分析，了解特征的分布")
print("- 处理缺失值和异常值")
print("- 对特征进行适当的缩放")
print("- 选择对模型预测最有用的特征")
print("- 尝试不同的特征提取方法")
print("- 不断迭代和优化特征工程过程")

# 9. 特征工程的评估
print("\n9. 特征工程的评估")

print("评估特征工程的效果")

# 评估不同特征处理方法的效果
from sklearn.metrics import accuracy_score

# 原始特征
model_original = LogisticRegression(max_iter=1000, random_state=42)
model_original.fit(X_train, y_train)
y_pred_original = model_original.predict(X_test)
accuracy_original = accuracy_score(y_test, y_pred_original)
print(f"原始特征的准确率: {accuracy_original:.4f}")

# 标准化特征
model_scaled = LogisticRegression(max_iter=1000, random_state=42)
model_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = model_scaled.predict(X_test_scaled)
accuracy_scaled = accuracy_score(y_test, y_pred_scaled)
print(f"标准化特征的准确率: {accuracy_scaled:.4f}")

# 选择后的特征
model_selected = LogisticRegression(max_iter=1000, random_state=42)
model_selected.fit(X_train_selected, y_train)
y_pred_selected = model_selected.predict(X_test_selected)
accuracy_selected = accuracy_score(y_test, y_pred_selected)
print(f"选择后特征的准确率: {accuracy_selected:.4f}")

# PCA特征
model_pca = LogisticRegression(max_iter=1000, random_state=42)
model_pca.fit(X_train_pca, y_train)
y_pred_pca = model_pca.predict(X_test_pca)
accuracy_pca = accuracy_score(y_test, y_pred_pca)
print(f"PCA特征的准确率: {accuracy_pca:.4f}")

# 10. 练习
print("\n10. 练习")

# 练习1: 处理缺失值
print("练习1: 处理缺失值")
print("- 创建一个包含缺失值的数据集")
print("- 使用不同的方法处理缺失值（均值填充、中位数填充、KNN填充）")
print("- 评估不同处理方法的效果")

# 练习2: 处理异常值
print("\n练习2: 处理异常值")
print("- 创建一个包含异常值的数据集")
print("- 使用不同的方法检测和处理异常值（IQR方法、Z-score方法）")
print("- 评估不同处理方法的效果")

# 练习3: 文本特征工程
print("\n练习3: 文本特征工程")
print("- 加载一个文本数据集")
print("- 使用不同的方法提取文本特征（TF-IDF、词袋模型）")
print("- 评估不同特征提取方法的效果")

# 练习4: 时间特征工程
print("\n练习4: 时间特征工程")
print("- 创建一个包含时间特征的数据集")
print("- 提取时间相关的特征（年、月、日、星期几、小时）")
print("- 评估时间特征对模型性能的影响")

# 练习5: 特征工程综合实践
print("\n练习5: 特征工程综合实践")
print("- 选择一个数据集，进行完整的特征工程")
print("- 包括特征预处理、特征选择、特征提取、特征构建")
print("- 评估特征工程后的模型性能")

print("\n=== 第44天学习示例结束 ===")
