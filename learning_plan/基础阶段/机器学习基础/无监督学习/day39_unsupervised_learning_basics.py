#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第39天：无监督学习基础
机器学习基础学习示例
内容：无监督学习的基本概念、聚类和降维
"""

print("=== 第39天：无监督学习基础 ===")

# 1. 无监督学习基本概念
print("\n1. 无监督学习基本概念")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("无监督学习是一种机器学习方法，其中模型从无标记的训练数据中学习")
print("- 训练数据不包含标签信息")
print("- 模型学习数据的内在结构和模式")
print("- 主要应用：聚类、降维、异常检测")

# 2. 聚类问题
print("\n2. 聚类问题")

# 生成聚类数据
X, y = make_blobs(
    n_samples=300, n_features=2, centers=3, 
    cluster_std=1.0, random_state=42
)

print(f"特征形状: {X.shape}")
print(f"真实标签形状: {y.shape}")
print(f"前5个特征:")
print(X[:5])
print(f"前5个真实标签:")
print(y[:5])

# 可视化数据
print("\n可视化聚类数据:")
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.6)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Clustering Data')
plt.show()

# 3. K-means聚类
print("\n3. K-means聚类")

from sklearn.cluster import KMeans

# 创建并训练K-means模型
kmeans = KMeans(n_clusters=3, random_state=42)
y_pred = kmeans.fit_predict(X)

# 可视化聚类结果
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', alpha=0.6)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=200, c='red', marker='X', label='Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-means Clustering')
plt.legend()
plt.show()

# 4. 层次聚类
print("\n4. 层次聚类")

from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

# 可视化层次聚类树状图
plt.figure(figsize=(10, 6))
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Samples')
plt.ylabel('Euclidean distances')
plt.show()

# 创建并训练层次聚类模型
hierarchical = AgglomerativeClustering(n_clusters=3, linkage='ward')
y_pred_hierarchical = hierarchical.fit_predict(X)

# 可视化聚类结果
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_pred_hierarchical, cmap='viridis', alpha=0.6)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Hierarchical Clustering')
plt.show()

# 5. DBSCAN聚类
print("\n5. DBSCAN聚类")

from sklearn.cluster import DBSCAN

# 创建并训练DBSCAN模型
dbscan = DBSCAN(eps=0.5, min_samples=5)
y_pred_dbscan = dbscan.fit_predict(X)

# 可视化聚类结果
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_pred_dbscan, cmap='viridis', alpha=0.6)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('DBSCAN Clustering')
plt.show()

# 6. 降维
print("\n6. 降维")

from sklearn.decomposition import PCA

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target

print(f"原始特征维度: {X.shape[1]}")

# 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用PCA降维到2维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"降维后特征维度: {X_pca.shape[1]}")
print(f"前5个降维后的特征:")
print(X_pca[:5])

# 可视化降维结果
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Dimensionality Reduction')
plt.show()

# 7. t-SNE降维
print("\n7. t-SNE降维")

from sklearn.manifold import TSNE

# 使用t-SNE降维到2维
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# 可视化降维结果
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.6)
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('t-SNE Dimensionality Reduction')
plt.show()

# 8. 主成分分析（PCA）的解释方差
print("\n8. 主成分分析（PCA）的解释方差")

# 计算累积解释方差
pca = PCA()
pca.fit(X_scaled)
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

print("各主成分的解释方差:")
for i, variance in enumerate(pca.explained_variance_ratio_):
    print(f"PC{i+1}: {variance:.4f}")

print("\n累积解释方差:")
for i, cumulative_variance in enumerate(cumulative_variance_ratio):
    print(f"前{i+1}个主成分: {cumulative_variance:.4f}")

# 绘制累积解释方差曲线
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, 'o-')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Cumulative Explained Variance')
plt.legend()
plt.show()

# 9. 异常检测
print("\n9. 异常检测")

from sklearn.neighbors import LocalOutlierFactor

# 生成包含异常值的数据
np.random.seed(42)
X_inliers = 0.3 * np.random.randn(100, 2)
X_inliers = np.r_[X_inliers + 2, X_inliers - 2]
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
X = np.r_[X_inliers, X_outliers]
y = np.zeros(len(X), dtype=int)
y[-20:] = 1  # 最后20个是异常值

# 使用LOF进行异常检测
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
y_pred = lof.fit_predict(X)

# 可视化异常检测结果
plt.figure(figsize=(8, 6))
plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], c='blue', label='Inliers')
plt.scatter(X[y_pred == -1, 0], X[y_pred == -1, 1], c='red', label='Outliers')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Local Outlier Factor (LOF)')
plt.legend()
plt.show()

# 10. 练习
print("\n10. 练习")

# 练习1: 不同聚类算法的比较
print("练习1: 不同聚类算法的比较")

# 生成复杂数据
X, y = make_moons(n_samples=200, noise=0.05, random_state=42)

# 尝试不同的聚类算法
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

# K-means
kmeans = KMeans(n_clusters=2, random_state=42)
y_pred_kmeans = kmeans.fit_predict(X)

# 层次聚类
hierarchical = AgglomerativeClustering(n_clusters=2, linkage='ward')
y_pred_hierarchical = hierarchical.fit_predict(X)

# DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
y_pred_dbscan = dbscan.fit_predict(X)

# 可视化结果
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_pred_kmeans, cmap='viridis', alpha=0.6)
plt.title('K-means')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.subplot(1, 3, 2)
plt.scatter(X[:, 0], X[:, 1], c=y_pred_hierarchical, cmap='viridis', alpha=0.6)
plt.title('Hierarchical')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.subplot(1, 3, 3)
plt.scatter(X[:, 0], X[:, 1], c=y_pred_dbscan, cmap='viridis', alpha=0.6)
plt.title('DBSCAN')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()

# 练习2: 确定最佳聚类数
print("\n练习2: 确定最佳聚类数")

# 使用肘部法则
inertias = []
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# 绘制肘部曲线
plt.figure(figsize=(8, 6))
plt.plot(k_values, inertias, 'o-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

# 练习3: 不同降维方法的比较
print("\n练习3: 不同降维方法的比较")

from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE, Isomap

# 生成高维数据
np.random.seed(42)
X_high = np.random.randn(100, 10)  # 10维数据

# 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_high)

# PCA降维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Kernel PCA降维
kpca = KernelPCA(n_components=2, kernel='rbf')
X_kpca = kpca.fit_transform(X_scaled)

# t-SNE降维
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Isomap降维
isomap = Isomap(n_components=2)
X_isomap = isomap.fit_transform(X_scaled)

# 可视化结果
plt.figure(figsize=(15, 8))

plt.subplot(2, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
plt.title('PCA')
plt.xlabel('Component 1')
plt.ylabel('Component 2')

plt.subplot(2, 2, 2)
plt.scatter(X_kpca[:, 0], X_kpca[:, 1], alpha=0.6)
plt.title('Kernel PCA')
plt.xlabel('Component 1')
plt.ylabel('Component 2')

plt.subplot(2, 2, 3)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.6)
plt.title('t-SNE')
plt.xlabel('Component 1')
plt.ylabel('Component 2')

plt.subplot(2, 2, 4)
plt.scatter(X_isomap[:, 0], X_isomap[:, 1], alpha=0.6)
plt.title('Isomap')
plt.xlabel('Component 1')
plt.ylabel('Component 2')

plt.tight_layout()
plt.show()

print("\n=== 第39天学习示例结束 ===")
