#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第40天：聚类算法
机器学习基础学习示例
内容：聚类算法的深入理解、评估和应用
"""

print("=== 第40天：聚类算法 ===")

# 1. 聚类算法概述
print("\n1. 聚类算法概述")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler

print("聚类是一种无监督学习方法，用于将相似的数据点分组到同一簇中")
print("常用的聚类算法:")
print("1. K-means: 基于距离的聚类算法")
print("2. 层次聚类: 基于层次结构的聚类算法")
print("3. DBSCAN: 基于密度的聚类算法")
print("4. Gaussian Mixture Models: 基于概率模型的聚类算法")

# 2. K-means聚类深入
print("\n2. K-means聚类深入")

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 生成聚类数据
X, y = make_blobs(
    n_samples=300, n_features=2, centers=4, 
    cluster_std=0.6, random_state=42
)

# 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 肘部法则确定最佳k值
inertias = []
silhouette_scores = []
k_values = range(2, 10)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# 绘制肘部曲线
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(k_values, inertias, 'o-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')

plt.subplot(1, 2, 2)
plt.plot(k_values, silhouette_scores, 'o-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Optimal k')

plt.tight_layout()
plt.show()

# 使用最佳k值进行聚类
best_k = 4
kmeans = KMeans(n_clusters=best_k, random_state=42)
y_pred = kmeans.fit_predict(X_scaled)

# 可视化聚类结果
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_pred, cmap='viridis', alpha=0.6)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=200, c='red', marker='X', label='Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title(f'K-means Clustering (k={best_k})')
plt.legend()
plt.show()

# 3. 层次聚类深入
print("\n3. 层次聚类深入")

from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

# 生成数据
X, y = make_blobs(
    n_samples=100, n_features=2, centers=3, 
    cluster_std=0.6, random_state=42
)

# 特征标准化
X_scaled = scaler.fit_transform(X)

# 绘制层次聚类树状图
plt.figure(figsize=(12, 6))
dendrogram = sch.dendrogram(sch.linkage(X_scaled, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Samples')
plt.ylabel('Euclidean distances')
plt.axhline(y=5, color='r', linestyle='--')
plt.show()

# 尝试不同的链接方法
linkage_methods = ['ward', 'complete', 'average', 'single']

plt.figure(figsize=(15, 10))

for i, method in enumerate(linkage_methods, 1):
    # 绘制树状图
    plt.subplot(2, 2, i)
    dendrogram = sch.dendrogram(sch.linkage(X_scaled, method=method))
    plt.title(f'Dendrogram (method={method})')
    plt.xlabel('Samples')
    plt.ylabel('Distance')

plt.tight_layout()
plt.show()

# 使用不同的链接方法进行聚类
plt.figure(figsize=(15, 10))

for i, method in enumerate(linkage_methods, 1):
    # 创建并训练层次聚类模型
    hierarchical = AgglomerativeClustering(n_clusters=3, linkage=method)
    y_pred = hierarchical.fit_predict(X_scaled)
    
    # 可视化聚类结果
    plt.subplot(2, 2, i)
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_pred, cmap='viridis', alpha=0.6)
    plt.title(f'Hierarchical Clustering (method={method})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()

# 4. DBSCAN聚类深入
print("\n4. DBSCAN聚类深入")

from sklearn.cluster import DBSCAN

# 生成不同类型的数据
plt.figure(figsize=(15, 5))

# 数据1: 高斯分布
X1, y1 = make_blobs(n_samples=100, centers=3, random_state=42)
plt.subplot(1, 3, 1)
plt.scatter(X1[:, 0], X1[:, 1], c=y1, cmap='viridis', alpha=0.6)
plt.title('Gaussian Blobs')

# 数据2: 新月形
X2, y2 = make_moons(n_samples=100, noise=0.05, random_state=42)
plt.subplot(1, 3, 2)
plt.scatter(X2[:, 0], X2[:, 1], c=y2, cmap='viridis', alpha=0.6)
plt.title('Moons')

# 数据3: 环形
X3, y3 = make_circles(n_samples=100, noise=0.05, factor=0.5, random_state=42)
plt.subplot(1, 3, 3)
plt.scatter(X3[:, 0], X3[:, 1], c=y3, cmap='viridis', alpha=0.6)
plt.title('Circles')

plt.tight_layout()
plt.show()

# 使用DBSCAN聚类不同类型的数据
datasets = [(X1, 'Blobs'), (X2, 'Moons'), (X3, 'Circles')]

plt.figure(figsize=(15, 5))

for i, (X_data, name) in enumerate(datasets, 1):
    # 特征标准化
    X_scaled = scaler.fit_transform(X_data)
    
    # DBSCAN聚类
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    y_pred = dbscan.fit_predict(X_scaled)
    
    # 可视化聚类结果
    plt.subplot(1, 3, i)
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_pred, cmap='viridis', alpha=0.6)
    plt.title(f'DBSCAN on {name}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()

# 5. Gaussian Mixture Models
print("\n5. Gaussian Mixture Models")

from sklearn.mixture import GaussianMixture

# 生成数据
X, y = make_blobs(
    n_samples=300, n_features=2, centers=3, 
    cluster_std=0.6, random_state=42
)

# 特征标准化
X_scaled = scaler.fit_transform(X)

# 尝试不同的组件数
n_components = range(1, 10)
bic_scores = []
aic_scores = []

for n in n_components:
    gmm = GaussianMixture(n_components=n, random_state=42)
    gmm.fit(X_scaled)
    bic_scores.append(gmm.bic(X_scaled))
    aic_scores.append(gmm.aic(X_scaled))

# 绘制BIC和AIC曲线
plt.figure(figsize=(8, 6))
plt.plot(n_components, bic_scores, 'o-', label='BIC')
plt.plot(n_components, aic_scores, 'o-', label='AIC')
plt.xlabel('Number of Components')
plt.ylabel('Score')
plt.title('BIC and AIC for Gaussian Mixture Models')
plt.legend()
plt.show()

# 使用最佳组件数进行聚类
best_n = 3
gmm = GaussianMixture(n_components=best_n, random_state=42)
y_pred = gmm.fit_predict(X_scaled)

# 可视化聚类结果
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_pred, cmap='viridis', alpha=0.6)

# 绘制高斯分布的轮廓
x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))
Z = gmm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.1, cmap='viridis')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title(f'Gaussian Mixture Model (n_components={best_n})')
plt.show()

# 6. 聚类评估
print("\n6. 聚类评估")

from sklearn.metrics import silhouette_score, adjusted_rand_score, homogeneity_score, completeness_score, v_measure_score

# 生成数据
X, y_true = make_blobs(
    n_samples=300, n_features=2, centers=3, 
    cluster_std=0.6, random_state=42
)

# 特征标准化
X_scaled = scaler.fit_transform(X)

# 应用不同的聚类算法
kmeans = KMeans(n_clusters=3, random_state=42)
y_pred_kmeans = kmeans.fit_predict(X_scaled)

hierarchical = AgglomerativeClustering(n_clusters=3)
y_pred_hierarchical = hierarchical.fit_predict(X_scaled)

dbscan = DBSCAN(eps=0.3, min_samples=5)
y_pred_dbscan = dbscan.fit_predict(X_scaled)

gmm = GaussianMixture(n_components=3, random_state=42)
y_pred_gmm = gmm.fit_predict(X_scaled)

# 评估聚类结果
algorithms = ['K-means', 'Hierarchical', 'DBSCAN', 'GMM']
y_preds = [y_pred_kmeans, y_pred_hierarchical, y_pred_dbscan, y_pred_gmm]

print("聚类算法评估:")
print("算法\t\t轮廓系数\tARI\t\t同质性\t完整性\tV-measure")
print("-" * 80)

for alg, y_pred in zip(algorithms, y_preds):
    # 计算评估指标
    silhouette = silhouette_score(X_scaled, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    homogeneity = homogeneity_score(y_true, y_pred)
    completeness = completeness_score(y_true, y_pred)
    v_measure = v_measure_score(y_true, y_pred)
    
    print(f"{alg}\t\t{silhouette:.4f}\t\t{ari:.4f}\t\t{homogeneity:.4f}\t{completeness:.4f}\t{v_measure:.4f}")

# 7. 实际应用：客户分群
print("\n7. 实际应用：客户分群")

from sklearn.datasets import make_classification

# 生成客户数据
X_customer, _ = make_classification(
    n_samples=1000, n_features=5, n_informative=5, 
    n_redundant=0, random_state=42
)

# 特征标准化
X_customer_scaled = scaler.fit_transform(X_customer)

# 使用K-means进行客户分群
kmeans = KMeans(n_clusters=4, random_state=42)
customer_clusters = kmeans.fit_predict(X_customer_scaled)

# 分析各簇的特征
print("客户分群结果分析:")
for i in range(4):
    cluster_data = X_customer_scaled[customer_clusters == i]
    print(f"\n簇 {i}:")
    print(f"  客户数量: {len(cluster_data)}")
    print(f"  特征均值:")
    for j in range(5):
        print(f"    特征 {j+1}: {cluster_data[:, j].mean():.4f}")

# 8. 练习
print("\n8. 练习")

# 练习1: 不同eps值对DBSCAN的影响
print("练习1: 不同eps值对DBSCAN的影响")

# 生成数据
X, y = make_moons(n_samples=200, noise=0.05, random_state=42)
X_scaled = scaler.fit_transform(X)

# 尝试不同的eps值
eps_values = [0.1, 0.2, 0.3, 0.4, 0.5]

plt.figure(figsize=(15, 10))

for i, eps in enumerate(eps_values, 1):
    # DBSCAN聚类
    dbscan = DBSCAN(eps=eps, min_samples=5)
    y_pred = dbscan.fit_predict(X_scaled)
    
    # 可视化聚类结果
    plt.subplot(2, 3, i)
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_pred, cmap='viridis', alpha=0.6)
    plt.title(f'DBSCAN (eps={eps})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()

# 练习2: 不同min_samples值对DBSCAN的影响
print("\n练习2: 不同min_samples值对DBSCAN的影响")

# 尝试不同的min_samples值
min_samples_values = [2, 5, 10, 15, 20]

plt.figure(figsize=(15, 10))

for i, min_samples in enumerate(min_samples_values, 1):
    # DBSCAN聚类
    dbscan = DBSCAN(eps=0.3, min_samples=min_samples)
    y_pred = dbscan.fit_predict(X_scaled)
    
    # 可视化聚类结果
    plt.subplot(2, 3, i)
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_pred, cmap='viridis', alpha=0.6)
    plt.title(f'DBSCAN (min_samples={min_samples})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()

print("\n=== 第40天学习示例结束 ===")
