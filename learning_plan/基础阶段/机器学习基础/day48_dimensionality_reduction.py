#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第48天：降维
机器学习基础学习示例
内容：降维的基本概念、主成分分析（PCA）、t-SNE、线性判别分析（LDA）
"""

print("=== 第48天：降维 ===")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine, make_blobs, fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE, MDS, Isomap
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# 1. 降维概述
print("\n1. 降维概述")

print("降维是一种将高维数据转换为低维表示的技术")
print("- 目的：减少数据维度，提高模型性能，可视化高维数据")
print("- 类型：线性降维和非线性降维")
print("- 应用：数据可视化、特征提取、噪声去除")

# 2. 数据集准备
print("\n2. 数据集准备")

# 加载iris数据集
iris = load_iris()
X, y = iris.data, iris.target
print(f"Iris数据集形状: {X.shape}, {y.shape}")

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 主成分分析（PCA）
print("\n3. 主成分分析（PCA）")

print("PCA是一种线性降维方法，通过正交变换将数据转换到新的坐标系")
print("- 目的：找到数据中最主要的方差方向")
print("- 步骤：1. 数据标准化 2. 计算协方差矩阵 3. 计算特征值和特征向量 4. 选择主成分")

# 3.1 基本PCA
print("\n3.1 基本PCA")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA前特征数量: {X_scaled.shape[1]}")
print(f"PCA后特征数量: {X_pca.shape[1]}")
print(f"解释方差比例: {pca.explained_variance_ratio_}")
print(f"累计解释方差比例: {np.cumsum(pca.explained_variance_ratio_)}")

# 可视化PCA结果
plt.figure(figsize=(10, 6))
for i in range(len(np.unique(y))):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=iris.target_names[i])
plt.title('PCA结果可视化')
plt.xlabel('主成分1')
plt.ylabel('主成分2')
plt.legend()
plt.savefig('pca_visualization.png')
print("PCA结果已保存为 pca_visualization.png")

# 3.2 选择主成分数量
print("\n3.2 选择主成分数量")

# 计算不同主成分数量的解释方差
pca_full = PCA()
pca_full.fit(X_scaled)
explained_variance = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# 可视化解释方差
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance) + 1), cumulative_variance, 'o-')
plt.axhline(y=0.95, color='r', linestyle='--')
plt.title('累计解释方差')
plt.xlabel('主成分数量')
plt.ylabel('累计解释方差比例')
plt.grid()
plt.savefig('pca_variance.png')
print("PCA解释方差图已保存为 pca_variance.png")

# 3.3 核PCA
print("\n3.3 核PCA")

print("核PCA是PCA的扩展，用于处理非线性数据")

# 生成非线性数据
X_nonlinear, y_nonlinear = make_blobs(n_samples=300, centers=3, n_features=10, random_state=42)

# 应用核PCA
kernel_pca = KernelPCA(n_components=2, kernel='rbf', gamma=0.1)
X_kernel_pca = kernel_pca.fit_transform(X_nonlinear)

# 可视化核PCA结果
plt.figure(figsize=(10, 6))
for i in range(len(np.unique(y_nonlinear))):
    plt.scatter(X_kernel_pca[y_nonlinear == i, 0], X_kernel_pca[y_nonlinear == i, 1], label=f'类别{i}')
plt.title('核PCA结果可视化')
plt.xlabel('主成分1')
plt.ylabel('主成分2')
plt.legend()
plt.savefig('kernel_pca_visualization.png')
print("核PCA结果已保存为 kernel_pca_visualization.png")

# 4. t-SNE
print("\n4. t-SNE")

print("t-SNE是一种非线性降维方法，特别适合高维数据的可视化")
print("- 目的：保持数据的局部结构")
print("- 优点：可视化效果好，能很好地分离不同类别的数据")

# 应用t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)

# 可视化t-SNE结果
plt.figure(figsize=(10, 6))
for i in range(len(np.unique(y))):
    plt.scatter(X_tsne[y == i, 0], X_tsne[y == i, 1], label=iris.target_names[i])
plt.title('t-SNE结果可视化')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.legend()
plt.savefig('tsne_visualization.png')
print("t-SNE结果已保存为 tsne_visualization.png")

# 5. 线性判别分析（LDA）
print("\n5. 线性判别分析（LDA）")

print("LDA是一种监督式降维方法，考虑类别信息")
print("- 目的：最大化类间距离，最小化类内距离")
print("- 应用：分类问题的特征提取")

# 应用LDA
lda = LDA(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)

# 可视化LDA结果
plt.figure(figsize=(10, 6))
for i in range(len(np.unique(y))):
    plt.scatter(X_lda[y == i, 0], X_lda[y == i, 1], label=iris.target_names[i])
plt.title('LDA结果可视化')
plt.xlabel('LDA 1')
plt.ylabel('LDA 2')
plt.legend()
plt.savefig('lda_visualization.png')
print("LDA结果已保存为 lda_visualization.png")

# 6. 其他降维方法
print("\n6. 其他降维方法")

print("- MDS：多维 scaling，保持数据点之间的距离")
print("- Isomap：等距映射，保持流形结构")

# 应用MDS
mds = MDS(n_components=2, random_state=42)
X_mds = mds.fit_transform(X_scaled)

# 应用Isomap
isomap = Isomap(n_components=2, n_neighbors=10)
X_isomap = isomap.fit_transform(X_scaled)

# 可视化其他降维方法的结果
plt.figure(figsize=(15, 6))

# MDS
ax1 = plt.subplot(1, 2, 1)
for i in range(len(np.unique(y))):
    ax1.scatter(X_mds[y == i, 0], X_mds[y == i, 1], label=iris.target_names[i])
ax1.set_title('MDS结果可视化')
ax1.set_xlabel('MDS 1')
ax1.set_ylabel('MDS 2')
ax1.legend()

# Isomap
ax2 = plt.subplot(1, 2, 2)
for i in range(len(np.unique(y))):
    ax2.scatter(X_isomap[y == i, 0], X_isomap[y == i, 1], label=iris.target_names[i])
ax2.set_title('Isomap结果可视化')
ax2.set_xlabel('Isomap 1')
ax2.set_ylabel('Isomap 2')
ax2.legend()

plt.tight_layout()
plt.savefig('other_dimensionality_reduction.png')
print("其他降维方法结果已保存为 other_dimensionality_reduction.png")

# 7. 降维的应用：数据可视化
print("\n7. 降维的应用：数据可视化")

print("降维在数据可视化中的应用")

# 加载LFW人脸数据集
try:
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    X_faces = lfw_people.data
    y_faces = lfw_people.target
    target_names = lfw_people.target_names
    n_samples, h, w = lfw_people.images.shape
    print(f"LFW数据集: {n_samples} 张图片, 每张图片 {h}x{w} 像素")
    
    # 应用PCA进行人脸数据降维
    n_components = 150
    pca_faces = PCA(n_components=n_components, whiten=True, random_state=42)
    X_faces_pca = pca_faces.fit_transform(X_faces)
    
    # 应用t-SNE进行可视化
    tsne_faces = TSNE(n_components=2, random_state=42, perplexity=30)
    X_faces_tsne = tsne_faces.fit_transform(X_faces_pca)
    
    # 可视化人脸数据
    plt.figure(figsize=(15, 10))
    for i in range(len(target_names)):
        plt.scatter(X_faces_tsne[y_faces == i, 0], X_faces_tsne[y_faces == i, 1], label=target_names[i])
    plt.title('LFW人脸数据t-SNE可视化')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend()
    plt.savefig('faces_visualization.png')
    print("人脸数据可视化结果已保存为 faces_visualization.png")
except Exception as e:
    print(f"加载LFW数据集失败: {e}")
    print("跳过人脸数据可视化部分")

# 8. 降维的应用：分类任务
print("\n8. 降维的应用：分类任务")

print("降维在分类任务中的应用")

# 加载wine数据集
wine = load_wine()
X_wine, y_wine = wine.data, wine.target
X_wine_scaled = StandardScaler().fit_transform(X_wine)
X_wine_train, X_wine_test, y_wine_train, y_wine_test = train_test_split(X_wine_scaled, y_wine, test_size=0.3, random_state=42)

# 原始数据分类
model_original = LogisticRegression(max_iter=1000, random_state=42)
model_original.fit(X_wine_train, y_wine_train)
y_wine_pred_original = model_original.predict(X_wine_test)
accuracy_original = accuracy_score(y_wine_test, y_wine_pred_original)
print(f"原始数据分类准确率: {accuracy_original:.4f}")

# PCA降维后分类
pca_wine = PCA(n_components=2)
X_wine_train_pca = pca_wine.fit_transform(X_wine_train)
X_wine_test_pca = pca_wine.transform(X_wine_test)
model_pca = LogisticRegression(max_iter=1000, random_state=42)
model_pca.fit(X_wine_train_pca, y_wine_train)
y_wine_pred_pca = model_pca.predict(X_wine_test_pca)
accuracy_pca = accuracy_score(y_wine_test, y_wine_pred_pca)
print(f"PCA降维后分类准确率: {accuracy_pca:.4f}")

# LDA降维后分类
lda_wine = LDA(n_components=2)
X_wine_train_lda = lda_wine.fit_transform(X_wine_train, y_wine_train)
X_wine_test_lda = lda_wine.transform(X_wine_test)
model_lda = LogisticRegression(max_iter=1000, random_state=42)
model_lda.fit(X_wine_train_lda, y_wine_train)
y_wine_pred_lda = model_lda.predict(X_wine_test_lda)
accuracy_lda = accuracy_score(y_wine_test, y_wine_pred_lda)
print(f"LDA降维后分类准确率: {accuracy_lda:.4f}")

# 9. 降维的应用：聚类任务
print("\n9. 降维的应用：聚类任务")

print("降维在聚类任务中的应用")

# 应用K-means聚类
kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# 可视化聚类结果
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, cmap='viridis')
plt.title('K-means聚类结果（PCA降维）')
plt.xlabel('主成分1')
plt.ylabel('主成分2')
plt.savefig('clustering_result.png')
print("聚类结果已保存为 clustering_result.png")

# 10. 降维方法的比较
print("\n10. 降维方法的比较")

print("不同降维方法的特点:")
print("- PCA：线性降维，计算效率高，适合大规模数据")
print("- 核PCA：非线性降维，适合处理非线性数据")
print("- t-SNE：非线性降维，可视化效果好，但计算成本高")
print("- LDA：监督式降维，考虑类别信息，适合分类任务")
print("- MDS：保持数据点之间的距离，适合度量学习")
print("- Isomap：保持流形结构，适合非线性数据")

# 11. 降维的最佳实践
print("\n11. 降维的最佳实践")

print("降维的最佳实践:")
print("- 数据预处理：标准化数据")
print("- 选择合适的降维方法：根据数据特点和任务需求")
print("- 选择合适的维度：通过解释方差或交叉验证")
print("- 结合其他技术：降维后可以使用分类或聚类算法")
print("- 可视化：使用t-SNE等方法进行数据可视化")

# 12. 练习
print("\n12. 练习")

# 练习1: 实现PCA
print("练习1: 实现PCA")
print("- 手动实现PCA算法")
print("- 与sklearn的PCA结果比较")

# 练习2: 降维方法的参数调优
print("\n练习2: 降维方法的参数调优")
print("- 调优t-SNE的perplexity参数")
print("- 分析不同参数对可视化效果的影响")

# 练习3: 降维与特征选择的结合
print("\n练习3: 降维与特征选择的结合")
print("- 先使用特征选择，再使用降维")
print("- 评估结合方法的效果")

# 练习4: 大规模数据的降维
print("\n练习4: 大规模数据的降维")
print("- 生成大规模数据集")
print("- 测试不同降维方法的性能")

# 练习5: 降维的应用实践
print("\n练习5: 降维的应用实践")
print("- 选择一个实际数据集")
print("- 应用不同的降维方法")
print("- 评估降维对后续任务的影响")

print("\n=== 第48天学习示例结束 ===")
