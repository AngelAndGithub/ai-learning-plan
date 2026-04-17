import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_blobs, make_moons, fetch_openml
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE, LocallyLinearEmbedding, Isomap
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
from mlxtend.frequent_patterns import apriori, association_rules

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 1. K-means聚类示例
def kmeans_example():
    """K-means聚类示例"""
    print("=== K-means聚类示例 ===")
    
    # 生成聚类数据集
    X, y = make_blobs(n_samples=300, n_features=2, centers=4, cluster_std=0.6, random_state=42)
    
    # 数据可视化
    plt.scatter(X[:, 0], X[:, 1])
    plt.title("原始数据")
    plt.xlabel("特征 1")
    plt.ylabel("特征 2")
    plt.savefig('kmeans_original_data.png')
    plt.close()
    
    # 选择最佳K值
    inertia = []
    silhouette_scores = []
    K_range = range(2, 10)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))
    
    # 肘部法则
    plt.plot(K_range, inertia, 'bx-')
    plt.title('肘部法则')
    plt.xlabel('K值')
    plt.ylabel('惯性')
    plt.savefig('kmeans_elbow_method.png')
    plt.close()
    
    # 轮廓系数
    plt.plot(K_range, silhouette_scores, 'bx-')
    plt.title('轮廓系数')
    plt.xlabel('K值')
    plt.ylabel('轮廓系数')
    plt.savefig('kmeans_silhouette_score.png')
    plt.close()
    
    # 使用最佳K值（这里选择k=4）
    kmeans = KMeans(n_clusters=4, random_state=42)
    y_pred = kmeans.fit_predict(X)
    
    # 可视化聚类结果
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='*')
    plt.title('K-means聚类结果')
    plt.xlabel('特征 1')
    plt.ylabel('特征 2')
    plt.savefig('kmeans_clustering_result.png')
    plt.close()
    
    # 评估聚类质量
    print(f"K-means 轮廓系数: {silhouette_score(X, y_pred):.4f}")
    print(f"K-means Davies-Bouldin指数: {davies_bouldin_score(X, y_pred):.4f}")
    print(f"K-means Calinski-Harabasz指数: {calinski_harabasz_score(X, y_pred):.4f}")
    
    return kmeans

# 2. 层次聚类示例
def hierarchical_clustering_example():
    """层次聚类示例"""
    print("\n=== 层次聚类示例 ===")
    
    # 生成聚类数据集
    X, y = make_blobs(n_samples=100, n_features=2, centers=3, cluster_std=0.6, random_state=42)
    
    # 数据可视化
    plt.scatter(X[:, 0], X[:, 1])
    plt.title("原始数据")
    plt.xlabel("特征 1")
    plt.ylabel("特征 2")
    plt.savefig('hierarchical_original_data.png')
    plt.close()
    
    # 生成层次聚类的链接矩阵
    Z = linkage(X, method='ward')
    
    # 绘制 dendrogram
    plt.figure(figsize=(10, 7))
    dendrogram(Z)
    plt.title('层次聚类树状图')
    plt.xlabel('样本索引')
    plt.ylabel('距离')
    plt.savefig('hierarchical_dendrogram.png')
    plt.close()
    
    # 执行层次聚类
    hierarchical = AgglomerativeClustering(n_clusters=3, linkage='ward')
    y_pred = hierarchical.fit_predict(X)
    
    # 可视化聚类结果
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
    plt.title('层次聚类结果')
    plt.xlabel('特征 1')
    plt.ylabel('特征 2')
    plt.savefig('hierarchical_clustering_result.png')
    plt.close()
    
    # 评估聚类质量
    print(f"层次聚类 轮廓系数: {silhouette_score(X, y_pred):.4f}")
    print(f"层次聚类 Davies-Bouldin指数: {davies_bouldin_score(X, y_pred):.4f}")
    print(f"层次聚类 Calinski-Harabasz指数: {calinski_harabasz_score(X, y_pred):.4f}")
    
    return hierarchical

# 3. DBSCAN聚类示例
def dbscan_example():
    """DBSCAN聚类示例"""
    print("\n=== DBSCAN聚类示例 ===")
    
    # 生成非球形聚类数据集
    X, y = make_moons(n_samples=200, noise=0.05, random_state=42)
    
    # 数据可视化
    plt.scatter(X[:, 0], X[:, 1])
    plt.title("原始数据")
    plt.xlabel("特征 1")
    plt.ylabel("特征 2")
    plt.savefig('dbscan_original_data.png')
    plt.close()
    
    # 执行DBSCAN聚类
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    y_pred = dbscan.fit_predict(X)
    
    # 可视化聚类结果
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
    plt.title('DBSCAN聚类结果')
    plt.xlabel('特征 1')
    plt.ylabel('特征 2')
    plt.savefig('dbscan_clustering_result.png')
    plt.close()
    
    # 评估聚类质量
    print(f"DBSCAN 轮廓系数: {silhouette_score(X, y_pred):.4f}")
    print(f"DBSCAN Davies-Bouldin指数: {davies_bouldin_score(X, y_pred):.4f}")
    print(f"DBSCAN Calinski-Harabasz指数: {calinski_harabasz_score(X, y_pred):.4f}")
    print(f"DBSCAN 发现的簇数: {len(set(y_pred)) - (1 if -1 in y_pred else 0)}")
    print(f"DBSCAN 噪声点数: {list(y_pred).count(-1)}")
    
    return dbscan

# 4. PCA降维示例
def pca_example():
    """PCA降维示例"""
    print("\n=== PCA降维示例 ===")
    
    # 加载鸢尾花数据集
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 执行PCA降维
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # 解释方差
    print(f"PCA 解释方差: {pca.explained_variance_ratio_}")
    print(f"PCA 累计解释方差: {sum(pca.explained_variance_ratio_):.4f}")
    
    # 可视化降维结果
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
    plt.title('PCA降维结果')
    plt.xlabel('主成分 1')
    plt.ylabel('主成分 2')
    plt.colorbar()
    plt.savefig('pca_result.png')
    plt.close()
    
    # 绘制解释方差图
    pca_full = PCA()
    pca_full.fit(X_scaled)
    explained_variance = pca_full.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    plt.plot(range(1, len(explained_variance) + 1), cumulative_variance, 'bx-')
    plt.title('PCA累计解释方差')
    plt.xlabel('主成分数')
    plt.ylabel('累计解释方差')
    plt.axhline(y=0.95, color='r', linestyle='--')
    plt.savefig('pca_explained_variance.png')
    plt.close()
    
    return pca

# 5. t-SNE降维示例
def tsne_example():
    """t-SNE降维示例"""
    print("\n=== t-SNE降维示例 ===")
    
    # 加载鸢尾花数据集
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 执行t-SNE降维
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    
    # 可视化降维结果
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
    plt.title('t-SNE降维结果')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.colorbar()
    plt.savefig('tsne_result.png')
    plt.close()
    
    return tsne

# 6. 其他降维技术示例
def other_dimension_reduction_examples():
    """其他降维技术示例"""
    print("\n=== 其他降维技术示例 ===")
    
    # 加载鸢尾花数据集
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # LLE降维
    lle = LocallyLinearEmbedding(n_components=2, random_state=42)
    X_lle = lle.fit_transform(X_scaled)
    
    # Isomap降维
    isomap = Isomap(n_components=2)
    X_isomap = isomap.fit_transform(X_scaled)
    
    # 可视化结果
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.scatter(X_lle[:, 0], X_lle[:, 1], c=y, cmap='viridis')
    plt.title('LLE降维结果')
    plt.xlabel('LLE 1')
    plt.ylabel('LLE 2')
    
    plt.subplot(1, 3, 2)
    plt.scatter(X_isomap[:, 0], X_isomap[:, 1], c=y, cmap='viridis')
    plt.title('Isomap降维结果')
    plt.xlabel('Isomap 1')
    plt.ylabel('Isomap 2')
    
    # PCA作为对比
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.subplot(1, 3, 3)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
    plt.title('PCA降维结果')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    
    plt.tight_layout()
    plt.savefig('dimension_reduction_comparison.png')
    plt.close()
    
    return lle, isomap, pca

# 7. 关联规则学习示例
def association_rules_example():
    """关联规则学习示例"""
    print("\n=== 关联规则学习示例 ===")
    
    # 创建示例交易数据
    data = [
        ['牛奶', '面包', '鸡蛋'],
        ['牛奶', '面包', '啤酒'],
        ['牛奶', '尿布'],
        ['面包', '尿布', '啤酒', '鸡蛋'],
        ['面包', '尿布', '啤酒'],
        ['尿布', '啤酒']
    ]
    
    # 转换为one-hot编码
    from mlxtend.preprocessing import TransactionEncoder
    te = TransactionEncoder()
    te_ary = te.fit(data).transform(data)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    
    print("交易数据:")
    print(df)
    
    # 使用Apriori算法挖掘频繁项集
    frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)
    print("\n频繁项集:")
    print(frequent_itemsets)
    
    # 生成关联规则
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
    print("\n关联规则:")
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
    
    return frequent_itemsets, rules

# 8. 异常检测示例
def anomaly_detection_example():
    """异常检测示例"""
    print("\n=== 异常检测示例 ===")
    
    # 生成正常数据
    X_normal = np.random.normal(0, 1, (100, 2))
    
    # 生成异常数据
    X_anomaly = np.random.normal(5, 1, (10, 2))
    
    # 合并数据
    X = np.vstack([X_normal, X_anomaly])
    y_true = np.zeros(len(X), dtype=int)
    y_true[-10:] = 1  # 标记异常点
    
    # 数据可视化
    plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis')
    plt.title('异常检测数据集')
    plt.xlabel('特征 1')
    plt.ylabel('特征 2')
    plt.savefig('anomaly_detection_data.png')
    plt.close()
    
    # 使用DBSCAN进行异常检测
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    y_pred = dbscan.fit_predict(X)
    
    # 可视化异常检测结果
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
    plt.title('DBSCAN异常检测结果')
    plt.xlabel('特征 1')
    plt.ylabel('特征 2')
    plt.savefig('anomaly_detection_result.png')
    plt.close()
    
    # 评估异常检测性能
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_true, (y_pred == -1).astype(int)).ravel()
    print(f"真阴性: {tn}, 假阳性: {fp}, 假阴性: {fn}, 真阳性: {tp}")
    print(f"准确率: {(tn + tp) / (tn + fp + fn + tp):.4f}")
    print(f"精确率: {tp / (tp + fp):.4f}")
    print(f"召回率: {tp / (tp + fn):.4f}")
    
    return dbscan

# 9. 完整的无监督学习流程示例
def complete_unsupervised_workflow():
    """完整的无监督学习流程示例"""
    print("\n=== 完整的无监督学习流程示例 ===")
    
    # 加载MNIST数据集
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist.data, mnist.target
    
    # 只使用前1000个样本以加快计算
    X = X[:1000]
    y = y[:1000]
    
    print(f"MNIST数据集形状: X={X.shape}, y={y.shape}")
    
    # 数据预处理
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 降维
    print("执行PCA降维...")
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X_scaled)
    print(f"PCA降维后形状: {X_pca.shape}")
    print(f"PCA累计解释方差: {sum(pca.explained_variance_ratio_):.4f}")
    
    # 聚类
    print("执行K-means聚类...")
    kmeans = KMeans(n_clusters=10, random_state=42)
    y_pred = kmeans.fit_predict(X_pca)
    
    # 评估聚类质量
    print(f"K-means 轮廓系数: {silhouette_score(X_pca, y_pred):.4f}")
    print(f"K-means Davies-Bouldin指数: {davies_bouldin_score(X_pca, y_pred):.4f}")
    print(f"K-means Calinski-Harabasz指数: {calinski_harabasz_score(X_pca, y_pred):.4f}")
    
    # 可视化聚类结果
    print("执行t-SNE可视化...")
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_pca)
    
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_pred, cmap='viridis')
    plt.title('MNIST聚类结果可视化')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.colorbar()
    plt.savefig('mnist_clustering_visualization.png')
    plt.close()
    
    # 分析聚类结果
    cluster_labels = pd.DataFrame({'true_label': y, 'cluster': y_pred})
    print("\n聚类结果分析:")
    print(cluster_labels.groupby('cluster').agg({'true_label': lambda x: x.value_counts().index[0]}))
    
    return X_pca, y_pred

if __name__ == "__main__":
    # 运行所有示例
    kmeans_example()
    hierarchical_clustering_example()
    dbscan_example()
    pca_example()
    tsne_example()
    other_dimension_reduction_examples()
    association_rules_example()
    anomaly_detection_example()
    complete_unsupervised_workflow()
    
    print("\n所有示例运行完成！")