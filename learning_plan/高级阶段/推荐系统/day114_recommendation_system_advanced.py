#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第114天：推荐系统高级方法
推荐系统学习示例
内容：矩阵分解、深度学习推荐系统和混合推荐
"""

print("=== 第114天：推荐系统高级方法 ===")

# 1. 矩阵分解
print("\n1. 矩阵分解")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error

print("矩阵分解是将用户-物品矩阵分解为低维矩阵的方法")
print("- 奇异值分解 (SVD)")
print("- 非负矩阵分解 (NMF)")
print("- 隐因子模型")

# 示例用户-物品评分矩阵
ratings = np.array([
    [5, 4, 0, 0, 1],
    [5, 0, 0, 4, 0],
    [0, 0, 5, 0, 4],
    [0, 3, 4, 0, 5],
    [0, 5, 0, 0, 4]
])

users = ['User1', 'User2', 'User3', 'User4', 'User5']
items = ['Item1', 'Item2', 'Item3', 'Item4', 'Item5']

ratings_df = pd.DataFrame(ratings, index=users, columns=items)
print("用户-物品评分矩阵:")
print(ratings_df)

# 2. 奇异值分解 (SVD)
print("\n2. 奇异值分解 (SVD)")

# 执行SVD
U, sigma, Vt = svds(ratings, k=2)

# 将sigma转换为对角矩阵
sigma = np.diag(sigma)

print(f"U矩阵形状: {U.shape}")
print(f"Sigma矩阵形状: {sigma.shape}")
print(f"Vt矩阵形状: {Vt.shape}")

# 重建评分矩阵
reconstructed_ratings = np.dot(np.dot(U, sigma), Vt)
print("\n重建的评分矩阵:")
print(pd.DataFrame(reconstructed_ratings, index=users, columns=items))

# 计算均方误差
mse = mean_squared_error(ratings[ratings > 0], reconstructed_ratings[ratings > 0])
print(f"\nSVD 均方误差: {mse:.4f}")

# 3. 非负矩阵分解 (NMF)
print("\n3. 非负矩阵分解 (NMF)")

# 执行NMF
nmf = NMF(n_components=2, init='random', random_state=42)
W = nmf.fit_transform(ratings)
H = nmf.components_

print(f"W矩阵形状: {W.shape}")
print(f"H矩阵形状: {H.shape}")

# 重建评分矩阵
reconstructed_ratings_nmf = np.dot(W, H)
print("\n重建的评分矩阵:")
print(pd.DataFrame(reconstructed_ratings_nmf, index=users, columns=items))

# 计算均方误差
mse_nmf = mean_squared_error(ratings[ratings > 0], reconstructed_ratings_nmf[ratings > 0])
print(f"\nNMF 均方误差: {mse_nmf:.4f}")

# 4. 深度学习推荐系统
print("\n4. 深度学习推荐系统")

print("深度学习推荐系统使用神经网络来学习用户和物品的表示")
print("- 神经网络协同过滤")
print("- 深度因子分解机")
print("- 自编码器")
print("- 注意力机制")

# 示例：神经网络协同过滤
print("\n神经网络协同过滤示例:")
print("import tensorflow as tf")
print("from tensorflow.keras.models import Model")
print("from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate")
print("")
print("# 定义模型")
print("user_input = Input(shape=(1,))")
print("item_input = Input(shape=(1,))")
print("")
print("# 用户嵌入")
print("user_embedding = Embedding(input_dim=5, output_dim=10)(user_input)")
print("user_flat = Flatten()(user_embedding)")
print("")
print("# 物品嵌入")
print("item_embedding = Embedding(input_dim=5, output_dim=10)(item_input)")
print("item_flat = Flatten()(item_embedding)")
print("")
print("# 合并特征")
print("concat = Concatenate()([user_flat, item_flat])")
print("dense1 = Dense(32, activation='relu')(concat)")
print("dense2 = Dense(16, activation='relu')(dense1)")
print("output = Dense(1, activation='linear')(dense2)")
print("")
print("# 创建模型")
print("model = Model(inputs=[user_input, item_input], outputs=output)")
print("model.compile(optimizer='adam', loss='mean_squared_error')")
print("")
print("# 准备数据")
print("user_ids = [0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4]")
print("item_ids = [0, 1, 4, 0, 3, 2, 4, 1, 2, 4, 1, 4]")
print("ratings = [5, 4, 1, 5, 4, 5, 4, 3, 4, 5, 5, 4]")
print("")
print("# 训练模型")
print("model.fit([user_ids, item_ids], ratings, epochs=100, batch_size=1)")

# 5. 混合推荐系统
print("\n5. 混合推荐系统")

print("混合推荐系统结合多种推荐方法")
print("- 加权混合：对不同推荐方法的结果进行加权")
print("- 切换混合：根据情况选择不同的推荐方法")
print("- 混合特征：将不同方法的特征结合")
print("- 分层混合：使用一种方法的结果作为另一种方法的输入")

# 示例：加权混合
print("\n加权混合示例:")

# 基于用户的协同过滤预测
user_based_pred = np.array([3.5, 4.2, 2.8])
# 基于物品的协同过滤预测
item_based_pred = np.array([3.8, 4.0, 3.0])
# 基于内容的推荐预测
content_based_pred = np.array([3.2, 3.9, 3.1])

# 权重
weights = [0.4, 0.3, 0.3]

# 加权混合
hybrid_pred = (user_based_pred * weights[0] + 
               item_based_pred * weights[1] + 
               content_based_pred * weights[2])

print(f"基于用户的预测: {user_based_pred}")
print(f"基于物品的预测: {item_based_pred}")
print(f"基于内容的预测: {content_based_pred}")
print(f"加权混合预测: {hybrid_pred}")

# 6. 推荐系统的评估
print("\n6. 推荐系统的评估")

print("推荐系统的评估指标:")
print("1. 离线评估：使用历史数据评估")
print("2. 在线评估：通过A/B测试评估")
print("3. 综合评估：结合离线和在线评估")

# 7. 推荐系统的可扩展性
print("\n7. 推荐系统的可扩展性")

print("推荐系统的可扩展性挑战:")
print("- 数据量增长")
print("- 用户和物品数量增加")
print("- 实时推荐需求")

print("解决方案:")
print("1. 分布式计算")
print("2. 缓存策略")
print("3. 近似算法")
print("4. 模型压缩")

# 8. 推荐系统的多样性
print("\n8. 推荐系统的多样性")

print("推荐系统的多样性与准确性的平衡")
print("- 多样性：推荐不同类型的物品")
print("- 准确性：推荐用户可能喜欢的物品")
print("- 新颖性：推荐用户尚未接触的物品")
print("- 覆盖率：推荐系统能够推荐的物品比例")

# 9. 推荐系统的隐私保护
print("\n9. 推荐系统的隐私保护")

print("推荐系统的隐私保护考虑:")
print("- 数据脱敏")
print("- 联邦学习")
print("- 差分隐私")
print("- 本地计算")

# 10. 练习
print("\n10. 练习")

# 练习1: 矩阵分解
print("练习1: 矩阵分解")
print("- 尝试不同的分解维度")
print("- 比较不同矩阵分解方法")
print("- 分析分解结果")

# 练习2: 深度学习推荐系统
print("\n练习2: 深度学习推荐系统")
print("- 实现神经网络协同过滤")
print("- 尝试不同的网络结构")
print("- 评估模型性能")

# 练习3: 混合推荐系统
print("\n练习3: 混合推荐系统")
print("- 实现不同的混合策略")
print("- 调优权重参数")
print("- 评估混合推荐效果")

# 练习4: 实时推荐
print("\n练习4: 实时推荐")
print("- 设计实时推荐系统")
print("- 实现增量学习")
print("- 优化推荐速度")

print("\n=== 第114天学习示例结束 ===")
