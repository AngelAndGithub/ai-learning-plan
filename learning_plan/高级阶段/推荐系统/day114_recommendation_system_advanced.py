#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第114天：推荐系统进阶
推荐系统学习示例
内容：矩阵分解、深度学习推荐、实时推荐、推荐系统评估
"""

print("=== 第114天：推荐系统进阶 ===")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from math import sqrt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Dense, Flatten, Input, Dot, Add
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import random

# 1. 矩阵分解
print("\n1. 矩阵分解")

print("矩阵分解是推荐系统的重要方法")
print("- 原理：将用户-物品评分矩阵分解为用户矩阵和物品矩阵的乘积")
print("- 方法：SVD（奇异值分解）、NMF（非负矩阵分解）")
print("- 优点：能够处理稀疏数据，捕捉潜在特征")

# 准备数据
np.random.seed(42)

# 用户ID
users = list(range(1, 11))

# 物品ID
items = list(range(1, 11))

# 随机生成用户-物品评分矩阵
ratings = np.zeros((len(users), len(items)))
for i in range(len(users)):
    # 每个用户随机评分5个物品
    rated_items = random.sample(items, 5)
    for item in rated_items:
        ratings[i, item-1] = random.randint(1, 5)

# 创建DataFrame
ratings_df = pd.DataFrame(ratings, index=users, columns=items)
print("用户-物品评分矩阵:")
print(ratings_df)

# 2. SVD矩阵分解
print("\n2. SVD矩阵分解")

# 填充缺失值（使用均值）
mean_rating = np.mean(ratings[ratings > 0])
ratings_filled = ratings.copy()
ratings_filled[ratings_filled == 0] = mean_rating

# 应用SVD
n_components = 2
svd = TruncatedSVD(n_components=n_components)
user_matrix = svd.fit_transform(ratings_filled)
item_matrix = svd.components_.T

print(f"用户矩阵形状: {user_matrix.shape}")
print(f"物品矩阵形状: {item_matrix.shape}")

# 重构评分矩阵
ratings_reconstructed = np.dot(user_matrix, item_matrix.T)
print("\n重构的评分矩阵:")
print(pd.DataFrame(ratings_reconstructed, index=users, columns=items))

# 计算重构误差
mse = mean_squared_error(ratings_filled.flatten(), ratings_reconstructed.flatten())
rmse = sqrt(mse)
print(f"\n重构误差 (RMSE): {rmse:.4f}")

# 3. 深度学习推荐系统
print("\n3. 深度学习推荐系统")

print("深度学习推荐系统使用神经网络来建模用户和物品的交互")
print("- 类型：")
print("  - 协同过滤神经网络")
print("  - 深度推荐系统（DeepFM、Wide & Deep等）")
print("  - 序列推荐系统（基于RNN、Transformer等）")

# 准备数据
ratings_long = []
for user in users:
    for item in items:
        rating = ratings[user-1, item-1]
        if rating > 0:
            ratings_long.append({'user_id': user, 'item_id': item, 'rating': rating})

ratings_long_df = pd.DataFrame(ratings_long)
print("\n长格式评分数据:")
print(ratings_long_df.head())

# 分割训练集和测试集
train_df, test_df = train_test_split(ratings_long_df, test_size=0.2, random_state=42)

# 4. 矩阵分解神经网络
print("\n4. 矩阵分解神经网络")

# 获取用户和物品的数量
n_users = len(users)
n_items = len(items)

# 构建矩阵分解神经网络
user_input = Input(shape=(1,))
item_input = Input(shape=(1,))

# 嵌入层
user_embedding = Embedding(n_users + 1, 10)(user_input)
item_embedding = Embedding(n_items + 1, 10)(item_input)

# 展平
user_flat = Flatten()(user_embedding)
item_flat = Flatten()(item_embedding)

# 点积
dot_product = Dot(axes=1)([user_flat, item_flat])

# 模型
model = Model(inputs=[user_input, item_input], outputs=dot_product)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

model.summary()

# 训练模型
print("训练矩阵分解神经网络...")
history = model.fit(
    [train_df['user_id'], train_df['item_id']],
    train_df['rating'],
    epochs=100,
    batch_size=4,
    validation_data=([test_df['user_id'], test_df['item_id']], test_df['rating']),
    verbose=1
)

# 评估模型
test_loss = model.evaluate([test_df['user_id'], test_df['item_id']], test_df['rating'], verbose=0)
print(f"测试损失 (MSE): {test_loss:.4f}")
print(f"测试损失 (RMSE): {sqrt(test_loss):.4f}")

# 5. 深度推荐系统
print("\n5. 深度推荐系统")

# 构建深度推荐系统
user_input = Input(shape=(1,))
item_input = Input(shape=(1,))

# 嵌入层
user_embedding = Embedding(n_users + 1, 10)(user_input)
item_embedding = Embedding(n_items + 1, 10)(item_input)

# 展平
user_flat = Flatten()(user_embedding)
item_flat = Flatten()(item_embedding)

# 连接
concat = tf.keras.layers.Concatenate()([user_flat, item_flat])

# 全连接层
fc1 = Dense(64, activation='relu')(concat)
fc2 = Dense(32, activation='relu')(fc1)
out = Dense(1)(fc2)

# 模型
deep_model = Model(inputs=[user_input, item_input], outputs=out)
deep_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

deep_model.summary()

# 训练模型
print("训练深度推荐系统...")
deep_history = deep_model.fit(
    [train_df['user_id'], train_df['item_id']],
    train_df['rating'],
    epochs=100,
    batch_size=4,
    validation_data=([test_df['user_id'], test_df['item_id']], test_df['rating']),
    verbose=1
)

# 评估模型
deep_test_loss = deep_model.evaluate([test_df['user_id'], test_df['item_id']], test_df['rating'], verbose=0)
print(f"深度推荐系统测试损失 (MSE): {deep_test_loss:.4f}")
print(f"深度推荐系统测试损失 (RMSE): {sqrt(deep_test_loss):.4f}")

# 6. 实时推荐
print("\n6. 实时推荐")

print("实时推荐是根据用户的实时行为进行推荐")
print("- 挑战：需要快速响应，处理实时数据")
print("- 方法：")
print("  - 在线学习：实时更新模型")
print("  - 缓存策略：缓存热门推荐")
print("  - 分布式计算：处理大规模数据")

# 7. 推荐系统评估
print("\n7. 推荐系统评估")

print("推荐系统的评估指标:")
print("- 离线评估：RMSE、MAE、Precision@k、Recall@k、F1@k、NDCG@k")
print("- 在线评估：点击率、转化率、用户停留时间")
print("- A/B测试：比较不同推荐算法的性能")

# 计算Precision@k和Recall@k
def precision_recall_at_k(model, test_df, k=5):
    """计算Precision@k和Recall@k"""
    user_precision = {}
    user_recall = {}
    
    # 按用户分组
    user_groups = test_df.groupby('user_id')
    
    for user_id, group in user_groups:
        # 获取用户未评分的物品
        rated_items = set(group['item_id'])
        all_items = set(items)
        unrated_items = list(all_items - rated_items)
        
        if not unrated_items:
            continue
        
        # 预测评分
        user_ids = np.array([user_id] * len(unrated_items))
        item_ids = np.array(unrated_items)
        predictions = model.predict([user_ids, item_ids]).flatten()
        
        # 排序并取前k个
        top_k_indices = np.argsort(predictions)[-k:]
        top_k_items = [unrated_items[i] for i in top_k_indices]
        
        # 计算相关物品（实际评分>=4的物品）
        relevant_items = set(group[group['rating'] >= 4]['item_id'])
        
        if relevant_items:
            # 计算Precision@k
            intersection = set(top_k_items) & relevant_items
            precision = len(intersection) / k
            user_precision[user_id] = precision
            
            # 计算Recall@k
            recall = len(intersection) / len(relevant_items)
            user_recall[user_id] = recall
    
    # 计算平均Precision@k和Recall@k
    avg_precision = np.mean(list(user_precision.values())) if user_precision else 0
    avg_recall = np.mean(list(user_recall.values())) if user_recall else 0
    
    return avg_precision, avg_recall

# 计算Precision@k和Recall@k
precision, recall = precision_recall_at_k(model, test_df, k=3)
print(f"\n矩阵分解神经网络:")
print(f"Precision@3: {precision:.4f}")
print(f"Recall@3: {recall:.4f}")

precision_deep, recall_deep = precision_recall_at_k(deep_model, test_df, k=3)
print(f"\n深度推荐系统:")
print(f"Precision@3: {precision_deep:.4f}")
print(f"Recall@3: {recall_deep:.4f}")

# 8. 推荐系统的挑战与解决方案
print("\n8. 推荐系统的挑战与解决方案")

print("推荐系统的挑战及解决方案:")
print("- 冷启动问题：")
print("  - 基于内容的推荐")
print("  - 流行度推荐")
print("  - 迁移学习")
print("- 数据稀疏性：")
print("  - 矩阵分解")
print("  - 深度学习")
print("  - 数据增强")
print("- 推荐多样性：")
print("  - 多样化推荐算法")
print("  - 重新排序")
print("  - 探索与利用平衡")
print("- 可扩展性：")
print("  - 分布式计算")
print("  - 模型压缩")
print("  - 缓存策略")

# 9. 推荐系统的未来趋势
print("\n9. 推荐系统的未来趋势")

print("推荐系统的未来趋势:")
print("- 深度学习：使用更复杂的神经网络结构")
print("- 图神经网络：建模用户和物品的关系")
print("- 强化学习：优化长期用户价值")
print("- 联邦学习：保护用户隐私")
print("- 多模态推荐：结合文本、图像、视频等多种数据")
print("- 可解释性：提高推荐的透明度")

# 10. 练习
print("\n10. 练习")

# 练习1: 矩阵分解
print("练习1: 矩阵分解")
print("- 实现SVD和NMF矩阵分解")
print("- 比较不同分解方法的性能")

# 练习2: 深度学习推荐系统
print("\n练习2: 深度学习推荐系统")
print("- 实现不同的深度学习推荐模型")
print("- 评估模型性能")

# 练习3: 实时推荐
print("\n练习3: 实时推荐")
print("- 实现简单的实时推荐系统")
print("- 测试系统的响应速度")

# 练习4: 推荐系统评估
print("\n练习4: 推荐系统评估")
print("- 实现不同的评估指标")
print("- 评估不同推荐算法的性能")

# 练习5: 推荐系统部署
print("\n练习5: 推荐系统部署")
print("- 构建完整的推荐系统")
print("- 部署到实际环境")

print("\n=== 第114天学习示例结束 ===")
