#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第113天：推荐系统基础
推荐系统学习示例
内容：推荐系统的基本概念、协同过滤和内容过滤
"""

print("=== 第113天：推荐系统基础 ===")

# 1. 推荐系统基本概念
print("\n1. 推荐系统基本概念")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

print("推荐系统是一种信息过滤系统，用于预测用户对物品的偏好")
print("- 协同过滤：基于用户或物品的相似性")
print("- 内容过滤：基于物品的特征")
print("- 混合推荐：结合多种方法")
print("- 常见应用：电商推荐、电影推荐、音乐推荐、新闻推荐")

# 2. 数据集准备
print("\n2. 数据集准备")

# 示例用户-物品评分矩阵
# 行：用户，列：物品
ratings = np.array([
    [5, 4, 0, 0, 1],
    [5, 0, 0, 4, 0],
    [0, 0, 5, 0, 4],
    [0, 3, 4, 0, 5],
    [0, 5, 0, 0, 4]
])

# 用户和物品名称
users = ['User1', 'User2', 'User3', 'User4', 'User5']
items = ['Item1', 'Item2', 'Item3', 'Item4', 'Item5']

# 转换为DataFrame
ratings_df = pd.DataFrame(ratings, index=users, columns=items)
print("用户-物品评分矩阵:")
print(ratings_df)

# 3. 协同过滤
print("\n3. 协同过滤")

print("协同过滤分为基于用户的协同过滤和基于物品的协同过滤")

# 3.1 基于用户的协同过滤
print("\n3.1 基于用户的协同过滤")

# 计算用户之间的相似度
user_similarity = cosine_similarity(ratings)
user_similarity_df = pd.DataFrame(user_similarity, index=users, columns=users)
print("用户相似度矩阵:")
print(user_similarity_df)

# 为用户1推荐物品
target_user = 'User1'
target_user_idx = users.index(target_user)

# 找到最相似的用户
similar_users = user_similarity[target_user_idx].argsort()[::-1][1:3]  # 排除自己，取前2个
print(f"\n与 {target_user} 最相似的用户:")
for idx in similar_users:
    print(f"{users[idx]}: {user_similarity[target_user_idx][idx]:.4f}")

# 推荐物品
print("\n推荐给 User1 的物品:")
target_user_ratings = ratings[target_user_idx]
unrated_items = np.where(target_user_ratings == 0)[0]

for item_idx in unrated_items:
    # 计算加权评分
    weighted_sum = 0
    similarity_sum = 0
    for user_idx in similar_users:
        if ratings[user_idx][item_idx] > 0:
            weighted_sum += user_similarity[target_user_idx][user_idx] * ratings[user_idx][item_idx]
            similarity_sum += user_similarity[target_user_idx][user_idx]
    
    if similarity_sum > 0:
        predicted_rating = weighted_sum / similarity_sum
        print(f"{items[item_idx]}: 预测评分 = {predicted_rating:.4f}")

# 3.2 基于物品的协同过滤
print("\n3.2 基于物品的协同过滤")

# 计算物品之间的相似度
item_similarity = cosine_similarity(ratings.T)
item_similarity_df = pd.DataFrame(item_similarity, index=items, columns=items)
print("物品相似度矩阵:")
print(item_similarity_df)

# 为用户1推荐物品
target_user = 'User1'
target_user_idx = users.index(target_user)
target_user_ratings = ratings[target_user_idx]

print("\n基于物品的推荐给 User1 的物品:")
unrated_items = np.where(target_user_ratings == 0)[0]

for item_idx in unrated_items:
    # 计算加权评分
    weighted_sum = 0
    similarity_sum = 0
    for rated_item_idx in np.where(target_user_ratings > 0)[0]:
        similarity = item_similarity[item_idx][rated_item_idx]
        weighted_sum += similarity * target_user_ratings[rated_item_idx]
        similarity_sum += similarity
    
    if similarity_sum > 0:
        predicted_rating = weighted_sum / similarity_sum
        print(f"{items[item_idx]}: 预测评分 = {predicted_rating:.4f}")

# 4. 内容过滤
print("\n4. 内容过滤")

# 物品特征
item_features = {
    'Item1': [1, 0, 1],  # 特征1, 特征2, 特征3
    'Item2': [0, 1, 1],
    'Item3': [1, 1, 0],
    'Item4': [0, 1, 0],
    'Item5': [1, 0, 0]
}

# 转换为DataFrame
item_features_df = pd.DataFrame(item_features).T
item_features_df.columns = ['Feature1', 'Feature2', 'Feature3']
print("物品特征矩阵:")
print(item_features_df)

# 用户偏好
user_preferences = {
    'User1': [1, 0, 1],  # 对特征的偏好
    'User2': [0, 1, 1],
    'User3': [1, 1, 0],
    'User4': [0, 1, 0],
    'User5': [1, 0, 0]
}

# 转换为DataFrame
user_preferences_df = pd.DataFrame(user_preferences).T
user_preferences_df.columns = ['Feature1', 'Feature2', 'Feature3']
print("\n用户偏好矩阵:")
print(user_preferences_df)

# 计算用户对物品的评分
print("\n基于内容的评分预测:")
for user in users:
    print(f"\n{user} 的预测评分:")
    user_pref = user_preferences[user]
    for item in items:
        item_feat = item_features[item]
        # 计算点积作为评分
        score = np.dot(user_pref, item_feat)
        print(f"{item}: {score}")

# 5. 矩阵分解
print("\n5. 矩阵分解")

print("矩阵分解是推荐系统中常用的方法，将用户-物品矩阵分解为用户矩阵和物品矩阵")
print("- 奇异值分解 (SVD)")
print("- 非负矩阵分解 (NMF)")
print("- 隐因子模型")

# 6. 评估指标
print("\n6. 评估指标")

print("推荐系统的评估指标:")
print("1. 均方误差 (MSE): 预测评分与实际评分的均方差")
print("2. 平均绝对误差 (MAE): 预测评分与实际评分的平均绝对差")
print("3. 准确率 (Precision): 推荐的物品中用户喜欢的比例")
print("4. 召回率 (Recall): 用户喜欢的物品中被推荐的比例")
print("5. F1分数: 准确率和召回率的调和平均")
print("6. NDCG (Normalized Discounted Cumulative Gain): 考虑推荐顺序的评估指标")

# 7. 冷启动问题
print("\n7. 冷启动问题")

print("冷启动问题是推荐系统面临的挑战")
print("- 新用户冷启动：新用户没有历史行为")
print("- 新物品冷启动：新物品没有被评分")
print("解决方案:")
print("- 基于内容的推荐")
print("- 利用用户 demographic 信息")
print("- 流行度推荐")
print("- 探索与利用平衡")

# 8. 推荐系统的挑战
print("\n8. 推荐系统的挑战")

print("推荐系统面临的挑战:")
print("1. 冷启动问题")
print("2. 数据稀疏性")
print("3. 可扩展性")
print("4. 多样性与准确性的平衡")
print("5. 隐私保护")
print("6. 实时性")

# 9. 推荐系统的应用
print("\n9. 推荐系统的应用")

print("推荐系统的常见应用:")
print("1. 电商平台：推荐商品")
print("2. 视频平台：推荐电影、电视剧")
print("3. 音乐平台：推荐歌曲")
print("4. 新闻平台：推荐新闻")
print("5. 社交平台：推荐好友、内容")
print("6. 搜索引擎：推荐搜索结果")

# 10. 练习
print("\n10. 练习")

# 练习1: 实现协同过滤
print("练习1: 实现协同过滤")
print("- 实现基于用户的协同过滤")
print("- 实现基于物品的协同过滤")
print("- 比较两种方法的效果")

# 练习2: 矩阵分解
print("\n练习2: 矩阵分解")
print("- 使用SVD分解用户-物品矩阵")
print("- 使用NMF分解用户-物品矩阵")
print("- 比较分解后的推荐效果")

# 练习3: 混合推荐
print("\n练习3: 混合推荐")
print("- 结合协同过滤和内容过滤")
print("- 设计混合推荐策略")
print("- 评估混合推荐的效果")

# 练习4: 评估推荐系统
print("\n练习4: 评估推荐系统")
print("- 实现不同的评估指标")
print("- 评估推荐系统的性能")
print("- 分析评估结果")

print("\n=== 第113天学习示例结束 ===")
