#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第113天：推荐系统基础
推荐系统学习示例
内容：推荐系统的基本概念、协同过滤、内容过滤、混合推荐
"""

print("=== 第113天：推荐系统基础 ===")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import random

# 1. 推荐系统概述
print("\n1. 推荐系统概述")

print("推荐系统是一种信息过滤系统，用于预测用户对物品的偏好")
print("- 目标：为用户推荐可能感兴趣的物品")
print("- 应用：电商、音乐、视频、新闻等")
print("- 类型：协同过滤、内容过滤、混合推荐")

# 2. 数据集准备
print("\n2. 数据集准备")

# 创建示例数据集
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

# 转换为长格式
ratings_long = []
for user in users:
    for item in items:
        rating = ratings[user-1, item-1]
        if rating > 0:
            ratings_long.append({'user_id': user, 'item_id': item, 'rating': rating})

ratings_long_df = pd.DataFrame(ratings_long)
print("\n长格式评分数据:")
print(ratings_long_df.head())

# 3. 协同过滤
print("\n3. 协同过滤")

print("协同过滤是基于用户或物品的相似性进行推荐")
print("- 类型：基于用户的协同过滤（User-based CF）、基于物品的协同过滤（Item-based CF）")

# 3.1 基于用户的协同过滤
print("\n3.1 基于用户的协同过滤")

# 计算用户相似度
user_similarity = cosine_similarity(ratings)
user_similarity_df = pd.DataFrame(user_similarity, index=users, columns=users)
print("用户相似度矩阵:")
print(user_similarity_df)

# 基于用户相似度预测评分
def predict_user_based(user_id, item_id, ratings, user_similarity):
    """基于用户的协同过滤预测评分"""
    user_idx = user_id - 1
    item_idx = item_id - 1
    
    # 找出已评分该物品的用户
    rated_users = [i for i in range(len(ratings)) if ratings[i, item_idx] > 0 and i != user_idx]
    
    if not rated_users:
        return np.mean(ratings[user_idx, ratings[user_idx, :] > 0]) if np.sum(ratings[user_idx, :] > 0) > 0 else 3.0
    
    # 计算加权平均
    numerator = 0
    denominator = 0
    
    for i in rated_users:
        similarity = user_similarity[user_idx, i]
        numerator += similarity * ratings[i, item_idx]
        denominator += abs(similarity)
    
    return numerator / denominator if denominator > 0 else 3.0

# 测试基于用户的协同过滤
user_id = 1
item_id = 6
prediction = predict_user_based(user_id, item_id, ratings, user_similarity)
print(f"基于用户的协同过滤预测用户{user_id}对物品{item_id}的评分: {prediction:.2f}")

# 3.2 基于物品的协同过滤
print("\n3.2 基于物品的协同过滤")

# 计算物品相似度
item_similarity = cosine_similarity(ratings.T)
item_similarity_df = pd.DataFrame(item_similarity, index=items, columns=items)
print("物品相似度矩阵:")
print(item_similarity_df)

# 基于物品相似度预测评分
def predict_item_based(user_id, item_id, ratings, item_similarity):
    """基于物品的协同过滤预测评分"""
    user_idx = user_id - 1
    item_idx = item_id - 1
    
    # 找出用户已评分的物品
    rated_items = [i for i in range(len(ratings[user_idx])) if ratings[user_idx, i] > 0]
    
    if not rated_items:
        return np.mean(ratings[:, item_idx]) if np.sum(ratings[:, item_idx] > 0) > 0 else 3.0
    
    # 计算加权平均
    numerator = 0
    denominator = 0
    
    for i in rated_items:
        similarity = item_similarity[item_idx, i]
        numerator += similarity * ratings[user_idx, i]
        denominator += abs(similarity)
    
    return numerator / denominator if denominator > 0 else 3.0

# 测试基于物品的协同过滤
prediction = predict_item_based(user_id, item_id, ratings, item_similarity)
print(f"基于物品的协同过滤预测用户{user_id}对物品{item_id}的评分: {prediction:.2f}")

# 4. 内容过滤
print("\n4. 内容过滤")

print("内容过滤是基于物品的特征进行推荐")
print("- 原理：分析物品的特征，找到与用户历史偏好相似的物品")
print("- 优点：不依赖用户行为数据，没有冷启动问题")
print("- 缺点：推荐多样性差，无法发现用户的潜在兴趣")

# 创建物品特征
item_features = {
    1: {'action': 1, 'comedy': 0, 'drama': 0, 'scifi': 1},
    2: {'action': 1, 'comedy': 1, 'drama': 0, 'scifi': 0},
    3: {'action': 0, 'comedy': 1, 'drama': 1, 'scifi': 0},
    4: {'action': 0, 'comedy': 0, 'drama': 1, 'scifi': 0},
    5: {'action': 1, 'comedy': 0, 'drama': 0, 'scifi': 1},
    6: {'action': 0, 'comedy': 1, 'drama': 0, 'scifi': 1},
    7: {'action': 1, 'comedy': 1, 'drama': 1, 'scifi': 0},
    8: {'action': 0, 'comedy': 0, 'drama': 1, 'scifi': 1},
    9: {'action': 1, 'comedy': 0, 'drama': 1, 'scifi': 0},
    10: {'action': 0, 'comedy': 1, 'drama': 0, 'scifi': 0}
}

# 转换为DataFrame
item_features_df = pd.DataFrame.from_dict(item_features, orient='index')
print("物品特征矩阵:")
print(item_features_df)

# 计算用户偏好
user_preferences = {}
for user_id in users:
    user_idx = user_id - 1
    rated_items = [item_id for item_id in items if ratings[user_idx, item_id-1] > 0]
    if rated_items:
        # 基于用户评分计算偏好
        preference = np.zeros(len(item_features_df.columns))
        total_rating = 0
        for item_id in rated_items:
            rating = ratings[user_idx, item_id-1]
            preference += rating * item_features_df.loc[item_id].values
            total_rating += rating
        if total_rating > 0:
            preference /= total_rating
        user_preferences[user_id] = preference
    else:
        user_preferences[user_id] = np.zeros(len(item_features_df.columns))

print("\n用户偏好:")
for user_id, preference in user_preferences.items():
    print(f"用户{user_id}: {preference}")

# 基于内容过滤预测评分
def predict_content_based(user_id, item_id, user_preferences, item_features_df):
    """基于内容过滤预测评分"""
    user_preference = user_preferences[user_id]
    item_feature = item_features_df.loc[item_id].values
    
    # 计算相似度
    similarity = np.dot(user_preference, item_feature) / (
        np.linalg.norm(user_preference) * np.linalg.norm(item_feature) + 1e-10
    )
    
    # 映射到评分范围
    return 1 + 4 * similarity

# 测试基于内容过滤
prediction = predict_content_based(user_id, item_id, user_preferences, item_features_df)
print(f"基于内容过滤预测用户{user_id}对物品{item_id}的评分: {prediction:.2f}")

# 5. 混合推荐
print("\n5. 混合推荐")

print("混合推荐结合多种推荐方法")
print("- 方法：加权混合、切换混合、特征组合、层叠混合")
print("- 优点：综合不同方法的优势，提高推荐质量")

# 加权混合预测
def predict_hybrid(user_id, item_id, ratings, user_similarity, item_similarity, user_preferences, item_features_df, weights=[0.33, 0.33, 0.34]):
    """混合推荐预测评分"""
    pred_user = predict_user_based(user_id, item_id, ratings, user_similarity)
    pred_item = predict_item_based(user_id, item_id, ratings, item_similarity)
    pred_content = predict_content_based(user_id, item_id, user_preferences, item_features_df)
    
    # 加权平均
    return weights[0] * pred_user + weights[1] * pred_item + weights[2] * pred_content

# 测试混合推荐
prediction = predict_hybrid(user_id, item_id, ratings, user_similarity, item_similarity, user_preferences, item_features_df)
print(f"混合推荐预测用户{user_id}对物品{item_id}的评分: {prediction:.2f}")

# 6. 评估推荐系统
print("\n6. 评估推荐系统")

print("推荐系统的评估指标:")
print("- 均方根误差（RMSE）：衡量预测评分与实际评分的差异")
print("- 平均绝对误差（MAE）：衡量预测评分与实际评分的绝对差异")
print("- 准确率（Precision）：推荐列表中相关物品的比例")
print("- 召回率（Recall）：推荐列表中包含的相关物品占总相关物品的比例")
print("- F1分数：准确率和召回率的调和平均")

# 分割训练集和测试集
train_df, test_df = train_test_split(ratings_long_df, test_size=0.2, random_state=42)

# 构建训练集评分矩阵
train_ratings = np.zeros((len(users), len(items)))
for _, row in train_df.iterrows():
    user_idx = row['user_id'] - 1
    item_idx = row['item_id'] - 1
    train_ratings[user_idx, item_idx] = row['rating']

# 计算训练集的用户和物品相似度
train_user_similarity = cosine_similarity(train_ratings)
train_item_similarity = cosine_similarity(train_ratings.T)

# 计算训练集的用户偏好
train_user_preferences = {}
for user_id in users:
    user_idx = user_id - 1
    rated_items = [item_id for item_id in items if train_ratings[user_idx, item_id-1] > 0]
    if rated_items:
        preference = np.zeros(len(item_features_df.columns))
        total_rating = 0
        for item_id in rated_items:
            rating = train_ratings[user_idx, item_id-1]
            preference += rating * item_features_df.loc[item_id].values
            total_rating += rating
        if total_rating > 0:
            preference /= total_rating
        train_user_preferences[user_id] = preference
    else:
        train_user_preferences[user_id] = np.zeros(len(item_features_df.columns))

# 评估不同推荐方法
methods = ['User-based CF', 'Item-based CF', 'Content-based', 'Hybrid']
predictions = {}

for method in methods:
    predictions[method] = []

for _, row in test_df.iterrows():
    user_id = row['user_id']
    item_id = row['item_id']
    
    # 基于用户的协同过滤
    pred_user = predict_user_based(user_id, item_id, train_ratings, train_user_similarity)
    predictions['User-based CF'].append(pred_user)
    
    # 基于物品的协同过滤
    pred_item = predict_item_based(user_id, item_id, train_ratings, train_item_similarity)
    predictions['Item-based CF'].append(pred_item)
    
    # 基于内容的过滤
    pred_content = predict_content_based(user_id, item_id, train_user_preferences, item_features_df)
    predictions['Content-based'].append(pred_content)
    
    # 混合推荐
    pred_hybrid = predict_hybrid(user_id, item_id, train_ratings, train_user_similarity, train_item_similarity, train_user_preferences, item_features_df)
    predictions['Hybrid'].append(pred_hybrid)

# 计算RMSE
print("\n不同推荐方法的RMSE:")
y_true = test_df['rating'].values
for method in methods:
    rmse = sqrt(mean_squared_error(y_true, predictions[method]))
    print(f"{method}: {rmse:.4f}")

# 7. 推荐系统的挑战
print("\n7. 推荐系统的挑战")

print("推荐系统的主要挑战:")
print("- 冷启动问题：新用户或新物品没有历史数据")
print("- 数据稀疏性：用户只评分了少量物品")
print("- 推荐多样性：推荐结果过于相似")
print("- 推荐新颖性：推荐用户已经知道的物品")
print("- 可扩展性：处理大规模数据")

# 8. 推荐系统的应用
print("\n8. 推荐系统的应用")

print("推荐系统的主要应用:")
print("- 电商：推荐商品")
print("- 音乐：推荐歌曲")
print("- 视频：推荐电影、电视剧")
print("- 新闻：推荐新闻文章")
print("- 社交媒体：推荐好友、内容")
print("- 搜索引擎：推荐搜索结果")

# 9. 练习
print("\n9. 练习")

# 练习1: 协同过滤
print("练习1: 协同过滤")
print("- 实现基于用户和基于物品的协同过滤")
print("- 评估不同方法的性能")

# 练习2: 内容过滤
print("\n练习2: 内容过滤")
print("- 实现基于内容的过滤")
print("- 评估内容过滤的性能")

# 练习3: 混合推荐
print("\n练习3: 混合推荐")
print("- 实现不同的混合推荐方法")
print("- 比较不同混合方法的性能")

# 练习4: 冷启动问题
print("\n练习4: 冷启动问题")
print("- 实现解决冷启动问题的方法")
print("- 评估方法的效果")

# 练习5: 推荐系统部署
print("\n练习5: 推荐系统部署")
print("- 构建一个完整的推荐系统")
print("- 部署到实际环境")

print("\n=== 第113天学习示例结束 ===")
