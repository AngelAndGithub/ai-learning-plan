import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 1. 基于用户的协同过滤
def user_based_cf():
    """基于用户的协同过滤"""
    # 创建示例数据
    data = {
        'user_id': [1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
        'item_id': [1, 2, 3, 1, 3, 2, 3, 4, 1, 4],
        'rating': [5, 4, 3, 4, 5, 5, 4, 3, 3, 5]
    }
    df = pd.DataFrame(data)
    
    # 创建用户-物品评分矩阵
    user_item_matrix = df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
    print("用户-物品评分矩阵:")
    print(user_item_matrix)
    
    # 计算用户相似度（余弦相似度）
    user_similarity = cosine_similarity(user_item_matrix)
    user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)
    print("\n用户相似度矩阵:")
    print(user_similarity_df)
    
    # 为用户1推荐物品
    target_user = 1
    similar_users = user_similarity_df[target_user].sort_values(ascending=False).index[1:3]  # 排除自己
    print(f"\n与用户{target_user}最相似的用户: {similar_users.tolist()}")
    
    # 找出相似用户喜欢但目标用户未评分的物品
    target_user_rated = user_item_matrix.loc[target_user][user_item_matrix.loc[target_user] > 0].index
    recommendations = []
    
    for user in similar_users:
        user_rated = user_item_matrix.loc[user][user_item_matrix.loc[user] > 0].index
        new_items = user_rated.difference(target_user_rated)
        for item in new_items:
            recommendations.append((item, user_item_matrix.loc[user, item]))
    
    # 去重并排序
    recommendations = list(set(recommendations))
    recommendations.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n为用户{target_user}推荐的物品:")
    for item, rating in recommendations:
        print(f"物品{item}，预测评分: {rating}")
    
    return user_item_matrix, user_similarity_df

# 2. 基于物品的协同过滤
def item_based_cf():
    """基于物品的协同过滤"""
    # 创建示例数据
    data = {
        'user_id': [1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
        'item_id': [1, 2, 3, 1, 3, 2, 3, 4, 1, 4],
        'rating': [5, 4, 3, 4, 5, 5, 4, 3, 3, 5]
    }
    df = pd.DataFrame(data)
    
    # 创建用户-物品评分矩阵
    user_item_matrix = df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
    
    # 计算物品相似度（余弦相似度）
    item_similarity = cosine_similarity(user_item_matrix.T)
    item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)
    print("物品相似度矩阵:")
    print(item_similarity_df)
    
    # 为用户1推荐物品
    target_user = 1
    target_user_rated = user_item_matrix.loc[target_user][user_item_matrix.loc[target_user] > 0].index
    
    # 找出与目标用户已评分物品相似的物品
    recommendations = []
    for item in target_user_rated:
        similar_items = item_similarity_df[item].sort_values(ascending=False).index[1:3]  # 排除自己
        for similar_item in similar_items:
            if similar_item not in target_user_rated:
                similarity = item_similarity_df.loc[item, similar_item]
                recommendations.append((similar_item, similarity))
    
    # 去重并排序
    recommendations = list(set(recommendations))
    recommendations.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n为用户{target_user}推荐的物品:")
    for item, similarity in recommendations:
        print(f"物品{item}，相似度: {similarity:.4f}")
    
    return user_item_matrix, item_similarity_df

# 3. 矩阵分解
def matrix_factorization():
    """矩阵分解"""
    # 创建示例数据
    data = {
        'user_id': [1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
        'item_id': [1, 2, 3, 1, 3, 2, 3, 4, 1, 4],
        'rating': [5, 4, 3, 4, 5, 5, 4, 3, 3, 5]
    }
    df = pd.DataFrame(data)
    
    # 创建用户-物品评分矩阵
    user_item_matrix = df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
    
    # 矩阵分解参数
    n_users = user_item_matrix.shape[0]
    n_items = user_item_matrix.shape[1]
    n_factors = 2
    learning_rate = 0.01
    n_epochs = 1000
    
    # 初始化用户和物品隐因子
    P = np.random.normal(scale=1./n_factors, size=(n_users, n_factors))
    Q = np.random.normal(scale=1./n_factors, size=(n_items, n_factors))
    
    # 训练
    for epoch in range(n_epochs):
        for i in range(n_users):
            for j in range(n_items):
                if user_item_matrix.iloc[i, j] > 0:
                    # 计算误差
                    e = user_item_matrix.iloc[i, j] - np.dot(P[i, :], Q[j, :])
                    # 更新隐因子
                    P[i, :] += learning_rate * e * Q[j, :]
                    Q[j, :] += learning_rate * e * P[i, :]
    
    # 预测评分
    predicted_matrix = np.dot(P, Q.T)
    predicted_df = pd.DataFrame(predicted_matrix, index=user_item_matrix.index, columns=user_item_matrix.columns)
    
    print("原始评分矩阵:")
    print(user_item_matrix)
    print("\n预测评分矩阵:")
    print(predicted_df)
    
    # 计算RMSE
    mask = user_item_matrix > 0
    rmse = np.sqrt(mean_squared_error(user_item_matrix[mask], predicted_df[mask]))
    print(f"\nRMSE: {rmse:.4f}")
    
    return P, Q, predicted_df

# 4. 基于内容的推荐
def content_based_recommendation():
    """基于内容的推荐"""
    # 创建示例数据
    items = {
        'item_id': [1, 2, 3, 4],
        'title': ['The Shawshank Redemption', 'The Godfather', 'The Dark Knight', 'Pulp Fiction'],
        'genre': ['Drama', 'Crime, Drama', 'Action, Crime, Drama', 'Crime, Drama']
    }
    items_df = pd.DataFrame(items)
    
    users = {
        'user_id': [1, 2, 3, 4],
        'liked_items': [[1, 3], [2, 4], [1, 2], [3, 4]]
    }
    users_df = pd.DataFrame(users)
    
    # 提取物品特征（使用TF-IDF）
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(items_df['genre'])
    item_features = pd.DataFrame(tfidf_matrix.toarray(), index=items_df['item_id'], columns=tfidf.get_feature_names_out())
    print("物品特征矩阵:")
    print(item_features)
    
    # 构建用户偏好profile
    user_profiles = {}
    for _, row in users_df.iterrows():
        user_id = row['user_id']
        liked_items = row['liked_items']
        user_profile = item_features.loc[liked_items].mean()
        user_profiles[user_id] = user_profile
    
    # 为用户1推荐物品
    target_user = 1
    target_user_profile = user_profiles[target_user]
    
    # 计算物品与用户偏好的相似度
    similarities = cosine_similarity([target_user_profile], item_features)[0]
    similarity_df = pd.DataFrame({'item_id': item_features.index, 'similarity': similarities})
    
    # 排除用户已喜欢的物品
    liked_items = users_df.loc[users_df['user_id'] == target_user, 'liked_items'].iloc[0]
    recommendations = similarity_df[~similarity_df['item_id'].isin(liked_items)].sort_values('similarity', ascending=False)
    
    print(f"\n为用户{target_user}推荐的物品:")
    for _, row in recommendations.iterrows():
        item_id = row['item_id']
        title = items_df.loc[items_df['item_id'] == item_id, 'title'].iloc[0]
        similarity = row['similarity']
        print(f"{title}，相似度: {similarity:.4f}")
    
    return item_features, user_profiles

# 5. 混合推荐系统
def hybrid_recommendation():
    """混合推荐系统"""
    # 基于用户的协同过滤结果
    user_cf_recs = [(4, 5.0), (2, 4.0)]
    
    # 基于内容的推荐结果
    content_recs = [(4, 0.8), (2, 0.6)]
    
    # 加权混合
    weights = {'user_cf': 0.6, 'content': 0.4}
    hybrid_scores = {}
    
    for item, score in user_cf_recs:
        hybrid_scores[item] = hybrid_scores.get(item, 0) + score * weights['user_cf']
    
    for item, score in content_recs:
        hybrid_scores[item] = hybrid_scores.get(item, 0) + score * weights['content']
    
    # 排序推荐
    hybrid_recs = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
    
    print("混合推荐结果:")
    for item, score in hybrid_recs:
        print(f"物品{item}，综合评分: {score:.4f}")
    
    return hybrid_recs

# 6. 深度学习推荐系统
def deep_learning_recommendation():
    """深度学习推荐系统"""
    # 创建示例数据
    n_users = 5
    n_items = 5
    n_samples = 20
    
    user_ids = np.random.randint(0, n_users, n_samples)
    item_ids = np.random.randint(0, n_items, n_samples)
    ratings = np.random.randint(1, 6, n_samples)
    
    # 构建模型
    user_input = layers.Input(shape=(1,))
    item_input = layers.Input(shape=(1,))
    
    # 用户嵌入
    user_embedding = layers.Embedding(n_users, 32)(user_input)
    user_embedding = layers.Flatten()(user_embedding)
    
    # 物品嵌入
    item_embedding = layers.Embedding(n_items, 32)(item_input)
    item_embedding = layers.Flatten()(item_embedding)
    
    # 合并特征
    concatenated = layers.Concatenate()([user_embedding, item_embedding])
    
    # 全连接层
    dense1 = layers.Dense(64, activation='relu')(concatenated)
    dense2 = layers.Dense(32, activation='relu')(dense1)
    output = layers.Dense(1, activation='linear')(dense2)
    
    # 创建模型
    model = models.Model(inputs=[user_input, item_input], outputs=output)
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    
    # 训练模型
    model.fit([user_ids, item_ids], ratings, epochs=10, batch_size=4)
    
    # 预测
    test_user = np.array([0])
    test_item = np.array([0])
    prediction = model.predict([test_user, test_item])
    print(f"\n用户0对物品0的预测评分: {prediction[0][0]:.2f}")
    
    return model

# 7. 推荐系统评估
def evaluate_recommendation():
    """推荐系统评估"""
    # 创建示例数据
    actual = [5, 4, 3, 5, 4]
    predicted = [4.5, 4.2, 3.1, 4.8, 3.9]
    
    # 计算RMSE
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    print(f"RMSE: {rmse:.4f}")
    
    # 计算MAE
    mae = np.mean(np.abs(np.array(actual) - np.array(predicted)))
    print(f"MAE: {mae:.4f}")
    
    # 准确率和召回率示例
    relevant = {1, 2, 3}
    recommended = {2, 3, 4}
    
    precision = len(relevant & recommended) / len(recommended)
    recall = len(relevant & recommended) / len(relevant)
    f1 = 2 * precision * recall / (precision + recall)
    
    print(f"准确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    
    return rmse, mae, precision, recall, f1

# 8. 冷启动问题处理
def cold_start_handling():
    """冷启动问题处理"""
    # 新用户冷启动
    print("新用户冷启动策略:")
    print("1. 基于人口统计学特征推荐")
    print("2. 推荐热门物品")
    print("3. 基于内容的推荐")
    print("4. 询问用户兴趣")
    
    # 新物品冷启动
    print("\n新物品冷启动策略:")
    print("1. 基于内容的推荐")
    print("2. 利用物品属性相似性")
    print("3. 推荐给有相似兴趣的用户")
    print("4. 探索与利用平衡")
    
    return "冷启动策略处理完成"

# 主函数
if __name__ == "__main__":
    print("=== 推荐系统示例 ===")
    
    # 1. 基于用户的协同过滤
    print("\n1. 基于用户的协同过滤")
    user_item_matrix, user_similarity_df = user_based_cf()
    
    # 2. 基于物品的协同过滤
    print("\n2. 基于物品的协同过滤")
    user_item_matrix, item_similarity_df = item_based_cf()
    
    # 3. 矩阵分解
    print("\n3. 矩阵分解")
    P, Q, predicted_df = matrix_factorization()
    
    # 4. 基于内容的推荐
    print("\n4. 基于内容的推荐")
    item_features, user_profiles = content_based_recommendation()
    
    # 5. 混合推荐系统
    print("\n5. 混合推荐系统")
    hybrid_recs = hybrid_recommendation()
    
    # 6. 深度学习推荐系统
    print("\n6. 深度学习推荐系统")
    try:
        model = deep_learning_recommendation()
        print("深度学习推荐系统创建完成")
    except Exception as e:
        print(f"深度学习推荐系统创建失败: {e}")
    
    # 7. 推荐系统评估
    print("\n7. 推荐系统评估")
    rmse, mae, precision, recall, f1 = evaluate_recommendation()
    
    # 8. 冷启动问题处理
    print("\n8. 冷启动问题处理")
    cold_start_result = cold_start_handling()
    
    print("\n=== 推荐系统示例完成 ===")
