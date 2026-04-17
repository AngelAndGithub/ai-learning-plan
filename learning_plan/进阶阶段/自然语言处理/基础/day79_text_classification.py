#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第79天：文本分类
自然语言处理学习示例
内容：文本分类的基本概念、常见算法和应用
"""

print("=== 第79天：文本分类 ===")

# 1. 文本分类基本概念
print("\n1. 文本分类基本概念")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

print("文本分类是将文本分为不同类别的任务")
print("- 二分类：将文本分为两个类别（如垃圾邮件检测）")
print("- 多分类：将文本分为多个类别（如新闻分类）")
print("- 多标签分类：一个文本可以属于多个类别")
print("- 常见应用：情感分析、垃圾邮件检测、新闻分类、主题分类")

# 2. 数据集准备
print("\n2. 数据集准备")

# 示例数据集
data = {
    'text': [
        'I love this product! It\'s amazing.',
        'This product is terrible. I hate it.',
        'The product is okay, not great but not bad.',
        'I really enjoy using this product.',
        'This is the worst product I\'ve ever bought.',
        'The product works as expected.',
        'I\'m very satisfied with this purchase.',
        'I regret buying this product.'
    ],
    'label': ['positive', 'negative', 'neutral', 'positive', 'negative', 'neutral', 'positive', 'negative']
}

df = pd.DataFrame(data)
print("示例数据集:")
print(df)

# 标签编码
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])
print("\n标签编码:")
print(df[['label', 'label_encoded']])
print(f"标签映射: {dict(zip(le.classes_, le.transform(le.classes_))}")

# 划分训练集和测试集
X = df['text']
y = df['label_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print(f"\n训练集大小: {len(X_train)}")
print(f"测试集大小: {len(X_test)}")

# 3. 特征提取
print("\n3. 特征提取")

# 词袋模型
print("词袋模型:")
vectorizer = CountVectorizer()
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

print(f"特征数量: {len(vectorizer.get_feature_names_out())}")
print(f"训练集形状: {X_train_bow.shape}")
print(f"测试集形状: {X_test_bow.shape}")
print(f"特征名称: {vectorizer.get_feature_names_out()}")

# TF-IDF
print("\nTF-IDF:")
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print(f"特征数量: {len(tfidf_vectorizer.get_feature_names_out())}")
print(f"训练集形状: {X_train_tfidf.shape}")
print(f"测试集形状: {X_test_tfidf.shape}")

# 4. 模型训练和评估
print("\n4. 模型训练和评估")

# 朴素贝叶斯分类器
print("朴素贝叶斯分类器:")
nb_model = MultinomialNB()
nb_model.fit(X_train_bow, y_train)
y_pred_nb = nb_model.predict(X_test_bow)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print(f"准确率: {accuracy_nb:.4f}")
print("分类报告:")
print(classification_report(y_test, y_pred_nb, target_names=le.classes_))
print("混淆矩阵:")
print(confusion_matrix(y_test, y_pred_nb))

# 支持向量机
print("\n支持向量机:")
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_tfidf, y_train)
y_pred_svm = svm_model.predict(X_test_tfidf)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"准确率: {accuracy_svm:.4f}")
print("分类报告:")
print(classification_report(y_test, y_pred_svm, target_names=le.classes_))

# 逻辑回归
print("\n逻辑回归:")
lr_model = LogisticRegression()
lr_model.fit(X_train_tfidf, y_train)
y_pred_lr = lr_model.predict(X_test_tfidf)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"准确率: {accuracy_lr:.4f}")
print("分类报告:")
print(classification_report(y_test, y_pred_lr, target_names=le.classes_))

# 5. 管道
print("\n5. 管道")

# 创建管道
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', LogisticRegression())
])

# 训练管道
pipeline.fit(X_train, y_train)

# 评估管道
y_pred_pipeline = pipeline.predict(X_test)
accuracy_pipeline = accuracy_score(y_test, y_pred_pipeline)
print(f"管道准确率: {accuracy_pipeline:.4f}")

# 6. 超参数调优
print("\n6. 超参数调优")

from sklearn.model_selection import GridSearchCV

# 定义参数网格
param_grid = {
    'vectorizer__max_features': [100, 500, 1000],
    'vectorizer__ngram_range': [(1, 1), (1, 2)],
    'classifier__C': [0.1, 1, 10]
}

# 网格搜索
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证准确率: {grid_search.best_score_:.4f}")

# 使用最佳模型
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f"最佳模型测试准确率: {accuracy_best:.4f}")

# 7. 文本分类的挑战
print("\n7. 文本分类的挑战")

print("文本分类面临的挑战:")
print("1. 数据不平衡：某些类别的样本数量远多于其他类别")
print("2. 特征选择：选择有效的特征")
print("3. 模型选择：选择适合任务的模型")
print("4. 过拟合：模型在训练数据上表现良好，但在测试数据上表现差")
print("5. 文本预处理：处理噪声、拼写错误等")

# 8. 深度学习方法
print("\n8. 深度学习方法")

print("用于文本分类的深度学习方法:")
print("1. 卷积神经网络 (CNN)")
print("2. 循环神经网络 (RNN/LSTM/GRU)")
print("3. Transformer")
print("4. 预训练模型 (BERT, RoBERTa, etc.)")

# 示例：使用BERT进行文本分类
print("\n使用BERT进行文本分类:")
print("from transformers import BertTokenizer, BertForSequenceClassification")
print("from transformers import Trainer, TrainingArguments")
print("")
print("# 加载模型和分词器")
print("tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')")
print("model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)")
print("")
print("# 准备数据集")
print("# 训练模型")
print("# 评估模型")

# 9. 情感分析
print("\n9. 情感分析")

print("情感分析是文本分类的一个重要应用")
print("- 二分类：正面/负面")
print("- 多分类：正面/中性/负面")
print("- 情感强度：评分1-5星")

# 10. 练习
print("\n10. 练习")

# 练习1: 不同特征提取方法
print("练习1: 不同特征提取方法")
print("- 尝试不同的n-gram范围")
print("- 尝试不同的停用词处理")
print("- 尝试不同的特征选择方法")

# 练习2: 不同分类算法
print("\n练习2: 不同分类算法")
print("- 尝试决策树")
print("- 尝试随机森林")
print("- 尝试梯度提升树")

# 练习3: 处理不平衡数据
print("\n练习3: 处理不平衡数据")
print("- 尝试过采样")
print("- 尝试欠采样")
print("- 尝试类别权重")

# 练习4: 深度学习方法
print("\n练习4: 深度学习方法")
print("- 尝试使用CNN进行文本分类")
print("- 尝试使用LSTM进行文本分类")
print("- 尝试使用BERT进行文本分类")

print("\n=== 第79天学习示例结束 ===")
