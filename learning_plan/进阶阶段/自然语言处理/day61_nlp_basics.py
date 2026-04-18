#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第61天：自然语言处理基础
自然语言处理学习示例
内容：NLP的基本概念、文本预处理、词向量
"""

print("=== 第61天：自然语言处理基础 ===")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import string
import re

# 下载NLTK数据
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# 1. NLP概述
print("\n1. NLP概述")

print("自然语言处理（NLP）是让计算机理解和处理人类语言的技术")
print("- 目标：使计算机能够理解、生成和处理自然语言")
print("- 应用：文本分类、情感分析、机器翻译、问答系统等")
print("- 挑战：语言歧义、上下文依赖、语言多样性等")

# 2. 文本预处理
print("\n2. 文本预处理")

print("文本预处理是NLP的基础步骤")
print("- 步骤：分词、去停用词、词干提取、词形还原等")

# 示例文本
test_text = "Hello, world! This is a test sentence. I love natural language processing."
print(f"原始文本: {test_text}")

# 2.1 分词
print("\n2.1 分词")

# 单词分词
words = word_tokenize(test_text)
print(f"单词分词: {words}")

# 句子分词
sentences = sent_tokenize(test_text)
print(f"句子分词: {sentences}")

# 2.2 大小写转换
print("\n2.2 大小写转换")

lower_text = test_text.lower()
print(f"小写转换: {lower_text}")

# 2.3 去除标点符号
print("\n2.3 去除标点符号")

no_punct_text = test_text.translate(str.maketrans('', '', string.punctuation))
print(f"去除标点符号: {no_punct_text}")

# 2.4 去除数字
print("\n2.4 去除数字")

no_digits_text = re.sub(r'\d+', '', test_text)
print(f"去除数字: {no_digits_text}")

# 2.5 去除停用词
print("\n2.5 去除停用词")

stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word.lower() not in stop_words]
print(f"去除停用词: {filtered_words}")

# 2.6 词干提取
print("\n2.6 词干提取")

stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in filtered_words]
print(f"词干提取: {stemmed_words}")

# 2.7 词形还原
print("\n2.7 词形还原")

lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
print(f"词形还原: {lemmatized_words}")

# 3. 文本向量化
print("\n3. 文本向量化")

print("文本向量化是将文本转换为数值表示的过程")
print("- 方法：词袋模型、TF-IDF、词向量等")

# 示例文档
corpus = [
    "I love natural language processing",
    "I love machine learning",
    "I love artificial intelligence",
    "Natural language processing is interesting"
]

# 3.1 词袋模型
print("\n3.1 词袋模型")

vectorizer = CountVectorizer()
X_bow = vectorizer.fit_transform(corpus)
print(f"词袋模型特征: {vectorizer.get_feature_names()}")
print(f"词袋模型向量形状: {X_bow.shape}")
print(f"词袋模型向量:\n{X_bow.toarray()}")

# 3.2 TF-IDF
print("\n3.2 TF-IDF")

tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(corpus)
print(f"TF-IDF特征: {tfidf_vectorizer.get_feature_names()}")
print(f"TF-IDF向量形状: {X_tfidf.shape}")
print(f"TF-IDF向量:\n{X_tfidf.toarray()}")

# 4. 词向量
print("\n4. 词向量")

print("词向量是将单词表示为低维稠密向量的方法")
print("- 方法：Word2Vec、GloVe、FastText等")
print("- 特点：语义相似的单词在向量空间中距离较近")

# 示例：使用预训练的Word2Vec模型
# 注意：这里只是展示概念，实际使用需要下载预训练模型
print("Word2Vec模型可以将单词转换为向量，语义相似的单词在向量空间中距离较近")
print("例如：'king' - 'man' + 'woman' ≈ 'queen'")

# 5. 文本分类基础
print("\n5. 文本分类基础")

print("文本分类是NLP的基本任务之一")
print("- 任务：将文本分类到预定义的类别")
print("- 应用：情感分析、垃圾邮件检测、主题分类等")

# 6. 主题模型
print("\n6. 主题模型")

print("主题模型用于发现文本中的潜在主题")
print("- 方法：LDA（潜在狄利克雷分配）")
print("- 应用：文本聚类、信息检索等")

# 应用LDA
lda = LatentDirichletAllocation(n_components=2, random_state=42)
lda.fit(X_bow)

# 打印主题
print("\nLDA主题:")
for i, topic in enumerate(lda.components_):
    print(f"主题{i}: {[vectorizer.get_feature_names()[j] for j in topic.argsort()[-3:]]}")

# 7. NLP工具和库
print("\n7. NLP工具和库")

print("常用的NLP工具和库:")
print("- NLTK：自然语言工具包")
print("- spaCy：工业级NLP库")
print("- Gensim：主题建模和词向量")
print("- Hugging Face Transformers：预训练语言模型")
print("- Stanford CoreNLP：斯坦福大学开发的NLP工具")

# 8. NLP的挑战
print("\n8. NLP的挑战")

print("NLP的主要挑战:")
print("- 歧义性：一词多义")
print("- 上下文依赖：语言理解依赖上下文")
print("- 数据稀疏性：某些单词出现频率低")
print("- 语言多样性：不同语言、方言、风格")
print("- 标注数据稀缺：高质量标注数据难以获取")

# 9. NLP的应用
print("\n9. NLP的应用")

print("NLP的主要应用:")
print("- 情感分析：分析文本的情感倾向")
print("- 机器翻译：将一种语言翻译为另一种语言")
print("- 问答系统：回答用户的问题")
print("- 文本摘要：生成文本的摘要")
print("- 命名实体识别：识别文本中的实体")
print("- 关系抽取：抽取实体之间的关系")
print("- 语音识别：将语音转换为文本")
print("- 文本生成：生成自然语言文本")

# 10. 练习
print("\n10. 练习")

# 练习1: 文本预处理
print("练习1: 文本预处理")
print("- 下载一个文本数据集")
print("- 进行文本预处理")

# 练习2: 文本向量化
print("\n练习2: 文本向量化")
print("- 使用不同的向量化方法")
print("- 比较不同方法的效果")

# 练习3: 词向量
print("\n练习3: 词向量")
print("- 训练自己的词向量模型")
print("- 分析词向量的语义关系")

# 练习4: 主题模型
print("\n练习4: 主题模型")
print("- 使用LDA分析文本的主题")
print("- 可视化主题")

# 练习5: 文本分类
print("\n练习5: 文本分类")
print("- 构建文本分类模型")
print("- 评估模型性能")

print("\n=== 第61天学习示例结束 ===")
