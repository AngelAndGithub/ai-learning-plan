#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第77天：自然语言处理基础
自然语言处理学习示例
内容：自然语言处理的基本概念、文本预处理和词向量
"""

print("=== 第77天：自然语言处理基础 ===")

# 1. 自然语言处理基本概念
print("\n1. 自然语言处理基本概念")

import nltk
import spacy
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

print("自然语言处理（NLP）是计算机科学和人工智能的一个分支，专注于计算机与人类语言的交互")
print("- 文本预处理：分词、词性标注、命名实体识别")
print("- 文本表示：词袋模型、TF-IDF、词嵌入")
print("- 文本分类：情感分析、垃圾邮件检测")
print("- 序列标注：命名实体识别、词性标注")
print("- 机器翻译：将一种语言翻译成另一种语言")
print("- 问答系统：回答用户的问题")
print("- 文本生成：生成文本内容")

# 2. 文本预处理
print("\n2. 文本预处理")

# 示例文本
text = "Hello, World! This is a sample text for natural language processing. It contains multiple sentences."
print(f"原始文本: {text}")

# 分词
print("\n分词:")
words = nltk.word_tokenize(text)
print(words)

# 词性标注
print("\n词性标注:")
tagged = nltk.pos_tag(words)
print(tagged)

# 命名实体识别
print("\n命名实体识别:")
entities = nltk.chunk.ne_chunk(tagged)
print(entities)

# 词干提取
print("\n词干提取:")
stemmer = nltk.PorterStemmer()
stems = [stemmer.stem(word) for word in words]
print(stems)

# 词形还原
print("\n词形还原:")
lemmatizer = nltk.WordNetLemmatizer()
lemmas = [lemmatizer.lemmatize(word) for word in words]
print(lemmas)

# 去除停用词
print("\n去除停用词:")
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word.lower() not in stop_words]
print(filtered_words)

# 3. 文本表示
print("\n3. 文本表示")

# 词袋模型
print("词袋模型:")
documents = [
    "I love natural language processing",
    "I love machine learning",
    "I love deep learning"
]

vectorizer = CountVectorizer()
x = vectorizer.fit_transform(documents)
print(f"特征名称: {vectorizer.get_feature_names_out()}")
print(f"词袋矩阵:\n{x.toarray()}")

# TF-IDF
print("\nTF-IDF:")
tfidf_vectorizer = TfidfVectorizer()
x_tfidf = tfidf_vectorizer.fit_transform(documents)
print(f"特征名称: {tfidf_vectorizer.get_feature_names_out()}")
print(f"TF-IDF矩阵:\n{x_tfidf.toarray()}")

# 4. 词频统计
print("\n4. 词频统计")

# 统计词频
word_counts = Counter(words)
print("词频统计:")
for word, count in word_counts.most_common(10):
    print(f"{word}: {count}")

# 绘制词频直方图
print("\n词频直方图:")
top_words = word_counts.most_common(10)
words_list, counts_list = zip(*top_words)

plt.figure(figsize=(10, 6))
plt.bar(words_list, counts_list)
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top 10 Most Frequent Words')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 5. 词向量
print("\n5. 词向量")

print("词向量是将单词映射到低维向量空间的表示方法")
print("- 独热编码：简单但维度高")
print("- Word2Vec：通过上下文学习词向量")
print("- GloVe：基于全局词频统计")
print("- fastText：考虑子词信息")
print("- BERT：基于Transformer的预训练词向量")

# 6. 使用预训练词向量
print("\n6. 使用预训练词向量")

print("使用Gensim加载预训练的Word2Vec模型")
print("步骤:")
print("1. 安装Gensim")
print("2. 下载预训练模型")
print("3. 加载模型")
print("4. 使用词向量")

# 示例代码
print("\nWord2Vec示例代码:")
print("from gensim.models import Word2Vec")
print("from gensim.models import KeyedVectors")
print("")
print("# 加载预训练模型")
print("model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)")
print("")
print("# 获取词向量")
print("vector = model['king']")
print("")
print("# 寻找相似词")
print("similar_words = model.most_similar('king')")
print("")
print("# 类比推理")
print("result = model.most_similar(positive=['king', 'woman'], negative=['man'])")

# 7. 文本分类
print("\n7. 文本分类")

print("文本分类是NLP的基本任务，用于将文本分为不同的类别")
print("- 情感分析：判断文本的情感倾向")
print("- 垃圾邮件检测：识别垃圾邮件")
print("- 主题分类：将文本归类到不同的主题")

# 示例：使用朴素贝叶斯进行文本分类
print("\n使用朴素贝叶斯进行文本分类:")
print("from sklearn.naive_bayes import MultinomialNB")
print("from sklearn.model_selection import train_test_split")
print("from sklearn.metrics import accuracy_score")
print("")
print("# 准备数据")
print("X = ["""I love this product! It's amazing.""", """This product is terrible. I hate it."""]")
print("y = [1, 0]  # 1: positive, 0: negative")
print("")
print("# 向量化")
print("vectorizer = CountVectorizer()")
print("X_vectorized = vectorizer.fit_transform(X)")
print("")
print("# 训练模型")
print("model = MultinomialNB()")
print("model.fit(X_vectorized, y)")
print("")
print("# 预测")
print("test_text = ["""I really like this product."""]")
print("test_vectorized = vectorizer.transform(test_text)")
print("prediction = model.predict(test_vectorized)")

# 8. 序列标注
print("\n8. 序列标注")

print("序列标注是为序列中的每个元素分配标签的任务")
print("- 词性标注：为每个单词标注词性")
print("- 命名实体识别：识别文本中的实体")
print("- 分词：将文本分割成单词")

# 9. 自然语言处理工具
print("\n9. 自然语言处理工具")

print("常用的NLP工具:")
print("1. NLTK：Natural Language Toolkit，Python的NLP库")
print("2. spaCy：工业级NLP库")
print("3. Stanford CoreNLP：斯坦福大学开发的NLP工具")
print("4. Gensim：主题建模和词向量")
print("5. Hugging Face Transformers：预训练语言模型")

# 10. 练习
print("\n10. 练习")

# 练习1: 文本预处理
print("练习1: 文本预处理")
print("- 尝试处理不同类型的文本")
print("- 尝试不同的分词方法")
print("- 尝试不同的词干提取和词形还原方法")

# 练习2: 文本表示
print("\n练习2: 文本表示")
print("- 尝试不同的文本表示方法")
print("- 比较词袋模型和TF-IDF的效果")
print("- 尝试使用词向量")

# 练习3: 文本分类
print("\n练习3: 文本分类")
print("- 尝试不同的分类算法")
print("- 评估分类性能")
print("- 尝试不同的特征提取方法")

# 练习4: 命名实体识别
print("\n练习4: 命名实体识别")
print("- 尝试使用不同的NER工具")
print("- 评估NER性能")
print("- 尝试自定义NER模型")

print("\n=== 第77天学习示例结束 ===")
