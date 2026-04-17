#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第78天：词向量和语言模型
自然语言处理学习示例
内容：词向量的原理、语言模型的基本概念和应用
"""

print("=== 第78天：词向量和语言模型 ===")

# 1. 词向量的基本概念
print("\n1. 词向量的基本概念")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

print("词向量是将单词映射到低维向量空间的表示方法")
print("- 分布式表示：通过上下文学习单词的含义")
print("- 语义相似性：语义相似的单词在向量空间中距离较近")
print("- 维度：通常为100-300维")
print("- 训练方法：CBOW、Skip-gram")

# 2. Word2Vec模型
print("\n2. Word2Vec模型")

print("Word2Vec是一种流行的词向量训练方法")
print("- CBOW (Continuous Bag of Words)：根据上下文预测中心词")
print("- Skip-gram：根据中心词预测上下文")
print("- 负采样：通过负例来训练模型")
print("- 层次Softmax：加速训练")

# 3. 训练Word2Vec模型
print("\n3. 训练Word2Vec模型")

# 示例文本
corpus = [
    "I love natural language processing",
    "I love machine learning",
    "I love deep learning",
    "Natural language processing is interesting",
    "Machine learning is fun",
    "Deep learning is powerful"
]

# 分词
tokenized_corpus = [sentence.lower().split() for sentence in corpus]
print("分词后的语料:")
for sentence in tokenized_corpus:
    print(sentence)

# 训练Word2Vec模型
model = Word2Vec(
    tokenized_corpus,
    vector_size=100,
    window=2,
    min_count=1,
    sg=1  # 1 for Skip-gram, 0 for CBOW
)

print("\nWord2Vec模型已训练完成")
print(f"词汇表大小: {len(model.wv.index_to_key)}")
print(f"词汇表: {model.wv.index_to_key}")

# 4. 词向量的应用
print("\n4. 词向量的应用")

# 获取词向量
word = "learning"
vector = model.wv[word]
print(f"词 '{word}' 的向量形状: {vector.shape}")
print(f"词 '{word}' 的向量前10维: {vector[:10]}")

# 寻找相似词
similar_words = model.wv.most_similar(word, topn=5)
print(f"\n与 '{word}' 相似的词:")
for word, similarity in similar_words:
    print(f"{word}: {similarity:.4f}")

# 词向量运算
print("\n词向量运算:")
try:
    # 类比推理：king - man + woman = queen
    result = model.wv.most_similar(positive=['woman', 'king'], negative=['man'], topn=3)
    print("king - man + woman = ")
    for word, similarity in result:
        print(f"{word}: {similarity:.4f}")
except KeyError as e:
    print(f"词汇表中不存在某些单词: {e}")

# 计算词之间的相似度
word1 = "natural"
word2 = "machine"
similarity = model.wv.similarity(word1, word2)
print(f"\n'{word1}' 和 '{word2}' 之间的相似度: {similarity:.4f}")

# 5. 词向量可视化
print("\n5. 词向量可视化")

# 获取所有词向量
words = model.wv.index_to_key
vectors = np.array([model.wv[word] for word in words])

# 使用PCA降维到2维
pca = PCA(n_components=2)
vectors_pca = pca.fit_transform(vectors)

# 绘制词向量
plt.figure(figsize=(10, 8))
plt.scatter(vectors_pca[:, 0], vectors_pca[:, 1])

# 添加词标签
for i, word in enumerate(words):
    plt.annotate(word, (vectors_pca[i, 0], vectors_pca[i, 1]))

plt.title('Word2Vec Vectors Visualization (PCA)')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.show()

# 使用t-SNE降维
print("\n使用t-SNE降维:")
tsne = TSNE(n_components=2, random_state=42)
vectors_tsne = tsne.fit_transform(vectors)

plt.figure(figsize=(10, 8))
plt.scatter(vectors_tsne[:, 0], vectors_tsne[:, 1])

# 添加词标签
for i, word in enumerate(words):
    plt.annotate(word, (vectors_tsne[i, 0], vectors_tsne[i, 1]))

plt.title('Word2Vec Vectors Visualization (t-SNE)')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.show()

# 6. 预训练词向量
print("\n6. 预训练词向量")

print("常用的预训练词向量:")
print("1. Google News Word2Vec：基于Google News语料")
print("2. GloVe：基于全局词频统计")
print("3. fastText：考虑子词信息")
print("4. BERT：基于Transformer的预训练模型")

# 7. 语言模型
print("\n7. 语言模型")

print("语言模型是预测文本序列概率的模型")
print("- 统计语言模型：基于n-gram")
print("- 神经语言模型：基于神经网络")
print("- 预训练语言模型：BERT、GPT、XLNet等")

# 8. n-gram语言模型
print("\n8. n-gram语言模型")

from nltk.util import ngrams
from collections import defaultdict, Counter

# 构建n-gram模型
def build_ngram_model(corpus, n):
    """构建n-gram模型"""
    model = defaultdict(Counter)
    for sentence in corpus:
        tokens = sentence.lower().split()
        for ngram in ngrams(tokens, n):
            prefix = ngram[:-1]
            suffix = ngram[-1]
            model[prefix][suffix] += 1
    return model

# 计算条件概率
def get_probability(model, prefix, suffix):
    """计算条件概率 P(suffix | prefix)"""
    if prefix not in model:
        return 0.0
    total = sum(model[prefix].values())
    if total == 0:
        return 0.0
    return model[prefix][suffix] / total

# 测试n-gram模型
print("构建2-gram模型:")
model_2gram = build_ngram_model(corpus, 2)
print("2-gram模型:")
for prefix, suffixes in model_2gram.items():
    print(f"{prefix}: {dict(suffixes)}")

# 计算概率
prefix = ('i',)
suffix = 'love'
prob = get_probability(model_2gram, prefix, suffix)
print(f"\nP('{suffix}' | {prefix}) = {prob:.4f}")

# 9. 神经语言模型
print("\n9. 神经语言模型")

print("神经语言模型使用神经网络来预测下一个单词")
print("- 循环神经网络 (RNN)")
print("- 长短期记忆网络 (LSTM)")
print("- 门控循环单元 (GRU)")
print("- Transformer")

# 10. 预训练语言模型
print("\n10. 预训练语言模型")

print("预训练语言模型是在大规模语料上预训练的模型")
print("- BERT：双向编码器表示")
print("- GPT：生成式预训练变换器")
print("- RoBERTa：BERT的改进版本")
print("- XLNet：基于排列语言模型")
print("- ALBERT：轻量级BERT")

# 示例：使用Hugging Face Transformers
print("\n使用Hugging Face Transformers:")
print("from transformers import BertTokenizer, BertModel")
print("")
print("# 加载预训练模型和分词器")
print("tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')")
print("model = BertModel.from_pretrained('bert-base-uncased')")
print("")
print("# 编码文本")
print("text = """Hello, I love natural language processing""")
print("inputs = tokenizer(text, return_tensors='pt')")
print("")
print("# 获取隐藏状态")
print("outputs = model(**inputs)")
print("last_hidden_states = outputs.last_hidden_state")

# 11. 练习
print("\n11. 练习")

# 练习1: 训练Word2Vec模型
print("练习1: 训练Word2Vec模型")
print("- 使用更大的语料库")
print("- 调整Word2Vec的超参数")
print("- 评估词向量的质量")

# 练习2: 词向量应用
print("\n练习2: 词向量应用")
print("- 尝试不同的词向量运算")
print("- 可视化不同维度的词向量")
print("- 比较不同词向量模型的效果")

# 练习3: 语言模型
print("\n练习3: 语言模型")
print("- 尝试不同n值的n-gram模型")
print("- 实现简单的神经语言模型")
print("- 使用预训练语言模型")

# 练习4: 文本生成
print("\n练习4: 文本生成")
print("- 使用n-gram模型生成文本")
print("- 使用预训练语言模型生成文本")
print("- 评估生成文本的质量")

print("\n=== 第78天学习示例结束 ===")
