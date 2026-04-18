#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第62天：语言模型
自然语言处理学习示例
内容：语言模型的基本概念、n-gram模型、RNN语言模型、Transformer
"""

print("=== 第62天：语言模型 ===")

import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import reuters
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input, Attention, Layer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import math

# 下载NLTK数据
nltk.download('reuters')

# 1. 语言模型概述
print("\n1. 语言模型概述")

print("语言模型是预测文本序列概率的模型")
print("- 目标：计算文本序列的概率")
print("- 应用：机器翻译、文本生成、语音识别等")
print("- 评估指标：困惑度（Perplexity）")

# 2. n-gram模型
print("\n2. n-gram模型")

print("n-gram模型是基于统计的语言模型")
print("- 原理：基于前n-1个词预测下一个词")
print("- 优点：简单，易于实现")
print("- 缺点：数据稀疏，无法捕捉长距离依赖")

# 准备语料库
corpus = reuters.sents()[:1000]  # 使用Reuters语料库的前1000个句子
print(f"语料库大小: {len(corpus)} 句子")

# 构建n-gram模型
class NGramModel:
    def __init__(self, n):
        self.n = n
        self.ngram_counts = {}
        self.context_counts = {}
    
    def train(self, corpus):
        for sentence in corpus:
            # 添加开始和结束标记
            tokens = ['<s>'] * (self.n-1) + sentence + ['</s>']
            for i in range(len(tokens) - self.n + 1):
                ngram = tuple(tokens[i:i+self.n])
                context = tuple(tokens[i:i+self.n-1])
                
                # 更新n-gram计数
                if ngram in self.ngram_counts:
                    self.ngram_counts[ngram] += 1
                else:
                    self.ngram_counts[ngram] = 1
                
                # 更新上下文计数
                if context in self.context_counts:
                    self.context_counts[context] += 1
                else:
                    self.context_counts[context] = 1
    
    def predict(self, context):
        """预测下一个词"""
        context = tuple(context)
        if context not in self.context_counts:
            return None
        
        # 计算所有可能的下一个词的概率
        predictions = {}
        for ngram in self.ngram_counts:
            if ngram[:-1] == context:
                next_word = ngram[-1]
                probability = self.ngram_counts[ngram] / self.context_counts[context]
                predictions[next_word] = probability
        
        # 按概率排序
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return sorted_predictions
    
    def calculate_perplexity(self, corpus):
        """计算困惑度"""
        total_log_prob = 0
        total_tokens = 0
        
        for sentence in corpus:
            tokens = ['<s>'] * (self.n-1) + sentence + ['</s>']
            for i in range(len(tokens) - self.n + 1):
                ngram = tuple(tokens[i:i+self.n])
                context = tuple(tokens[i:i+self.n-1])
                
                if ngram in self.ngram_counts and context in self.context_counts:
                    probability = self.ngram_counts[ngram] / self.context_counts[context]
                    total_log_prob += math.log(probability)
                else:
                    # 平滑处理
                    total_log_prob += math.log(1e-10)
                total_tokens += 1
        
        perplexity = math.exp(-total_log_prob / total_tokens)
        return perplexity

# 训练n-gram模型
print("训练n-gram模型...")
bigram_model = NGramModel(2)
bigram_model.train(corpus)

trigram_model = NGramModel(3)
trigram_model.train(corpus)

# 测试n-gram模型
print("\n测试n-gram模型:")
test_context = ['the']
bigram_predictions = bigram_model.predict(test_context)
print(f"Bigram模型预测 '{test_context}' 的下一个词: {bigram_predictions[:5]}")

test_context = ['the', 'company']
trigram_predictions = trigram_model.predict(test_context)
print(f"Trigram模型预测 '{test_context}' 的下一个词: {trigram_predictions[:5]}")

# 计算困惑度
print("\n计算困惑度:")
test_corpus = reuters.sents()[1000:1100]  # 使用测试集
bigram_perplexity = bigram_model.calculate_perplexity(test_corpus)
trigram_perplexity = trigram_model.calculate_perplexity(test_corpus)
print(f"Bigram模型困惑度: {bigram_perplexity:.4f}")
print(f"Trigram模型困惑度: {trigram_perplexity:.4f}")

# 3. RNN语言模型
print("\n3. RNN语言模型")

print("RNN语言模型使用循环神经网络来建模序列数据")
print("- 优点：能够捕捉长距离依赖")
print("- 缺点：训练速度慢，容易出现梯度消失")

# 准备数据
print("准备数据...")
# 合并所有句子
all_text = ' '.join([' '.join(sentence) for sentence in corpus])

# 分词
tokenizer = Tokenizer()
tokenizer.fit_on_texts([all_text])
vocab_size = len(tokenizer.word_index) + 1
print(f"词汇表大小: {vocab_size}")

# 创建序列数据
sequences = []
for sentence in corpus:
    token_list = tokenizer.texts_to_sequences([' '.join(sentence)])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        sequences.append(n_gram_sequence)

# 填充序列
max_sequence_length = max([len(seq) for seq in sequences])
sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='pre')
print(f"序列最大长度: {max_sequence_length}")

# 分割输入和输出
X, y = sequences[:, :-1], sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=vocab_size)

# 构建RNN语言模型
print("构建RNN语言模型...")
rnn_model = Sequential([
    Embedding(vocab_size, 100, input_length=max_sequence_length-1),
    LSTM(100),
    Dense(vocab_size, activation='softmax')
])

rnn_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

rnn_model.summary()

# 训练RNN语言模型
print("训练RNN语言模型...")
rnn_history = rnn_model.fit(
    X, y,
    epochs=10,
    batch_size=32,
    verbose=1
)

# 生成文本
print("\n生成文本:")
def generate_text(model, tokenizer, seed_text, max_length, num_words):
    generated_text = seed_text
    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([generated_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_length-1, padding='pre')
        predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        generated_text += " " + output_word
    return generated_text

seed_text = "the company"
generated_text = generate_text(rnn_model, tokenizer, seed_text, max_sequence_length, 10)
print(f"生成的文本: {generated_text}")

# 4. Transformer语言模型
print("\n4. Transformer语言模型")

print("Transformer语言模型使用自注意力机制来建模序列数据")
print("- 优点：并行计算，能够捕捉长距离依赖")
print("- 应用：BERT、GPT、T5等")

# 构建简单的Transformer模型
class PositionalEncoding(Layer):
    def __init__(self, max_seq_len, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.compute_positional_encoding(max_seq_len, d_model)
    
    def compute_positional_encoding(self, max_seq_len, d_model):
        pos = np.arange(max_seq_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        angle_rads = pos * angle_rates
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

def create_transformer_model(vocab_size, max_seq_len, d_model=64, n_heads=2, d_ff=128):
    inputs = Input(shape=(max_seq_len,))
    
    # Embedding
    x = Embedding(vocab_size, d_model)(inputs)
    
    # Positional Encoding
    x = PositionalEncoding(max_seq_len, d_model)(x)
    
    # Multi-Head Attention
    attention_output = Attention()([x, x, x])
    
    # Feed Forward
    x = Dense(d_ff, activation='relu')(attention_output)
    x = Dense(d_model)(x)
    
    # Output
    outputs = Dense(vocab_size, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# 创建Transformer模型
transformer_model = create_transformer_model(vocab_size, max_sequence_length-1)
transformer_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

transformer_model.summary()

# 5. 预训练语言模型
print("\n5. 预训练语言模型")

print("预训练语言模型是在大规模语料库上预先训练的模型")
print("- 类型：")
print("  - 自回归模型：GPT、GPT-2、GPT-3")
print("  - 自编码模型：BERT、RoBERTa、ALBERT")
print("  - 序列到序列模型：T5、BART")
print("- 优势：迁移学习，在下游任务上表现优异")

# 6. 语言模型的评估
print("\n6. 语言模型的评估")

print("语言模型的评估指标:")
print("- 困惑度（Perplexity）：衡量模型预测的不确定性")
print("- BLEU分数：衡量生成文本与参考文本的相似度")
print("- ROUGE分数：衡量自动摘要的质量")
print("- 人工评估：人类对生成文本的质量评估")

# 7. 语言模型的应用
print("\n7. 语言模型的应用")

print("语言模型的主要应用:")
print("- 文本生成：生成文章、诗歌、代码等")
print("- 机器翻译：将一种语言翻译为另一种语言")
print("- 问答系统：回答用户的问题")
print("- 文本摘要：生成文本的摘要")
print("- 情感分析：分析文本的情感倾向")
print("- 命名实体识别：识别文本中的实体")
print("- 语音识别：将语音转换为文本")
print("- 拼写检查：检查和纠正拼写错误")

# 8. 练习
print("\n8. 练习")

# 练习1: n-gram模型
print("练习1: n-gram模型")
print("- 实现不同n值的n-gram模型")
print("- 比较不同n值的困惑度")

# 练习2: RNN语言模型
print("\n练习2: RNN语言模型")
print("- 调整RNN语言模型的参数")
print("- 生成更长的文本")

# 练习3: Transformer模型
print("\n练习3: Transformer模型")
print("- 实现更复杂的Transformer模型")
print("- 训练并评估模型")

# 练习4: 预训练语言模型
print("\n练习4: 预训练语言模型")
print("- 使用Hugging Face Transformers加载预训练模型")
print("- 进行文本生成或分类任务")

# 练习5: 语言模型的应用
print("\n练习5: 语言模型的应用")
print("- 使用语言模型实现一个具体的应用")
print("- 评估应用的性能")

print("\n=== 第62天学习示例结束 ===")
