#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第55天：循环神经网络（RNN）
深度学习学习示例
内容：RNN的基本概念、结构、实现和应用
"""

print("=== 第55天：循环神经网络（RNN） ===")

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense, Embedding
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# 1. RNN概述
print("\n1. RNN概述")

print("循环神经网络（RNN）是一种专门用于处理序列数据的神经网络")
print("- 特点：具有记忆能力，能够处理可变长度的序列")
print("- 应用：自然语言处理、语音识别、时间序列预测等")
print("- 优势：能够捕捉序列数据中的依赖关系")

# 2. RNN的基本结构
print("\n2. RNN的基本结构")

print("RNN的基本结构包括:")
print("- 输入层：接收序列数据")
print("- 隐藏层：包含循环单元，具有记忆能力")
print("- 输出层：输出预测结果")
print("- 循环连接：隐藏层的输出作为下一个时间步的输入")

# 3. RNN的变体
print("\n3. RNN的变体")

print("常见的RNN变体:")
print("- SimpleRNN：基本的循环神经网络")
print("- LSTM：长短期记忆网络，解决梯度消失问题")
print("- GRU：门控循环单元，LSTM的简化版本")

# 4. 数据集准备
print("\n4. 数据集准备")

# 加载IMDB数据集
max_features = 10000  # 只考虑前10000个最常见的单词
maxlen = 100  # 每个序列的最大长度

print("加载IMDB数据集...")
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)
print(f"训练集形状: {X_train.shape}, {y_train.shape}")
print(f"测试集形状: {X_test.shape}, {y_test.shape}")

# 数据预处理
print("数据预处理...")
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)
print(f"预处理后训练集形状: {X_train.shape}")
print(f"预处理后测试集形状: {X_test.shape}")

# 5. 构建SimpleRNN模型
print("\n5. 构建SimpleRNN模型")

simple_rnn_model = Sequential([
    Embedding(max_features, 32, input_length=maxlen),
    SimpleRNN(32),
    Dense(1, activation='sigmoid')
])

# 编译模型
simple_rnn_model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 查看模型结构
simple_rnn_model.summary()

# 6. 构建LSTM模型
print("\n6. 构建LSTM模型")

lstm_model = Sequential([
    Embedding(max_features, 32, input_length=maxlen),
    LSTM(32),
    Dense(1, activation='sigmoid')
])

# 编译模型
lstm_model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 查看模型结构
lstm_model.summary()

# 7. 构建GRU模型
print("\n7. 构建GRU模型")

gru_model = Sequential([
    Embedding(max_features, 32, input_length=maxlen),
    GRU(32),
    Dense(1, activation='sigmoid')
])

# 编译模型
gru_model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 查看模型结构
gru_model.summary()

# 8. 模型训练
print("\n8. 模型训练")

# 训练SimpleRNN模型
print("训练SimpleRNN模型...")
simple_rnn_history = simple_rnn_model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# 训练LSTM模型
print("\n训练LSTM模型...")
lstm_history = lstm_model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# 训练GRU模型
print("\n训练GRU模型...")
gru_history = gru_model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# 9. 模型评估
print("\n9. 模型评估")

# 评估SimpleRNN模型
simple_rnn_loss, simple_rnn_accuracy = simple_rnn_model.evaluate(X_test, y_test, verbose=0)
print(f"SimpleRNN模型测试准确率: {simple_rnn_accuracy:.4f}")

# 评估LSTM模型
lstm_loss, lstm_accuracy = lstm_model.evaluate(X_test, y_test, verbose=0)
print(f"LSTM模型测试准确率: {lstm_accuracy:.4f}")

# 评估GRU模型
gru_loss, gru_accuracy = gru_model.evaluate(X_test, y_test, verbose=0)
print(f"GRU模型测试准确率: {gru_accuracy:.4f}")

# 10. 训练过程可视化
print("\n10. 训练过程可视化")

# 绘制训练历史
plt.figure(figsize=(15, 5))

# 准确率
plt.subplot(1, 3, 1)
plt.plot(simple_rnn_history.history['accuracy'], label='SimpleRNN训练')
plt.plot(simple_rnn_history.history['val_accuracy'], label='SimpleRNN验证')
plt.plot(lstm_history.history['accuracy'], label='LSTM训练')
plt.plot(lstm_history.history['val_accuracy'], label='LSTM验证')
plt.plot(gru_history.history['accuracy'], label='GRU训练')
plt.plot(gru_history.history['val_accuracy'], label='GRU验证')
plt.title('模型准确率')
plt.xlabel('epoch')
plt.ylabel('准确率')
plt.legend()
plt.grid()

# 损失
plt.subplot(1, 3, 2)
plt.plot(simple_rnn_history.history['loss'], label='SimpleRNN训练')
plt.plot(simple_rnn_history.history['val_loss'], label='SimpleRNN验证')
plt.plot(lstm_history.history['loss'], label='LSTM训练')
plt.plot(lstm_history.history['val_loss'], label='LSTM验证')
plt.plot(gru_history.history['loss'], label='GRU训练')
plt.plot(gru_history.history['val_loss'], label='GRU验证')
plt.title('模型损失')
plt.xlabel('epoch')
plt.ylabel('损失')
plt.legend()
plt.grid()

# 验证准确率
plt.subplot(1, 3, 3)
plt.plot(simple_rnn_history.history['val_accuracy'], label='SimpleRNN')
plt.plot(lstm_history.history['val_accuracy'], label='LSTM')
plt.plot(gru_history.history['val_accuracy'], label='GRU')
plt.title('验证准确率比较')
plt.xlabel('epoch')
plt.ylabel('准确率')
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig('rnn_training_history.png')
print("训练历史图已保存为 rnn_training_history.png")

# 11. 模型预测
print("\n11. 模型预测")

# 预测
lstm_predictions = lstm_model.predict(X_test)
predicted_classes = np.round(lstm_predictions).astype(int).reshape(-1)

# 打印预测结果
print("预测结果:")
for i in range(5):
    print(f"样本{i}: 预测={predicted_classes[i]}, 真实={y_test[i]}")

# 12. RNN的应用
print("\n12. RNN的应用")

print("RNN的主要应用:")
print("- 自然语言处理：文本分类、情感分析、机器翻译")
print("- 语音识别：将语音转换为文本")
print("- 时间序列预测：股票价格预测、天气预测")
print("- 视频分析：动作识别、视频分类")
print("- 推荐系统：基于用户历史行为的推荐")

# 13. RNN的挑战
print("\n13. RNN的挑战")

print("RNN的挑战:")
print("- 梯度消失：长序列训练困难")
print("- 计算效率：训练速度慢")
print("- 内存限制：难以处理长序列")
print("- 并行化困难：RNN的计算是顺序的")

# 14. RNN的改进
print("\n14. RNN的改进")

print("RNN的改进方法:")
print("- LSTM和GRU：解决梯度消失问题")
print("- 双向RNN：同时考虑过去和未来的信息")
print("- 多层RNN：增加模型深度")
print("- 注意力机制：关注序列中的重要部分")
print("- 批归一化：加速训练")

# 15. 练习
print("\n15. 练习")

# 练习1: 双向RNN
print("练习1: 双向RNN")
print("- 实现双向RNN模型")
print("- 比较双向RNN与单向RNN的性能")

# 练习2: 多层RNN
print("\n练习2: 多层RNN")
print("- 实现多层RNN模型")
print("- 分析层数对模型性能的影响")

# 练习3: 时间序列预测
print("\n练习3: 时间序列预测")
print("- 使用RNN预测时间序列数据")
print("- 评估预测效果")

# 练习4: 机器翻译
print("\n练习4: 机器翻译")
print("- 实现简单的机器翻译模型")
print("- 测试翻译效果")

# 练习5: 注意力机制
print("\n练习5: 注意力机制")
print("- 在RNN中添加注意力机制")
print("- 分析注意力机制对模型性能的影响")

print("\n=== 第55天学习示例结束 ===")
