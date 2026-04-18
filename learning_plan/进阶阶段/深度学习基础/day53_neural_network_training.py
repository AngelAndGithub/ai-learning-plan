#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第53天：神经网络的训练与调优
深度学习学习示例
内容：神经网络训练的技巧、超参数调优、正则化、早停
"""

print("=== 第53天：神经网络的训练与调优 ===")

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 数据集准备
print("\n1. 数据集准备")

# 加载乳腺癌数据集
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
print(f"数据集形状: {X.shape}, {y.shape}")

# 数据预处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
print(f"训练集形状: {X_train.shape}, {y_train.shape}")
print(f"测试集形状: {X_test.shape}, {y_test.shape}")

# 2. 基础模型
print("\n2. 基础模型")

# 构建基础模型
base_model = Sequential([
    Dense(32, activation='relu', input_shape=(30,)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# 编译模型
base_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 训练基础模型
base_history = base_model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

# 评估基础模型
base_loss, base_accuracy = base_model.evaluate(X_test, y_test, verbose=0)
print(f"基础模型测试损失: {base_loss:.4f}")
print(f"基础模型测试准确率: {base_accuracy:.4f}")

# 3. 学习率调优
print("\n3. 学习率调优")

print("学习率对模型训练的影响")

# 尝试不同的学习率
learning_rates = [0.0001, 0.001, 0.01, 0.1]
histories = []

for lr in learning_rates:
    model = Sequential([
        Dense(32, activation='relu', input_shape=(30,)),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    
    histories.append(history)

# 绘制不同学习率的训练曲线
plt.figure(figsize=(12, 6))

# 准确率曲线
plt.subplot(1, 2, 1)
for i, lr in enumerate(learning_rates):
    plt.plot(histories[i].history['val_accuracy'], label=f'lr={lr}')
plt.title('不同学习率的验证准确率')
plt.xlabel('epoch')
plt.ylabel('准确率')
plt.legend()
plt.grid()

# 损失曲线
plt.subplot(1, 2, 2)
for i, lr in enumerate(learning_rates):
    plt.plot(histories[i].history['val_loss'], label=f'lr={lr}')
plt.title('不同学习率的验证损失')
plt.xlabel('epoch')
plt.ylabel('损失')
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig('learning_rate_comparison.png')
print("学习率比较图已保存为 learning_rate_comparison.png")

# 4. 正则化
print("\n4. 正则化")

print("正则化用于防止过拟合")

# L1正则化
l1_model = Sequential([
    Dense(32, activation='relu', kernel_regularizer=l1(0.001), input_shape=(30,)),
    Dense(16, activation='relu', kernel_regularizer=l1(0.001)),
    Dense(1, activation='sigmoid')
])

l1_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# L2正则化
l2_model = Sequential([
    Dense(32, activation='relu', kernel_regularizer=l2(0.001), input_shape=(30,)),
    Dense(16, activation='relu', kernel_regularizer=l2(0.001)),
    Dense(1, activation='sigmoid')
])

l2_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 训练正则化模型
l1_history = l1_model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

l2_history = l2_model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

# 评估正则化模型
l1_loss, l1_accuracy = l1_model.evaluate(X_test, y_test, verbose=0)
l2_loss, l2_accuracy = l2_model.evaluate(X_test, y_test, verbose=0)

print(f"L1正则化模型测试准确率: {l1_accuracy:.4f}")
print(f"L2正则化模型测试准确率: {l2_accuracy:.4f}")

# 5. Dropout
print("\n5. Dropout")

print("Dropout是一种有效的正则化方法")

# Dropout模型
dropout_model = Sequential([
    Dense(32, activation='relu', input_shape=(30,)),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

dropout_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 训练Dropout模型
dropout_history = dropout_model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

# 评估Dropout模型
dropout_loss, dropout_accuracy = dropout_model.evaluate(X_test, y_test, verbose=0)
print(f"Dropout模型测试准确率: {dropout_accuracy:.4f}")

# 6. 早停
print("\n6. 早停")

print("早停用于防止过拟合，提高模型泛化能力")

# 早停回调
earliestopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# 早停模型
earliestop_model = Sequential([
    Dense(32, activation='relu', input_shape=(30,)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

earliestop_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 训练早停模型
earliestop_history = earliestop_model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[earliestopping],
    verbose=0
)

# 评估早停模型
earliestop_loss, earliestop_accuracy = earliestop_model.evaluate(X_test, y_test, verbose=0)
print(f"早停模型测试准确率: {earliestop_accuracy:.4f}")
print(f"早停模型训练轮数: {len(earliestop_history.history['loss'])}")

# 7. 学习率调度器
print("\n7. 学习率调度器")

print("学习率调度器用于动态调整学习率")

# 定义学习率调度函数
def lr_scheduler(epoch, lr):
    if epoch % 20 == 0 and epoch > 0:
        return lr * 0.5
    return lr

# 学习率调度器回调
lr_schedule = LearningRateScheduler(lr_scheduler)

# 学习率调度模型
lr_model = Sequential([
    Dense(32, activation='relu', input_shape=(30,)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

lr_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 训练学习率调度模型
lr_history = lr_model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[lr_schedule],
    verbose=0
)

# 评估学习率调度模型
lr_loss, lr_accuracy = lr_model.evaluate(X_test, y_test, verbose=0)
print(f"学习率调度模型测试准确率: {lr_accuracy:.4f}")

# 8. 模型检查点
print("\n8. 模型检查点")

print("模型检查点用于保存最佳模型")

# 模型检查点回调
checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=0
)

# 模型检查点模型
checkpoint_model = Sequential([
    Dense(32, activation='relu', input_shape=(30,)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

checkpoint_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 训练模型检查点模型
checkpoint_history = checkpoint_model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[checkpoint],
    verbose=0
)

# 加载最佳模型
from tensorflow.keras.models import load_model
best_model = load_model('best_model.h5')

# 评估最佳模型
best_loss, best_accuracy = best_model.evaluate(X_test, y_test, verbose=0)
print(f"最佳模型测试准确率: {best_accuracy:.4f}")

# 9. 批量大小调优
print("\n9. 批量大小调优")

print("批量大小对模型训练的影响")

# 尝试不同的批量大小
batch_sizes = [16, 32, 64, 128]
batch_histories = []

for batch_size in batch_sizes:
    model = Sequential([
        Dense(32, activation='relu', input_shape=(30,)),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=0
    )
    
    batch_histories.append(history)

# 绘制不同批量大小的训练曲线
plt.figure(figsize=(12, 6))

# 准确率曲线
plt.subplot(1, 2, 1)
for i, batch_size in enumerate(batch_sizes):
    plt.plot(batch_histories[i].history['val_accuracy'], label=f'batch={batch_size}')
plt.title('不同批量大小的验证准确率')
plt.xlabel('epoch')
plt.ylabel('准确率')
plt.legend()
plt.grid()

# 损失曲线
plt.subplot(1, 2, 2)
for i, batch_size in enumerate(batch_sizes):
    plt.plot(batch_histories[i].history['val_loss'], label=f'batch={batch_size}')
plt.title('不同批量大小的验证损失')
plt.xlabel('epoch')
plt.ylabel('损失')
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig('batch_size_comparison.png')
print("批量大小比较图已保存为 batch_size_comparison.png")

# 10. 模型调优总结
print("\n10. 模型调优总结")

print("模型调优的主要方法:")
print("- 学习率调优：选择合适的学习率")
print("- 正则化：L1、L2正则化")
print("- Dropout：随机失活神经元")
print("- 早停：防止过拟合")
print("- 学习率调度：动态调整学习率")
print("- 批量大小调优：选择合适的批量大小")
print("- 模型检查点：保存最佳模型")

# 11. 模型性能比较
print("\n11. 模型性能比较")

print("不同模型的性能比较:")
print(f"基础模型: {base_accuracy:.4f}")
print(f"L1正则化模型: {l1_accuracy:.4f}")
print(f"L2正则化模型: {l2_accuracy:.4f}")
print(f"Dropout模型: {dropout_accuracy:.4f}")
print(f"早停模型: {earliestop_accuracy:.4f}")
print(f"学习率调度模型: {lr_accuracy:.4f}")
print(f"最佳模型: {best_accuracy:.4f}")

# 12. 练习
print("\n12. 练习")

# 练习1: 超参数调优
print("练习1: 超参数调优")
print("- 使用网格搜索或随机搜索调优超参数")
print("- 评估调优后的模型性能")

# 练习2: 集成学习
print("\n练习2: 集成学习")
print("- 实现模型集成")
print("- 比较集成模型与单一模型的性能")

# 练习3: 自定义回调
print("\n练习3: 自定义回调")
print("- 实现自定义回调函数")
print("- 用于监控训练过程")

# 练习4: 不同优化器比较
print("\n练习4: 不同优化器比较")
print("- 比较SGD、Adam、RMSprop等优化器")
print("- 分析不同优化器的性能")

# 练习5: 完整的模型调优流程
print("\n练习5: 完整的模型调优流程")
print("- 实现完整的模型调优流程")
print("- 从数据预处理到模型部署")

print("\n=== 第53天学习示例结束 ===")
