#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第52天：深度学习基础
深度学习学习示例
内容：深度学习的基本概念、神经网络基础、激活函数、损失函数
"""

print("=== 第52天：深度学习基础 ===")

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# 1. 深度学习概述
print("\n1. 深度学习概述")

print("深度学习是机器学习的一个分支，使用多层神经网络来学习数据的表示")
print("- 特点：能够自动学习特征，处理复杂数据")
print("- 应用：计算机视觉、自然语言处理、语音识别等")
print("- 优势：处理大规模数据，自动特征提取，性能优异")

# 2. 神经网络基础
print("\n2. 神经网络基础")

print("神经网络的基本组成:")
print("- 神经元：基本计算单元")
print("- 层：输入层、隐藏层、输出层")
print("- 权重和偏置：模型的参数")
print("- 激活函数：引入非线性")
print("- 前向传播：计算预测值")
print("- 反向传播：更新参数")

# 3. 激活函数
print("\n3. 激活函数")

print("激活函数的作用：引入非线性，使神经网络能够学习复杂函数")

# 定义激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def leaky_relu(x, alpha=0.01):
    return np.maximum(alpha * x, x)

# 绘制激活函数
x = np.linspace(-5, 5, 100)
plt.figure(figsize=(12, 8))

# Sigmoid
plt.subplot(2, 2, 1)
plt.plot(x, sigmoid(x))
plt.title('Sigmoid')
plt.grid()

# ReLU
plt.subplot(2, 2, 2)
plt.plot(x, relu(x))
plt.title('ReLU')
plt.grid()

# Tanh
plt.subplot(2, 2, 3)
plt.plot(x, tanh(x))
plt.title('Tanh')
plt.grid()

# Leaky ReLU
plt.subplot(2, 2, 4)
plt.plot(x, leaky_relu(x))
plt.title('Leaky ReLU')
plt.grid()

plt.tight_layout()
plt.savefig('activation_functions.png')
print("激活函数图已保存为 activation_functions.png")

# 4. 损失函数
print("\n4. 损失函数")

print("损失函数的作用：衡量模型预测与真实值之间的差异")
print("常见的损失函数:")
print("- 均方误差（MSE）：回归任务")
print("- 交叉熵损失：分类任务")
print("- 二元交叉熵：二分类任务")
print("- 稀疏分类交叉熵：多分类任务")

# 5. 优化器
print("\n5. 优化器")

print("优化器的作用：更新模型参数以最小化损失函数")
print("常见的优化器:")
print("- SGD：随机梯度下降")
print("- Adam：自适应矩估计")
print("- RMSprop：均方根传播")
print("- Adagrad：自适应梯度下降")

# 6. 神经网络的构建
print("\n6. 神经网络的构建")

print("使用TensorFlow/Keras构建神经网络")

# 加载iris数据集
iris = load_iris()
X, y = iris.data, iris.target

# 数据预处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 标签编码
encoder = OneHotEncoder(sparse=False)
y_encoded = encoder.fit_transform(y.reshape(-1, 1))

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)
print(f"训练集形状: {X_train.shape}, {y_train.shape}")
print(f"测试集形状: {X_test.shape}, {y_test.shape}")

# 构建神经网络模型
model = Sequential([
    Dense(10, activation='relu', input_shape=(4,)),
    Dense(10, activation='relu'),
    Dense(3, activation='softmax')
])

# 编译模型
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 查看模型结构
model.summary()

# 7. 模型训练
print("\n7. 模型训练")

# 训练模型
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    verbose=0
)

# 8. 模型评估
print("\n8. 模型评估")

# 评估模型
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"测试损失: {loss:.4f}")
print(f"测试准确率: {accuracy:.4f}")

# 9. 训练过程可视化
print("\n9. 训练过程可视化")

# 绘制训练历史
plt.figure(figsize=(12, 4))

# 准确率
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='训练准确率')
plt.plot(history.history['val_accuracy'], label='验证准确率')
plt.title('模型准确率')
plt.xlabel(' epoch')
plt.ylabel('准确率')
plt.legend()
plt.grid()

# 损失
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.title('模型损失')
plt.xlabel(' epoch')
plt.ylabel('损失')
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig('training_history.png')
print("训练历史图已保存为 training_history.png")

# 10. 模型预测
print("\n10. 模型预测")

# 预测
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

# 打印预测结果
print("预测结果:")
for i in range(5):
    print(f"样本{i}: 预测={iris.target_names[predicted_classes[i]]}, 真实={iris.target_names[true_classes[i]]}")

# 11. 神经网络的超参数
print("\n11. 神经网络的超参数")

print("神经网络的超参数:")
print("- 学习率：控制参数更新的步长")
print("- 批量大小：每次更新参数使用的样本数")
print("-  epoch数：训练的轮数")
print("- 隐藏层数量：网络的深度")
print("- 隐藏层神经元数量：网络的宽度")
print("- 激活函数：引入非线性")
print("- 优化器：参数更新的方法")
print("- 正则化：防止过拟合")

# 12. 过拟合与欠拟合
print("\n12. 过拟合与欠拟合")

print("过拟合与欠拟合的解决方法:")
print("- 过拟合：增加数据、正则化、 dropout、早停")
print("- 欠拟合：增加模型复杂度、调整超参数、改进特征工程")

# 13. 深度学习框架
print("\n13. 深度学习框架")

print("常用的深度学习框架:")
print("- TensorFlow：Google开发，功能强大")
print("- PyTorch：Facebook开发，动态计算图")
print("- Keras：高级API，易于使用")
print("- Caffe：BVLC开发，适合计算机视觉")
print("- MXNet：Amazon开发，高效")

# 14. 深度学习的挑战
print("\n14. 深度学习的挑战")

print("深度学习的挑战:")
print("- 数据需求：需要大量标注数据")
print("- 计算资源：需要GPU加速")
print("- 可解释性：模型黑盒特性")
print("- 过拟合：容易过拟合训练数据")
print("- 调参复杂：需要大量超参数调优")

# 15. 练习
print("\n15. 练习")

# 练习1: 构建不同结构的神经网络
print("练习1: 构建不同结构的神经网络")
print("- 尝试不同的隐藏层数量和神经元数量")
print("- 比较不同结构的性能")

# 练习2: 尝试不同的激活函数
print("\n练习2: 尝试不同的激活函数")
print("- 比较不同激活函数的效果")
print("- 分析激活函数对模型性能的影响")

# 练习3: 尝试不同的优化器
print("\n练习3: 尝试不同的优化器")
print("- 比较不同优化器的性能")
print("- 分析优化器对训练速度的影响")

# 练习4: 正则化
print("\n练习4: 正则化")
print("- 实现L1和L2正则化")
print("- 分析正则化对模型性能的影响")

# 练习5: 早停
print("\n练习5: 早停")
print("- 实现早停策略")
print("- 分析早停对模型性能的影响")

print("\n=== 第52天学习示例结束 ===")
