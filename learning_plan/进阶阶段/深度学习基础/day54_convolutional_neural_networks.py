#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第54天：卷积神经网络（CNN）
深度学习学习示例
内容：CNN的基本概念、结构、实现和应用
"""

print("=== 第54天：卷积神经网络（CNN） ===")

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. CNN概述
print("\n1. CNN概述")

print("卷积神经网络（CNN）是一种专门用于处理具有网格结构数据的神经网络")
print("- 特点：局部连接、权值共享、平移不变性")
print("- 应用：图像处理、计算机视觉、语音识别等")
print("- 优势：能够自动学习图像的特征，减少参数数量")

# 2. CNN的基本结构
print("\n2. CNN的基本结构")

print("CNN的基本结构包括:")
print("- 输入层：接收图像数据")
print("- 卷积层：提取图像特征")
print("- 池化层：降维，减少计算量")
print("- 全连接层：分类")
print("- 输出层：输出预测结果")

# 3. 卷积层
print("\n3. 卷积层")

print("卷积层的作用：提取图像的特征")
print("- 卷积核：用于提取特征的过滤器")
print("- 步长：卷积核移动的步长")
print("- 填充：保持输出特征图的大小")
print("- 激活函数：引入非线性")

# 4. 池化层
print("\n4. 池化层")

print("池化层的作用：降维，减少计算量")
print("- 最大池化：取区域内的最大值")
print("- 平均池化：取区域内的平均值")
print("- 池化大小：通常为2x2")
print("- 步长：通常与池化大小相同")

# 5. 数据集准备
print("\n5. 数据集准备")

# 加载MNIST数据集
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(f"训练集形状: {X_train.shape}, {y_train.shape}")
print(f"测试集形状: {X_test.shape}, {y_test.shape}")

# 数据预处理
# 调整数据形状
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# 归一化
X_train = X_train / 255.0
X_test = X_test / 255.0

# 标签编码
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print(f"预处理后训练集形状: {X_train.shape}, {y_train.shape}")
print(f"预处理后测试集形状: {X_test.shape}, {y_test.shape}")

# 6. 构建CNN模型
print("\n6. 构建CNN模型")

# 构建基础CNN模型
model = Sequential([
    # 第一层卷积
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    
    # 第二层卷积
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # 第三层卷积
    Conv2D(64, (3, 3), activation='relu'),
    
    # 展平
    Flatten(),
    
    # 全连接层
    Dense(64, activation='relu'),
    Dropout(0.5),
    
    # 输出层
    Dense(10, activation='softmax')
])

# 查看模型结构
model.summary()

# 编译模型
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 7. 模型训练
print("\n7. 模型训练")

# 数据增强
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

datagen.fit(X_train)

# 训练模型
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=10,
    validation_data=(X_test, y_test),
    verbose=1
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
plt.xlabel('epoch')
plt.ylabel('准确率')
plt.legend()
plt.grid()

# 损失
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.title('模型损失')
plt.xlabel('epoch')
plt.ylabel('损失')
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig('cnn_training_history.png')
print("训练历史图已保存为 cnn_training_history.png")

# 10. 模型预测
print("\n10. 模型预测")

# 预测
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

# 打印预测结果
print("预测结果:")
for i in range(5):
    print(f"样本{i}: 预测={predicted_classes[i]}, 真实={true_classes[i]}")

# 可视化预测结果
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_test[i].reshape(28, 28), cmap=plt.cm.binary)
    plt.xlabel(f"预测: {predicted_classes[i]}\n真实: {true_classes[i]}")
plt.tight_layout()
plt.savefig('cnn_predictions.png')
print("预测结果图已保存为 cnn_predictions.png")

# 11. CNN的变体
print("\n11. CNN的变体")

print("常见的CNN变体:")
print("- LeNet-5：最早的CNN模型")
print("- AlexNet：深层CNN模型")
print("- VGG：更深层的CNN模型")
print("- GoogLeNet/Inception：使用Inception模块")
print("- ResNet：使用残差连接")
print("- MobileNet：轻量级CNN模型")

# 12. CNN的应用
print("\n12. CNN的应用")

print("CNN的主要应用:")
print("- 图像分类：识别图像中的物体")
print("- 目标检测：定位图像中的物体")
print("- 图像分割：分割图像中的不同区域")
print("- 人脸识别：识别图像中的人脸")
print("- 图像生成：生成新的图像")
print("- 风格迁移：将一种图像风格应用到另一种图像")

# 13. CNN的性能优化
print("\n13. CNN的性能优化")

print("CNN的性能优化方法:")
print("- 批归一化：加速训练，提高模型性能")
print("- 残差连接：解决深层网络的梯度消失问题")
print("- 迁移学习：利用预训练模型")
print("- 模型量化：减少模型大小")
print("- 剪枝：减少模型参数")
print("- 硬件加速：使用GPU或TPU")

# 14. 练习
print("\n14. 练习")

# 练习1: 构建不同结构的CNN
print("练习1: 构建不同结构的CNN")
print("- 尝试不同的卷积层数量和通道数")
print("- 比较不同结构的性能")

# 练习2: 数据增强
print("\n练习2: 数据增强")
print("- 尝试不同的数据增强方法")
print("- 分析数据增强对模型性能的影响")

# 练习3: 迁移学习
print("\n练习3: 迁移学习")
print("- 使用预训练的CNN模型")
print("- 微调模型以适应新任务")

# 练习4: 目标检测
print("\n练习4: 目标检测")
print("- 实现简单的目标检测模型")
print("- 测试模型的检测效果")

# 练习5: 图像分割
print("\n练习5: 图像分割")
print("- 实现简单的图像分割模型")
print("- 测试模型的分割效果")

print("\n=== 第54天学习示例结束 ===")
