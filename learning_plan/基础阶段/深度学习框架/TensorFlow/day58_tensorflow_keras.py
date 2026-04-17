#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第58天：TensorFlow Keras API
深度学习框架学习示例
内容：Keras的基本概念、模型构建和训练
"""

print("=== 第58天：TensorFlow Keras API ===")

# 1. Keras基本概念
print("\n1. Keras基本概念")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

print("Keras是TensorFlow的高级API，用于快速构建和训练深度学习模型")
print("- 序贯模型 (Sequential): 线性堆叠的层")
print("- 函数式API (Functional API): 更灵活的模型构建")
print("- 层 (Layer): 神经网络的基本构建块")
print("- 激活函数 (Activation): 引入非线性")
print("- 损失函数 (Loss Function): 衡量模型性能")
print("- 优化器 (Optimizer): 更新模型参数")
print("- 评估指标 (Metrics): 评估模型性能")

# 2. 序贯模型
print("\n2. 序贯模型")

# 创建一个简单的序贯模型
model = Sequential([
    Dense(64, activation='relu', input_shape=(784,)),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])

# 查看模型结构
model.summary()

# 3. 模型编译
print("\n3. 模型编译")

# 编译模型
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("模型已编译，配置了优化器、损失函数和评估指标")

# 4. 数据加载和预处理
print("\n4. 数据加载和预处理")

# 加载MNIST数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(f"训练数据形状: {x_train.shape}")
print(f"测试数据形状: {x_test.shape}")
print(f"训练标签形状: {y_train.shape}")
print(f"测试标签形状: {y_test.shape}")

# 数据预处理
# 归一化
x_train = x_train / 255.0
x_test = x_test / 255.0

# 展平图像
x_train_flat = x_train.reshape(-1, 784)
x_test_flat = x_test.reshape(-1, 784)

# 标签独热编码
y_train_one_hot = to_categorical(y_train, 10)
y_test_one_hot = to_categorical(y_test, 10)

print(f"\n预处理后训练数据形状: {x_train_flat.shape}")
print(f"预处理后测试数据形状: {x_test_flat.shape}")
print(f"独热编码后训练标签形状: {y_train_one_hot.shape}")
print(f"独热编码后测试标签形状: {y_test_one_hot.shape}")

# 5. 模型训练
print("\n5. 模型训练")

# 训练模型
history = model.fit(
    x_train_flat,
    y_train_one_hot,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)

# 6. 模型评估
print("\n6. 模型评估")

# 评估模型
loss, accuracy = model.evaluate(x_test_flat, y_test_one_hot)
print(f"测试集损失: {loss:.4f}")
print(f"测试集准确率: {accuracy:.4f}")

# 7. 模型预测
print("\n7. 模型预测")

# 预测
predictions = model.predict(x_test_flat)

# 查看前5个预测
print("前5个预测:")
for i in range(5):
    predicted_label = np.argmax(predictions[i])
    true_label = np.argmax(y_test_one_hot[i])
    print(f"样本 {i+1}: 预测标签 = {predicted_label}, 真实标签 = {true_label}, 预测正确: {predicted_label == true_label}")

# 8. 模型保存和加载
print("\n8. 模型保存和加载")

# 保存模型
model.save('mnist_model.h5')
print("模型已保存到 'mnist_model.h5'")

# 加载模型
from tensorflow.keras.models import load_model
loaded_model = load_model('mnist_model.h5')
print("模型已加载")

# 测试加载的模型
loss, accuracy = loaded_model.evaluate(x_test_flat, y_test_one_hot)
print(f"加载模型的测试集准确率: {accuracy:.4f}")

# 9. 函数式API
print("\n9. 函数式API")

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

# 使用函数式API创建模型
inputs = Input(shape=(784,))
x = Dense(64, activation='relu')(inputs)
x = Dense(32, activation='relu')(x)
outputs = Dense(10, activation='softmax')(x)

functional_model = Model(inputs=inputs, outputs=outputs)

# 查看模型结构
functional_model.summary()

# 编译模型
functional_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 训练模型
functional_model.fit(
    x_train_flat,
    y_train_one_hot,
    epochs=5,
    batch_size=32,
    validation_split=0.2
)

# 10. 卷积神经网络
print("\n10. 卷积神经网络")

# 准备数据
x_train_cnn = x_train.reshape(-1, 28, 28, 1)
x_test_cnn = x_test.reshape(-1, 28, 28, 1)

print(f"CNN输入形状: {x_train_cnn.shape}")

# 创建卷积神经网络
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# 查看模型结构
cnn_model.summary()

# 编译模型
cnn_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 训练模型
cnn_history = cnn_model.fit(
    x_train_cnn,
    y_train_one_hot,
    epochs=5,
    batch_size=32,
    validation_split=0.2
)

# 评估模型
loss, accuracy = cnn_model.evaluate(x_test_cnn, y_test_one_hot)
print(f"CNN测试集准确率: {accuracy:.4f}")

# 11. 回调函数
print("\n11. 回调函数")

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 定义回调函数
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy')

# 训练模型时使用回调函数
model.fit(
    x_train_flat,
    y_train_one_hot,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, model_checkpoint]
)

# 12. 学习曲线
print("\n12. 学习曲线")

# 绘制训练和验证准确率
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='训练准确率')
plt.plot(history.history['val_accuracy'], label='验证准确率')
plt.title('模型准确率')
plt.xlabel('Epoch')
plt.ylabel('准确率')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.title('模型损失')
plt.xlabel('Epoch')
plt.ylabel('损失')
plt.legend()

plt.tight_layout()
plt.show()

# 13. 练习
print("\n13. 练习")

# 练习1: 构建不同的模型架构
print("练习1: 构建不同的模型架构")
print("- 尝试不同的隐藏层数量")
print("- 尝试不同的神经元数量")
print("- 尝试不同的激活函数")

# 练习2: 调整超参数
print("\n练习2: 调整超参数")
print("- 尝试不同的批量大小 (batch_size)")
print("- 尝试不同的学习率")
print("- 尝试不同的优化器")

# 练习3: 数据增强
print("\n练习3: 数据增强")
print("- 使用ImageDataGenerator进行数据增强")
print("- 应用旋转、平移、缩放等变换")

# 示例：数据增强
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

# 示例代码
print("\n数据增强示例代码:")
print("datagen = ImageDataGenerator(")
print("    rotation_range=10,")
print("    width_shift_range=0.1,")
print("    height_shift_range=0.1,")
print("    zoom_range=0.1")
print(")")
print("")
print("# 拟合数据生成器")
print("datagen.fit(x_train_cnn)")
print("")
print("# 使用生成器训练模型")
print("model.fit(datagen.flow(x_train_cnn, y_train_one_hot, batch_size=32),")
print("          epochs=10,")
print("          validation_data=(x_test_cnn, y_test_one_hot))")

print("\n=== 第58天学习示例结束 ===")
