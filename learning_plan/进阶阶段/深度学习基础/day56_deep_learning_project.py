#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第56天：深度学习项目实战
深度学习学习示例
内容：深度学习项目的完整流程、模型训练与评估、部署
"""

print("=== 第56天：深度学习项目实战 ===")

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os

# 1. 项目概述
print("\n1. 项目概述")

print("本项目使用CIFAR-10数据集进行图像分类任务")
print("- 数据集：CIFAR-10，包含10个类别的60000张32x32彩色图像")
print("- 任务：图像分类，识别图像中的物体类别")
print("- 目标：构建一个准确的图像分类模型")

# 2. 数据获取与探索
print("\n2. 数据获取与探索")

# 加载CIFAR-10数据集
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print(f"训练集形状: {X_train.shape}, {y_train.shape}")
print(f"测试集形状: {X_test.shape}, {y_test.shape}")

# 类别名称
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print(f"类别名称: {class_names}")

# 可视化样本
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i])
    plt.xlabel(class_names[y_train[i][0]])
plt.tight_layout()
plt.savefig('cifar10_samples.png')
print("样本图像已保存为 cifar10_samples.png")

# 3. 数据预处理
print("\n3. 数据预处理")

# 数据归一化
X_train = X_train / 255.0
X_test = X_test / 255.0

# 标签编码
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print(f"预处理后训练集形状: {X_train.shape}, {y_train.shape}")
print(f"预处理后测试集形状: {X_test.shape}, {y_test.shape}")

# 4. 基础模型构建
print("\n4. 基础模型构建")

# 构建基础CNN模型
base_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# 编译模型
base_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 查看模型结构
base_model.summary()

# 5. 数据增强
print("\n5. 数据增强")

# 数据增强
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

datagen.fit(X_train)

# 6. 模型训练
print("\n6. 模型训练")

# 训练基础模型
base_history = base_model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    epochs=20,
    validation_data=(X_test, y_test),
    verbose=1
)

# 7. 模型评估
print("\n7. 模型评估")

# 评估基础模型
base_loss, base_accuracy = base_model.evaluate(X_test, y_test, verbose=0)
print(f"基础模型测试准确率: {base_accuracy:.4f}")

# 预测
base_predictions = base_model.predict(X_test)
base_predicted_classes = np.argmax(base_predictions, axis=1)
base_true_classes = np.argmax(y_test, axis=1)

# 分类报告
print("\n基础模型分类报告:")
print(classification_report(base_true_classes, base_predictions, target_names=class_names))

# 混淆矩阵
base_cm = confusion_matrix(base_true_classes, base_predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(base_cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('基础模型混淆矩阵')
plt.xlabel('预测类别')
plt.ylabel('真实类别')
plt.tight_layout()
plt.savefig('base_model_confusion_matrix.png')
print("基础模型混淆矩阵已保存为 base_model_confusion_matrix.png")

# 8. 迁移学习
print("\n8. 迁移学习")

print("使用预训练的VGG16模型进行迁移学习")

# 加载预训练模型
base_model_vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# 冻结预训练模型的权重
for layer in base_model_vgg16.layers:
    layer.trainable = False

# 构建迁移学习模型
inputs = Input(shape=(32, 32, 3))
x = base_model_vgg16(inputs)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(10, activation='softmax')(x)

vgg16_model = Model(inputs, outputs)

# 编译模型
vgg16_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 查看模型结构
vgg16_model.summary()

# 训练迁移学习模型
vgg16_history = vgg16_model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    epochs=10,
    validation_data=(X_test, y_test),
    verbose=1
)

# 评估迁移学习模型
vgg16_loss, vgg16_accuracy = vgg16_model.evaluate(X_test, y_test, verbose=0)
print(f"VGG16迁移学习模型测试准确率: {vgg16_accuracy:.4f}")

# 9. 模型调优
print("\n9. 模型调优")

print("微调VGG16模型")

# 解冻部分层
for layer in base_model_vgg16.layers[-4:]:
    layer.trainable = True

# 重新编译模型
vgg16_model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 继续训练模型
fine_tune_history = vgg16_model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    epochs=10,
    validation_data=(X_test, y_test),
    verbose=1
)

# 评估微调后的模型
fine_tune_loss, fine_tune_accuracy = vgg16_model.evaluate(X_test, y_test, verbose=0)
print(f"微调后VGG16模型测试准确率: {fine_tune_accuracy:.4f}")

# 10. 模型比较
print("\n10. 模型比较")

print("不同模型的性能比较:")
print(f"基础CNN模型: {base_accuracy:.4f}")
print(f"VGG16迁移学习模型: {vgg16_accuracy:.4f}")
print(f"微调后VGG16模型: {fine_tune_accuracy:.4f}")

# 11. 训练过程可视化
print("\n11. 训练过程可视化")

# 绘制训练历史
plt.figure(figsize=(15, 5))

# 基础模型
plt.subplot(1, 3, 1)
plt.plot(base_history.history['accuracy'], label='训练准确率')
plt.plot(base_history.history['val_accuracy'], label='验证准确率')
plt.title('基础模型准确率')
plt.xlabel('epoch')
plt.ylabel('准确率')
plt.legend()
plt.grid()

# VGG16迁移学习模型
plt.subplot(1, 3, 2)
plt.plot(vgg16_history.history['accuracy'], label='训练准确率')
plt.plot(vgg16_history.history['val_accuracy'], label='验证准确率')
plt.title('VGG16迁移学习模型准确率')
plt.xlabel('epoch')
plt.ylabel('准确率')
plt.legend()
plt.grid()

# 微调后VGG16模型
plt.subplot(1, 3, 3)
plt.plot(fine_tune_history.history['accuracy'], label='训练准确率')
plt.plot(fine_tune_history.history['val_accuracy'], label='验证准确率')
plt.title('微调后VGG16模型准确率')
plt.xlabel('epoch')
plt.ylabel('准确率')
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig('model_training_history.png')
print("训练历史图已保存为 model_training_history.png")

# 12. 模型保存与加载
print("\n12. 模型保存与加载")

# 创建模型保存目录
if not os.path.exists('models'):
    os.makedirs('models')

# 保存最佳模型
vgg16_model.save('models/best_model.h5')
print("模型已保存到 models/best_model.h5")

# 加载模型
from tensorflow.keras.models import load_model
loaded_model = load_model('models/best_model.h5')
print("模型加载成功")

# 测试加载的模型
loaded_loss, loaded_accuracy = loaded_model.evaluate(X_test, y_test, verbose=0)
print(f"加载模型测试准确率: {loaded_accuracy:.4f}")

# 13. 模型部署
print("\n13. 模型部署")

print("创建模型部署文件")

# 创建Flask应用文件
with open('app.py', 'w') as f:
    f.write('''#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify
from PIL import Image
import io

app = Flask(__name__)

# 加载模型
model = load_model('models/best_model.h5')

# 类别名称
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

@app.route('/predict', methods=['POST'])
def predict():
    """模型预测API"""
    try:
        # 获取请求数据
        if 'file' not in request.files:
            return jsonify({'error': '请提供图像文件'}), 400
        
        # 读取图像
        file = request.files['file']
        img = Image.open(io.BytesIO(file.read()))
        
        # 预处理图像
        img = img.resize((32, 32))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        
        # 预测
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = predictions[0][predicted_class]
        
        # 构建响应
        response = {
            'prediction': class_names[predicted_class],
            'confidence': float(confidence),
            'class_index': int(predicted_class)
        }
        
        return jsonify(response), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """健康检查API"""
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
''')
print("app.py已创建")

# 创建Dockerfile
with open('Dockerfile', 'w') as f:
    f.write('''FROM tensorflow/tensorflow:2.8.0

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir flask pillow numpy

EXPOSE 5000

CMD ["python", "app.py"]
''')
print("Dockerfile已创建")

# 创建docker-compose.yml
with open('docker-compose.yml', 'w') as f:
    f.write('''version: '3'
services:
  cifar10-classifier:
    build: .
    ports:
      - "5000:5000"
    restart: always
''')
print("docker-compose.yml已创建")

# 14. 项目总结
print("\n14. 项目总结")

print("本项目成功完成了CIFAR-10图像分类任务")
print("- 数据探索：了解了CIFAR-10数据集的基本情况")
print("- 数据预处理：进行了数据归一化和标签编码")
print("- 基础模型：构建了基础CNN模型")
print("- 数据增强：使用数据增强提高模型性能")
print("- 迁移学习：使用VGG16预训练模型")
print("- 模型微调：微调VGG16模型以提高性能")
print("- 模型评估：评估了不同模型的性能")
print("- 模型部署：创建了模型部署文件")

print("\n项目成果:")
print(f"最佳模型: 微调后VGG16模型")
print(f"最佳准确率: {fine_tune_accuracy:.4f}")

# 15. 练习
print("\n15. 练习")

# 练习1: 尝试其他预训练模型
print("练习1: 尝试其他预训练模型")
print("- 使用ResNet50或MobileNet等预训练模型")
print("- 比较不同预训练模型的性能")

# 练习2: 模型优化
print("\n练习2: 模型优化")
print("- 尝试不同的优化器和学习率")
print("- 调整模型结构和超参数")

# 练习3: 模型解释
print("\n练习3: 模型解释")
print("- 使用Grad-CAM可视化模型关注的区域")
print("- 分析模型的决策过程")

# 练习4: 部署到云服务
print("\n练习4: 部署到云服务")
print("- 将模型部署到AWS或Azure等云服务")
print("- 测试云部署的性能")

# 练习5: 构建完整的MLOps流程
print("\n练习5: 构建完整的MLOps流程")
print("- 实现自动化的模型训练和部署")
print("- 构建CI/CD pipeline")

print("\n=== 第56天学习示例结束 ===")
