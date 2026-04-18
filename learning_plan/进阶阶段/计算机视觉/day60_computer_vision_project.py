#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第60天：计算机视觉项目实战
计算机视觉学习示例
内容：计算机视觉项目的完整流程、目标检测、图像分割、模型部署
"""

print("=== 第60天：计算机视觉项目实战 ===")

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, UpSampling2D, concatenate
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from flask import Flask, request, jsonify
from PIL import Image
import io

# 1. 项目概述
print("\n1. 项目概述")

print("本项目实现一个完整的计算机视觉应用，包括目标检测和图像分割")
print("- 数据集：自定义数据集")
print("- 任务：目标检测和图像分割")
print("- 目标：构建一个端到端的计算机视觉系统")

# 2. 数据准备
print("\n2. 数据准备")

# 创建示例数据集
if not os.path.exists('dataset'):
    os.makedirs('dataset')
    os.makedirs('dataset/images')
    os.makedirs('dataset/masks')

# 创建示例图像和掩码
for i in range(10):
    # 创建图像
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[:] = [240, 240, 240]  # 背景色
    
    # 绘制圆形
    center_x = np.random.randint(50, 200)
    center_y = np.random.randint(50, 200)
    radius = np.random.randint(20, 40)
    cv2.circle(img, (center_x, center_y), radius, (0, 0, 255), -1)
    
    # 绘制矩形
    x1 = np.random.randint(50, 150)
    y1 = np.random.randint(50, 150)
    x2 = x1 + np.random.randint(30, 60)
    y2 = y1 + np.random.randint(30, 60)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), -1)
    
    # 保存图像
    cv2.imwrite(f'dataset/images/image_{i}.jpg', img)
    
    # 创建掩码
    mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), radius, 1, -1)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 2, -1)
    
    # 保存掩码
    cv2.imwrite(f'dataset/masks/mask_{i}.png', mask)

print("示例数据集已创建")

# 3. 数据加载与预处理
print("\n3. 数据加载与预处理")

# 加载图像和掩码
images = []
masks = []

for i in range(10):
    # 加载图像
    img = cv2.imread(f'dataset/images/image_{i}.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    images.append(img)
    
    # 加载掩码
    mask = cv2.imread(f'dataset/masks/mask_{i}.png', cv2.IMREAD_GRAYSCALE)
    mask = mask / 255.0
    masks.append(mask)

# 转换为numpy数组
images = np.array(images)
masks = np.array(masks)
masks = np.expand_dims(masks, axis=-1)

print(f"图像形状: {images.shape}")
print(f"掩码形状: {masks.shape}")

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.3, random_state=42)
print(f"训练集形状: {X_train.shape}, {y_train.shape}")
print(f"测试集形状: {X_test.shape}, {y_test.shape}")

# 4. 目标检测模型
print("\n4. 目标检测模型")

# 构建简单的目标检测模型
def build_object_detection_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='sigmoid')  # 输出：x, y, w, h
    ])
    return model

# 创建目标检测模型
detection_model = build_object_detection_model((256, 256, 3))
detection_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)
detection_model.summary()

# 5. 图像分割模型
print("\n5. 图像分割模型")

# 构建U-Net模型
def build_unet(input_shape):
    inputs = Input(input_shape)
    
    # 编码器
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # 瓶颈
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    
    # 解码器
    up4 = UpSampling2D(size=(2, 2))(conv3)
    up4 = concatenate([up4, conv2], axis=3)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(up4)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)
    
    up5 = UpSampling2D(size=(2, 2))(conv4)
    up5 = concatenate([up5, conv1], axis=3)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up5)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv5)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# 创建分割模型
segmentation_model = build_unet((256, 256, 3))
segmentation_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
segmentation_model.summary()

# 6. 模型训练
print("\n6. 模型训练")

# 训练目标检测模型
print("训练目标检测模型...")
detection_history = detection_model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=4,
    validation_split=0.2,
    verbose=1
)

# 训练图像分割模型
print("\n训练图像分割模型...")
segmentation_history = segmentation_model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=4,
    validation_split=0.2,
    verbose=1
)

# 7. 模型评估
print("\n7. 模型评估")

# 评估目标检测模型
detection_loss, detection_mae = detection_model.evaluate(X_test, y_test, verbose=0)
print(f"目标检测模型MAE: {detection_mae:.4f}")

# 评估图像分割模型
segmentation_loss, segmentation_accuracy = segmentation_model.evaluate(X_test, y_test, verbose=0)
print(f"图像分割模型准确率: {segmentation_accuracy:.4f}")

# 8. 模型预测
print("\n8. 模型预测")

# 目标检测预测
detection_predictions = detection_model.predict(X_test)

# 图像分割预测
segmentation_predictions = segmentation_model.predict(X_test)
segmentation_predictions = (segmentation_predictions > 0.5).astype(np.uint8)

# 可视化预测结果
plt.figure(figsize=(15, 10))

for i in range(3):
    # 原始图像
    plt.subplot(3, 3, i*3 + 1)
    plt.imshow(X_test[i])
    plt.title('原始图像')
    plt.axis('off')
    
    # 真实掩码
    plt.subplot(3, 3, i*3 + 2)
    plt.imshow(y_test[i].squeeze(), cmap='gray')
    plt.title('真实掩码')
    plt.axis('off')
    
    # 预测掩码
    plt.subplot(3, 3, i*3 + 3)
    plt.imshow(segmentation_predictions[i].squeeze(), cmap='gray')
    plt.title('预测掩码')
    plt.axis('off')

plt.tight_layout()
plt.savefig('segmentation_predictions.png')
print("分割预测结果已保存为 segmentation_predictions.png")

# 9. 模型部署
print("\n9. 模型部署")

# 创建模型保存目录
if not os.path.exists('models'):
    os.makedirs('models')

# 保存模型
segmentation_model.save('models/segmentation_model.h5')
print("模型已保存到 models/segmentation_model.h5")

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
import cv2

app = Flask(__name__)

# 加载模型
model = load_model('models/segmentation_model.h5')

@app.route('/segment', methods=['POST'])
def segment():
    """图像分割API"""
    try:
        # 获取请求数据
        if 'file' not in request.files:
            return jsonify({'error': '请提供图像文件'}), 400
        
        # 读取图像
        file = request.files['file']
        img = Image.open(io.BytesIO(file.read()))
        
        # 预处理图像
        img = img.resize((256, 256))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # 预测
        prediction = model.predict(img_array)
        prediction = (prediction > 0.5).astype(np.uint8)
        
        # 转换为图像
        mask = Image.fromarray(prediction[0].squeeze() * 255)
        
        # 保存结果
        mask.save('segmentation_result.png')
        
        # 构建响应
        response = {
            'status': 'success',
            'message': '图像分割成功',
            'result_path': 'segmentation_result.png'
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

RUN pip install --no-cache-dir flask pillow numpy opencv-python

EXPOSE 5000

CMD ["python", "app.py"]
''')
print("Dockerfile已创建")

# 创建docker-compose.yml
with open('docker-compose.yml', 'w') as f:
    f.write('''version: '3'
services:
  segmentation-service:
    build: .
    ports:
      - "5000:5000"
    restart: always
''')
print("docker-compose.yml已创建")

# 10. 项目总结
print("\n10. 项目总结")

print("本项目成功完成了计算机视觉应用的开发")
print("- 数据准备：创建了示例数据集")
print("- 模型训练：训练了目标检测和图像分割模型")
print("- 模型评估：评估了模型性能")
print("- 模型部署：创建了模型部署文件")

print("\n项目成果:")
print(f"图像分割模型准确率: {segmentation_accuracy:.4f}")

# 11. 练习
print("\n11. 练习")

# 练习1: 扩展数据集
print("练习1: 扩展数据集")
print("- 创建更大的数据集")
print("- 包含更多类型的物体")

# 练习2: 模型优化
print("\n练习2: 模型优化")
print("- 尝试不同的模型结构")
print("- 调整超参数")

# 练习3: 目标检测与分割结合
print("\n练习3: 目标检测与分割结合")
print("- 实现同时进行目标检测和分割的模型")
print("- 测试模型性能")

# 练习4: 部署到云服务
print("\n练习4: 部署到云服务")
print("- 将模型部署到AWS或Azure等云服务")
print("- 测试云部署的性能")

# 练习5: 实时视频处理
print("\n练习5: 实时视频处理")
print("- 实现实时视频分割")
print("- 测试实时处理性能")

print("\n=== 第60天学习示例结束 ===")
