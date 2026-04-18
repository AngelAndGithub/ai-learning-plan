#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第59天：图像分割
计算机视觉学习示例
内容：图像分割的基本概念、语义分割、实例分割、全景分割
"""

print("=== 第59天：图像分割 ===")

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

# 1. 图像分割概述
print("\n1. 图像分割概述")

print("图像分割是将图像分割成不同区域的过程")
print("- 目标：将图像中的像素分配到不同的类别")
print("- 类型：语义分割、实例分割、全景分割")
print("- 应用：医学影像、自动驾驶、机器人视觉等")

# 2. 图像分割类型
print("\n2. 图像分割类型")

print("图像分割的主要类型:")
print("- 语义分割：将图像中的每个像素分配到一个语义类别")
print("- 实例分割：不仅要分类，还要区分不同的实例")
print("- 全景分割：结合语义分割和实例分割")

# 3. 数据集准备
print("\n3. 数据集准备")

# 创建示例图像
if not os.path.exists('images'):
    os.makedirs('images')

# 创建一个包含多个区域的示例图像
img = np.zeros((300, 400, 3), dtype=np.uint8)

# 背景
img[:] = [240, 240, 240]

# 圆形区域
cv2.circle(img, (100, 100), 50, (0, 0, 255), -1)

# 矩形区域
cv2.rectangle(img, (200, 50), (300, 150), (0, 255, 0), -1)

# 三角形区域
points = np.array([[300, 200], [350, 250], [250, 250]], np.int32)
cv2.fillPoly(img, [points], (255, 0, 0))

# 保存示例图像
cv2.imwrite('images/segmentation_example.jpg', img)
print("示例图像已创建为 images/segmentation_example.jpg")

# 读取并显示图像
image = cv2.imread('images/segmentation_example.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 6))
plt.imshow(image_rgb)
plt.title('示例图像')
plt.axis('off')
plt.savefig('images/displayed_segmentation.png')
print("显示的图像已保存为 images/displayed_segmentation.png")

# 4. 传统图像分割方法
print("\n4. 传统图像分割方法")

print("传统的图像分割方法:")
print("- 阈值分割：基于像素值的阈值")
print("- 边缘检测：基于边缘信息")
print("- 区域生长：基于相似性")
print("- 聚类：K-means、Mean Shift等")

# 4.1 阈值分割
print("\n4.1 阈值分割")

# 转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 简单阈值分割
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 自适应阈值分割
adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# 显示阈值分割结果
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(gray, cmap='gray')
plt.title('灰度图像')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(thresh, cmap='gray')
plt.title('简单阈值分割')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(adaptive_thresh, cmap='gray')
plt.title('自适应阈值分割')
plt.axis('off')

plt.tight_layout()
plt.savefig('images/threshold_segmentation.png')
print("阈值分割结果已保存为 images/threshold_segmentation.png")

# 4.2 K-means聚类分割
print("\n4.2 K-means聚类分割")

# 重塑图像为二维数组
Z = image.reshape((-1, 3))
Z = np.float32(Z)

# 定义K-means参数
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3
ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# 转换回uint8
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((image.shape))

# 显示K-means分割结果
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(res2, cv2.COLOR_BGR2RGB))
plt.title('K-means聚类分割')
plt.axis('off')
plt.savefig('images/kmeans_segmentation.png')
print("K-means聚类分割结果已保存为 images/kmeans_segmentation.png")

# 5. 深度学习图像分割
print("\n5. 深度学习图像分割")

print("基于深度学习的图像分割方法:")
print("- FCN（全卷积网络）：将全连接层替换为卷积层")
print("- U-Net：编码器-解码器结构，使用跳跃连接")
print("- SegNet：使用编码器的池化索引进行上采样")
print("- Mask R-CNN：实例分割的经典方法")

# 5.1 U-Net模型
print("\n5.1 U-Net模型")

# 构建简单的U-Net模型
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

# 创建U-Net模型
model = build_unet((300, 400, 3))
model.summary()

# 6. 语义分割
print("\n6. 语义分割")

print("语义分割是将图像中的每个像素分配到一个语义类别")
print("- 应用：自动驾驶中的道路分割、医学影像中的器官分割")
print("- 评估指标：IoU（交并比）、Dice系数")

# 7. 实例分割
print("\n7. 实例分割")

print("实例分割不仅要分类，还要区分不同的实例")
print("- 应用：行人检测、物体跟踪")
print("- 方法：Mask R-CNN、YOLACT等")

# 8. 全景分割
print("\n8. 全景分割")

print("全景分割结合了语义分割和实例分割")
print("- 目标：同时处理stuff（无实例的类别）和thing（有实例的类别）")
print("- 方法：Panoptic FPN、UPSNet等")

# 9. 图像分割的评估指标
print("\n9. 图像分割的评估指标")

print("图像分割的评估指标:")
print("- IoU（交并比）：预测区域与真实区域的重叠程度")
print("- Dice系数：2 * 交集 / (预测区域大小 + 真实区域大小)")
print("- 像素准确率：正确分类的像素占总像素的比例")
print("- 平均准确率：不同类别的平均准确率")

# 10. 图像分割的挑战
print("\n10. 图像分割的挑战")

print("图像分割的挑战:")
print("- 边界模糊：物体边界不清晰")
print("- 遮挡：物体被部分遮挡")
print("- 光照变化：光照条件影响分割效果")
print("- 类别不平衡：某些类别的像素较少")
print("- 计算资源：深度学习模型需要大量计算资源")

# 11. 图像分割的应用
print("\n11. 图像分割的应用")

print("图像分割的主要应用:")
print("- 医学影像：分割器官、病变区域")
print("- 自动驾驶：分割道路、车辆、行人")
print("- 机器人视觉：分割物体，便于抓取")
print("- 卫星影像：分割土地、建筑、植被")
print("- 视频编辑：分割前景和背景")

# 12. 练习
print("\n12. 练习")

# 练习1: 传统图像分割
print("练习1: 传统图像分割")
print("- 使用阈值分割和K-means聚类分割图像")
print("- 比较不同方法的分割效果")

# 练习2: U-Net模型训练
print("\n练习2: U-Net模型训练")
print("- 准备分割数据集")
print("- 训练U-Net模型")
print("- 评估模型性能")

# 练习3: 语义分割
print("\n练习3: 语义分割")
print("- 使用预训练的语义分割模型")
print("- 测试模型在不同图像上的分割效果")

# 练习4: 实例分割
print("\n练习4: 实例分割")
print("- 使用Mask R-CNN进行实例分割")
print("- 测试分割效果")

# 练习5: 医学影像分割
print("\n练习5: 医学影像分割")
print("- 下载医学影像数据集")
print("- 训练分割模型")
print("- 评估分割效果")

print("\n=== 第59天学习示例结束 ===")
