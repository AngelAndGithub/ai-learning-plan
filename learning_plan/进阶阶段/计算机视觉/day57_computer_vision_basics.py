#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第57天：计算机视觉基础
计算机视觉学习示例
内容：计算机视觉的基本概念、图像处理、OpenCV基础
"""

print("=== 第57天：计算机视觉基础 ===")

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os

# 1. 计算机视觉概述
print("\n1. 计算机视觉概述")

print("计算机视觉是让计算机理解和解释图像的科学")
print("- 目标：使计算机能够像人类一样理解图像内容")
print("- 应用：目标检测、图像分类、人脸识别、自动驾驶等")
print("- 挑战：光照变化、视角变化、遮挡等")

# 2. 图像处理基础
print("\n2. 图像处理基础")

print("图像处理是计算机视觉的基础")
print("- 像素：图像的基本单位")
print("- 分辨率：图像的大小")
print("- 颜色空间：RGB、灰度、HSV等")
print("- 图像操作：缩放、旋转、裁剪等")

# 3. OpenCV基础
print("\n3. OpenCV基础")

print("OpenCV是一个开源的计算机视觉库")
print("- 功能：图像处理、特征提取、目标检测等")
print("- 安装：pip install opencv-python")
print("- 导入：import cv2")

# 4. 图像读取与显示
print("\n4. 图像读取与显示")

# 创建示例图像
if not os.path.exists('images'):
    os.makedirs('images')

# 创建一个简单的彩色图像
img = np.zeros((200, 200, 3), dtype=np.uint8)
img[50:150, 50:150] = [0, 255, 0]  # 绿色方块
cv2.imwrite('images/example.jpg', img)
print("示例图像已创建为 images/example.jpg")

# 读取图像
image = cv2.imread('images/example.jpg')
print(f"图像形状: {image.shape}")
print(f"图像类型: {image.dtype}")

# 转换颜色空间（BGR -> RGB）
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 显示图像
plt.figure(figsize=(6, 6))
plt.imshow(image_rgb)
plt.title('示例图像')
plt.axis('off')
plt.savefig('images/displayed_image.png')
print("显示的图像已保存为 images/displayed_image.png")

# 5. 图像变换
print("\n5. 图像变换")

# 缩放图像
resized = cv2.resize(image, (100, 100))
resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

# 旋转图像
(h, w) = image.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))
rotated_rgb = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)

# 裁剪图像
cropped = image[50:150, 50:150]
cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

# 显示变换后的图像
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(resized_rgb)
plt.title('缩放')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(rotated_rgb)
plt.title('旋转')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cropped_rgb)
plt.title('裁剪')
plt.axis('off')

plt.tight_layout()
plt.savefig('images/image_transformations.png')
print("图像变换结果已保存为 images/image_transformations.png")

# 6. 颜色空间转换
print("\n6. 颜色空间转换")

# 转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 转换为HSV颜色空间
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 显示不同颜色空间的图像
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.title('RGB')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(gray, cmap='gray')
plt.title('灰度')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(hsv)
plt.title('HSV')
plt.axis('off')

plt.tight_layout()
plt.savefig('images/color_spaces.png')
print("颜色空间转换结果已保存为 images/color_spaces.png")

# 7. 图像阈值处理
print("\n7. 图像阈值处理")

# 简单阈值处理
ret, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO_INV)

# 显示阈值处理结果
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(gray, cmap='gray')
plt.title('原始灰度图像')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(thresh1, cmap='gray')
plt.title('二进制阈值')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(thresh2, cmap='gray')
plt.title('反二进制阈值')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(thresh3, cmap='gray')
plt.title('截断阈值')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(thresh4, cmap='gray')
plt.title('零阈值')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(thresh5, cmap='gray')
plt.title('反零阈值')
plt.axis('off')

plt.tight_layout()
plt.savefig('images/thresholding.png')
print("阈值处理结果已保存为 images/thresholding.png")

# 8. 图像滤波
print("\n8. 图像滤波")

# 添加噪声
noise = np.random.normal(0, 25, image.shape)
noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
noisy_image_rgb = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)

# 均值滤波
blurred = cv2.blur(noisy_image, (5, 5))
blurred_rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)

# 高斯滤波
gaussian_blurred = cv2.GaussianBlur(noisy_image, (5, 5), 0)
gaussian_blurred_rgb = cv2.cvtColor(gaussian_blurred, cv2.COLOR_BGR2RGB)

# 中值滤波
median_blurred = cv2.medianBlur(noisy_image, 5)
median_blurred_rgb = cv2.cvtColor(median_blurred, cv2.COLOR_BGR2RGB)

# 显示滤波结果
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.imshow(image_rgb)
plt.title('原始图像')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(noisy_image_rgb)
plt.title('带噪声图像')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(gaussian_blurred_rgb)
plt.title('高斯滤波')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(median_blurred_rgb)
plt.title('中值滤波')
plt.axis('off')

plt.tight_layout()
plt.savefig('images/filtering.png')
print("滤波结果已保存为 images/filtering.png")

# 9. 边缘检测
print("\n9. 边缘检测")

# Canny边缘检测
edges = cv2.Canny(gray, 50, 150)

# Sobel边缘检测
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobel = np.sqrt(sobel_x**2 + sobel_y**2)
sobel = np.uint8(sobel)

# Laplacian边缘检测
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
laplacian = np.uint8(np.absolute(laplacian))

# 显示边缘检测结果
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(edges, cmap='gray')
plt.title('Canny边缘检测')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(sobel, cmap='gray')
plt.title('Sobel边缘检测')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(laplacian, cmap='gray')
plt.title('Laplacian边缘检测')
plt.axis('off')

plt.tight_layout()
plt.savefig('images/edge_detection.png')
print("边缘检测结果已保存为 images/edge_detection.png")

# 10. 形态学操作
print("\n10. 形态学操作")

# 二值化图像
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 膨胀
kernel = np.ones((5, 5), np.uint8)
dilation = cv2.dilate(binary, kernel, iterations=1)

# 腐蚀
erosion = cv2.erode(binary, kernel, iterations=1)

# 开运算（先腐蚀后膨胀）
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

# 闭运算（先膨胀后腐蚀）
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# 显示形态学操作结果
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(binary, cmap='gray')
plt.title('二值图像')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(dilation, cmap='gray')
plt.title('膨胀')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(erosion, cmap='gray')
plt.title('腐蚀')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(opening, cmap='gray')
plt.title('开运算')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(closing, cmap='gray')
plt.title('闭运算')
plt.axis('off')

plt.tight_layout()
plt.savefig('images/morphological_operations.png')
print("形态学操作结果已保存为 images/morphological_operations.png")

# 11. 特征提取
print("\n11. 特征提取")

print("特征提取是计算机视觉的重要任务")
print("- 边缘特征：Canny、Sobel等")
print("- 角点特征：Harris角点检测")
print("- 纹理特征：LBP、HOG等")
print("- 深度学习特征：CNN特征")

# Harris角点检测
harris_corners = cv2.cornerHarris(gray, 2, 3, 0.04)
harris_corners = cv2.dilate(harris_corners, None)

# 标记角点
corner_image = image_rgb.copy()
corner_image[harris_corners > 0.01 * harris_corners.max()] = [255, 0, 0]

# 显示角点检测结果
plt.figure(figsize=(6, 6))
plt.imshow(corner_image)
plt.title('Harris角点检测')
plt.axis('off')
plt.savefig('images/corner_detection.png')
print("角点检测结果已保存为 images/corner_detection.png")

# 12. 计算机视觉应用
print("\n12. 计算机视觉应用")

print("计算机视觉的主要应用:")
print("- 目标检测：识别图像中的物体")
print("- 人脸识别：识别图像中的人脸")
print("- 图像分割：分割图像中的不同区域")
print("- 自动驾驶：车辆、行人检测")
print("- 医学影像：疾病诊断")
print("- 安防监控：异常行为检测")

# 13. 练习
print("\n13. 练习")

# 练习1: 图像处理
print("练习1: 图像处理")
print("- 加载一张真实图像")
print("- 进行各种图像处理操作")

# 练习2: 边缘检测
print("\n练习2: 边缘检测")
print("- 对不同图像进行边缘检测")
print("- 调整边缘检测的参数")

# 练习3: 特征提取
print("\n练习3: 特征提取")
print("- 提取图像的HOG特征")
print("- 分析特征的有效性")

# 练习4: 目标检测
print("\n练习4: 目标检测")
print("- 使用Haar级联分类器检测人脸")
print("- 测试检测效果")

# 练习5: 图像分类
print("\n练习5: 图像分类")
print("- 使用预训练的CNN模型进行图像分类")
print("- 测试分类效果")

print("\n=== 第57天学习示例结束 ===")
