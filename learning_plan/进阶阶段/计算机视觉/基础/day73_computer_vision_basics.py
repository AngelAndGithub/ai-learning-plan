#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第73天：计算机视觉基础
计算机视觉学习示例
内容：图像处理的基本操作、图像变换和图像增强
"""

print("=== 第73天：计算机视觉基础 ===")

# 1. 图像处理基础
print("\n1. 图像处理基础")

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

print("计算机视觉是研究如何使计算机理解和处理图像的学科")
print("- 图像处理：对图像进行各种操作和变换")
print("- 图像分析：提取图像中的信息")
print("- 图像理解：理解图像的内容和含义")

# 2. 图像读取和显示
print("\n2. 图像读取和显示")

# 读取图像
img = cv2.imread('lena.jpg')

if img is None:
    print("无法读取图像，请确保 'lena.jpg' 文件存在")
else:
    print(f"图像形状: {img.shape}")
    print(f"图像高度: {img.shape[0]}")
    print(f"图像宽度: {img.shape[1]}")
    print(f"图像通道数: {img.shape[2]}")

# 转换为RGB格式
if img is not None:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 6))
    plt.imshow(img_rgb)
    plt.title('原始图像')
    plt.axis('off')
    plt.show()

# 3. 图像基本操作
print("\n3. 图像基本操作")

if img is not None:
    # 裁剪图像
    height, width = img.shape[:2]
    crop_img = img[height//4:3*height//4, width//4:3*width//4]
    crop_img_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    
    # 调整图像大小
    resized_img = cv2.resize(img, (width//2, height//2))
    resized_img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    
    # 旋转图像
    M = cv2.getRotationMatrix2D((width//2, height//2), 45, 1.0)
    rotated_img = cv2.warpAffine(img, M, (width, height))
    rotated_img_rgb = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB)
    
    # 显示结果
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(crop_img_rgb)
    plt.title('裁剪图像')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(resized_img_rgb)
    plt.title('调整大小')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(rotated_img_rgb)
    plt.title('旋转图像')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# 4. 图像变换
print("\n4. 图像变换")

if img is not None:
    # 灰度转换
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 二值化
    ret, binary_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
    
    # 高斯模糊
    blur_img = cv2.GaussianBlur(img, (5, 5), 0)
    blur_img_rgb = cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB)
    
    # 边缘检测
    edges = cv2.Canny(gray_img, 100, 200)
    
    # 显示结果
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 4, 1)
    plt.imshow(gray_img, cmap='gray')
    plt.title('灰度图像')
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.imshow(binary_img, cmap='gray')
    plt.title('二值图像')
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.imshow(blur_img_rgb)
    plt.title('高斯模糊')
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.imshow(edges, cmap='gray')
    plt.title('边缘检测')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# 5. 图像增强
print("\n5. 图像增强")

if img is not None:
    # 亮度调整
    alpha = 1.5  # 对比度增益
    beta = 50    # 亮度增益
    bright_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    bright_img_rgb = cv2.cvtColor(bright_img, cv2.COLOR_BGR2RGB)
    
    # 直方图均衡化
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equalized_img = cv2.equalizeHist(gray_img)
    
    # 锐化
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened_img = cv2.filter2D(img, -1, kernel)
    sharpened_img_rgb = cv2.cvtColor(sharpened_img, cv2.COLOR_BGR2RGB)
    
    # 显示结果
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(bright_img_rgb)
    plt.title('亮度调整')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(equalized_img, cmap='gray')
    plt.title('直方图均衡化')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(sharpened_img_rgb)
    plt.title('锐化')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# 6. 形态学操作
print("\n6. 形态学操作")

if img is not None:
    # 转换为灰度图
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 二值化
    ret, binary_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
    
    # 定义结构元素
    kernel = np.ones((5, 5), np.uint8)
    
    # 腐蚀
    erosion = cv2.erode(binary_img, kernel, iterations=1)
    
    # 膨胀
    dilation = cv2.dilate(binary_img, kernel, iterations=1)
    
    # 开运算（先腐蚀后膨胀）
    opening = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
    
    # 闭运算（先膨胀后腐蚀）
    closing = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
    
    # 显示结果
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(binary_img, cmap='gray')
    plt.title('二值图像')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(erosion, cmap='gray')
    plt.title('腐蚀')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(dilation, cmap='gray')
    plt.title('膨胀')
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
    plt.show()

# 7. 图像分割
print("\n7. 图像分割")

if img is not None:
    # 转换为HSV颜色空间
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 定义蓝色的HSV范围
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    
    # 创建掩码
    mask = cv2.inRange(hsv_img, lower_blue, upper_blue)
    
    # 应用掩码
    blue_objects = cv2.bitwise_and(img, img, mask=mask)
    blue_objects_rgb = cv2.cvtColor(blue_objects, cv2.COLOR_BGR2RGB)
    
    # 显示结果
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('原始图像')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('蓝色掩码')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(blue_objects_rgb)
    plt.title('蓝色物体')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# 8. 特征提取
print("\n8. 特征提取")

if img is not None:
    # 转换为灰度图
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 使用SIFT提取特征
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray_img, None)
    
    # 绘制特征点
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, None)
    img_with_keypoints_rgb = cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB)
    
    print(f"检测到 {len(keypoints)} 个特征点")
    
    # 显示结果
    plt.figure(figsize=(8, 6))
    plt.imshow(img_with_keypoints_rgb)
    plt.title('SIFT特征点')
    plt.axis('off')
    plt.show()

# 9. 图像识别
print("\n9. 图像识别")

print("图像识别是计算机视觉的重要应用")
print("- 分类：识别图像属于哪个类别")
print("- 检测：定位图像中的物体")
print("- 分割：将图像分割成不同的区域")
print("- 跟踪：跟踪视频中的物体")

# 10. 练习
print("\n10. 练习")

# 练习1: 图像处理
print("练习1: 图像处理")
print("- 尝试不同的图像变换")
print("- 尝试不同的图像增强方法")
print("- 尝试不同的形态学操作")

# 练习2: 图像分割
print("\n练习2: 图像分割")
print("- 尝试分割不同颜色的物体")
print("- 尝试使用 watershed算法进行分割")
print("- 尝试使用 GrabCut 算法进行分割")

# 练习3: 特征提取
print("\n练习3: 特征提取")
print("- 尝试使用 SURF 提取特征")
print("- 尝试使用 ORB 提取特征")
print("- 尝试使用 BRIEF 提取特征")

print("\n=== 第73天学习示例结束 ===")
