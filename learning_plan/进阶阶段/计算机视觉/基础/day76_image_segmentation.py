#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第76天：图像分割
计算机视觉学习示例
内容：图像分割的基本概念、常见算法和应用
"""

print("=== 第76天：图像分割 ===")

# 1. 图像分割基本概念
print("\n1. 图像分割基本概念")

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image

print("图像分割是将图像分割成不同区域或物体的过程")
print("- 语义分割：将图像中的每个像素分配到特定的类别")
print("- 实例分割：区分同一类别的不同实例")
print("- 全景分割：同时进行语义分割和实例分割")
print("- 常见应用：医学影像分析、自动驾驶、机器人视觉")

# 2. 传统图像分割方法
print("\n2. 传统图像分割方法")

print("传统图像分割方法:")
print("1. 阈值分割：基于像素强度的阈值")
print("2. 边缘检测：基于边缘信息")
print("3. 区域生长：从种子点开始生长区域")
print("4. 聚类方法：如K-means、Mean Shift")
print("5. 分水岭算法：基于地形学的分割")

# 3. 阈值分割
print("\n3. 阈值分割")

# 读取图像
img = cv2.imread('lena.jpg')

if img is not None:
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 全局阈值
    ret, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # 自适应阈值
    thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Otsu阈值
    ret, thresh3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 显示结果
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 4, 1)
    plt.imshow(gray, cmap='gray')
    plt.title('灰度图像')
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.imshow(thresh1, cmap='gray')
    plt.title('全局阈值')
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.imshow(thresh2, cmap='gray')
    plt.title('自适应阈值')
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.imshow(thresh3, cmap='gray')
    plt.title('Otsu阈值')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
else:
    print("无法读取图像，请确保 'lena.jpg' 文件存在")

# 4. 分水岭算法
print("\n4. 分水岭算法")

if img is not None:
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 高斯模糊
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Otsu阈值
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 形态学操作
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # 确定背景区域
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # 确定前景区域
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # 找到未知区域
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # 标记连通区域
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    # 应用分水岭算法
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [0, 255, 0]
    
    # 转换为RGB格式
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 显示结果
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
    plt.title('原始图像')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(sure_fg, cmap='gray')
    plt.title('前景区域')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(img_rgb)
    plt.title('分水岭分割结果')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# 5. 深度学习图像分割
print("\n5. 深度学习图像分割")

print("深度学习图像分割模型:")
print("1. FCN (Fully Convolutional Network)：全卷积网络")
print("2. U-Net：用于医学影像分割")
print("3. SegNet：编码器-解码器结构")
print("4. DeepLab系列：使用空洞卷积")
print("5. Mask R-CNN：实例分割")
print("6. YOLOv5/YOLOv8-seg：实时实例分割")

# 6. U-Net模型
print("\n6. U-Net模型")

# 定义U-Net模型
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        # 编码器
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # 瓶颈
        self.bottleneck = self.conv_block(512, 1024)
        
        # 解码器
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        # 输出层
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 编码器
        e1 = self.enc1(x)
        e2 = self.enc2(nn.functional.max_pool2d(e1, 2))
        e3 = self.enc3(nn.functional.max_pool2d(e2, 2))
        e4 = self.enc4(nn.functional.max_pool2d(e3, 2))
        
        # 瓶颈
        b = self.bottleneck(nn.functional.max_pool2d(e4, 2))
        
        # 解码器
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        # 输出
        return self.output(d1)

# 创建U-Net模型
model = UNet()
print("U-Net模型:")
print(model)

# 7. 语义分割
print("\n7. 语义分割")

print("语义分割将图像中的每个像素分配到特定的类别")
print("- 数据集：Cityscapes、PASCAL VOC、COCO")
print("- 评价指标：mIoU (mean Intersection over Union)")
print("- 应用：自动驾驶、医学影像分析")

# 8. 实例分割
print("\n8. 实例分割")

print("实例分割区分同一类别的不同实例")
print("- 方法：Mask R-CNN、YOLOv5-seg、YOLOv8-seg")
print("- 应用：物体计数、机器人抓取")

# 9. 全景分割
print("\n9. 全景分割")

print("全景分割同时进行语义分割和实例分割")
print("- 方法：Panoptic FPN、UPSNet")
print("- 应用：场景理解、增强现实")

# 10. 练习
print("\n10. 练习")

# 练习1: 实现不同的分割算法
print("练习1: 实现不同的分割算法")
print("- 尝试实现阈值分割")
print("- 尝试实现分水岭算法")
print("- 尝试实现K-means聚类分割")

# 练习2: 使用深度学习模型
print("\n练习2: 使用深度学习模型")
print("- 尝试使用预训练的分割模型")
print("- 尝试训练自己的分割模型")
print("- 尝试不同的分割架构")

# 练习3: 医学影像分割
print("\n练习3: 医学影像分割")
print("- 尝试分割医学影像")
print("- 评估分割性能")
print("- 可视化分割结果")

# 练习4: 实时分割
print("\n练习4: 实时分割")
print("- 尝试实时视频分割")
print("- 优化模型以提高推理速度")
print("- 部署到边缘设备")

print("\n=== 第76天学习示例结束 ===")
