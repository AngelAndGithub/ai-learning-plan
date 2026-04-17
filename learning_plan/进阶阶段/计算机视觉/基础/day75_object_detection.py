#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第75天：目标检测
计算机视觉学习示例
内容：目标检测的基本概念、常见算法和应用
"""

print("=== 第75天：目标检测 ===")

# 1. 目标检测基本概念
print("\n1. 目标检测基本概念")

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
from PIL import Image

print("目标检测是计算机视觉的重要任务，用于定位和识别图像中的物体")
print("- 定位：确定物体在图像中的位置（边界框）")
print("- 分类：识别物体的类别")
print("- 常见应用：自动驾驶、安防监控、人脸识别、物体跟踪")

# 2. 传统目标检测方法
print("\n2. 传统目标检测方法")

print("传统目标检测方法:")
print("1. 滑动窗口：在图像上滑动不同大小的窗口")
print("2. 特征提取：使用HOG、SIFT等特征")
print("3. 分类器：使用SVM、Adaboost等分类器")
print("4. 示例：Haar级联分类器（用于人脸检测）")

# 3. Haar级联分类器人脸检测
print("\n3. Haar级联分类器人脸检测")

# 加载Haar级联分类器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# 读取图像
img = cv2.imread('lena.jpg')

if img is not None:
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 检测人脸
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # 绘制人脸和眼睛
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    
    # 转换为RGB格式
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 显示结果
    plt.figure(figsize=(8, 6))
    plt.imshow(img_rgb)
    plt.title('人脸检测结果')
    plt.axis('off')
    plt.show()
else:
    print("无法读取图像，请确保 'lena.jpg' 文件存在")

# 4. 深度学习目标检测
print("\n4. 深度学习目标检测")

print("深度学习目标检测算法:")
print("1. R-CNN系列：R-CNN、Fast R-CNN、Faster R-CNN")
print("2. YOLO系列：YOLOv1-v5、YOLOv7、YOLOv8")
print("3. SSD系列：SSD、FPN、RetinaNet")
print("4. 其他：CenterNet、DETR等")

# 5. 使用预训练的YOLO模型
print("\n5. 使用预训练的YOLO模型")

print("使用预训练的YOLOv5模型进行目标检测")
print("步骤:")
print("1. 安装YOLOv5")
print("2. 加载预训练模型")
print("3. 进行目标检测")
print("4. 可视化结果")

# 示例代码
print("\nYOLOv5示例代码:")
print("# 安装YOLOv5")
print("!git clone https://github.com/ultralytics/yolov5")
print("%cd yolov5")
print("!pip install -r requirements.txt")
print("")
print("# 导入库")
print("import torch")
print("from PIL import Image")
print("import cv2")
print("import numpy as np")
print("")
print("# 加载模型")
print("model = torch.hub.load('ultralytics/yolov5', 'yolov5s')")
print("")
print("# 加载图像")
print("img = 'https://ultralytics.com/images/zidane.jpg'")
print("")
print("# 进行检测")
print("results = model(img)")
print("")
print("# 显示结果")
print("results.show()")

# 6. 目标检测评估指标
print("\n6. 目标检测评估指标")

print("目标检测的评估指标:")
print("1. IoU (Intersection over Union)：预测边界框与真实边界框的交并比")
print("2. Precision：正确预测的正样本占所有预测正样本的比例")
print("3. Recall：正确预测的正样本占所有真实正样本的比例")
print("4. mAP (mean Average Precision)：平均精度的平均值")
print("5. F1 Score：Precision和Recall的调和平均")

# 7. 边界框回归
print("\n7. 边界框回归")

print("边界框回归是目标检测中的重要技术，用于精确定位物体")
print("- 边界框表示：(x, y, w, h) 或 (x1, y1, x2, y2)")
print("- 回归目标：预测边界框与真实边界框之间的偏移")
print("- 损失函数：平滑L1损失、IoU损失等")

# 8. 非极大值抑制
print("\n8. 非极大值抑制")

print("非极大值抑制（NMS）用于消除冗余的检测结果")
print("- 步骤：")
print("  1. 按置信度排序检测结果")
print("  2. 选择置信度最高的检测框")
print("  3. 移除与该检测框IoU大于阈值的其他检测框")
print("  4. 重复步骤2-3，直到所有检测框都被处理")

# 9. 目标检测数据集
print("\n9. 目标检测数据集")

print("常见的目标检测数据集:")
print("1. COCO (Common Objects in Context)：包含80个类别")
print("2. Pascal VOC：包含20个类别")
print("3. ImageNet：包含1000个类别")
print("4. Open Images：包含600个类别")
print("5. Cityscapes：城市场景数据集")

# 10. 练习
print("\n10. 练习")

# 练习1: 使用不同的目标检测模型
print("练习1: 使用不同的目标检测模型")
print("- 尝试使用YOLOv5、YOLOv7、YOLOv8")
print("- 尝试使用Faster R-CNN")
print("- 尝试使用SSD")

# 练习2: 自定义目标检测
print("\n练习2: 自定义目标检测")
print("- 准备自定义数据集")
print("- 标注数据")
print("- 训练模型")
print("- 评估模型")

# 练习3: 目标跟踪
print("\n练习3: 目标跟踪")
print("- 尝试使用SORT算法")
print("- 尝试使用DeepSORT算法")
print("- 尝试使用ByteTrack算法")

# 练习4: 实时目标检测
print("\n练习4: 实时目标检测")
print("- 使用摄像头进行实时目标检测")
print("- 优化模型以提高推理速度")
print("- 部署到边缘设备")

print("\n=== 第75天学习示例结束 ===")
