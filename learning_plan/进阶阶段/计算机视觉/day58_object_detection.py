#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第58天：目标检测
计算机视觉学习示例
内容：目标检测的基本概念、YOLO、Faster R-CNN、SSD
"""

print("=== 第58天：目标检测 ===")

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. 目标检测概述
print("\n1. 目标检测概述")

print("目标检测是计算机视觉的重要任务，用于识别和定位图像中的物体")
print("- 任务：不仅要识别物体的类别，还要确定物体的位置")
print("- 输出：物体的边界框和类别标签")
print("- 应用：自动驾驶、安防监控、人脸识别等")

# 2. 目标检测方法
print("\n2. 目标检测方法")

print("常见的目标检测方法:")
print("- 传统方法：Haar级联分类器、HOG+SVM")
print("- 基于区域的方法：R-CNN、Fast R-CNN、Faster R-CNN")
print("- 单阶段方法：YOLO、SSD、RetinaNet")
print("- anchor-free方法：CornerNet、CenterNet")

# 3. 数据集准备
print("\n3. 数据集准备")

# 创建示例图像
if not os.path.exists('images'):
    os.makedirs('images')

# 创建一个包含多个物体的示例图像
img = np.zeros((400, 600, 3), dtype=np.uint8)
img[:] = [240, 240, 240]  # 背景色

# 绘制圆形（代表球）
cv2.circle(img, (100, 100), 50, (0, 0, 255), -1)

# 绘制矩形（代表汽车）
cv2.rectangle(img, (200, 50), (300, 150), (0, 255, 0), -1)

# 绘制三角形（代表标志）
points = np.array([[400, 50], [450, 150], [350, 150]], np.int32)
cv2.fillPoly(img, [points], (255, 0, 0))

# 保存示例图像
cv2.imwrite('images/objects.jpg', img)
print("示例图像已创建为 images/objects.jpg")

# 读取并显示图像
image = cv2.imread('images/objects.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 6))
plt.imshow(image_rgb)
plt.title('示例图像')
plt.axis('off')
plt.savefig('images/displayed_objects.png')
print("显示的图像已保存为 images/displayed_objects.png")

# 4. Haar级联分类器
print("\n4. Haar级联分类器")

print("Haar级联分类器是一种传统的目标检测方法")
print("- 原理：基于Haar特征和AdaBoost算法")
print("- 优点：速度快，适合实时应用")
print("- 缺点：精度较低，对光照变化敏感")

# 加载预训练的人脸检测模型
hface_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# 创建包含人脸的示例图像
face_img = np.zeros((300, 400, 3), dtype=np.uint8)
face_img[:] = [240, 240, 240]  # 背景色

# 绘制人脸
cv2.circle(face_img, (200, 150), 50, (255, 220, 180), -1)

# 绘制眼睛
cv2.circle(face_img, (180, 130), 10, (0, 0, 0), -1)
cv2.circle(face_img, (220, 130), 10, (0, 0, 0), -1)

# 绘制嘴巴
cv2.ellipse(face_img, (200, 170), (20, 10), 0, 0, 180, (0, 0, 0), 2)

# 保存人脸图像
cv2.imwrite('images/face.jpg', face_img)
print("人脸示例图像已创建为 images/face.jpg")

# 读取并转换为灰度图像
face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

# 检测人脸
faces = face_cascade.detectMultiScale(face_gray, 1.3, 5)

# 绘制检测结果
for (x, y, w, h) in faces:
    cv2.rectangle(face_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    roi_gray = face_gray[y:y+h, x:x+w]
    roi_color = face_img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)

# 转换为RGB并显示
face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(8, 6))
plt.imshow(face_img_rgb)
plt.title('Haar级联分类器人脸检测')
plt.axis('off')
plt.savefig('images/face_detection.png')
print("人脸检测结果已保存为 images/face_detection.png")

# 5. YOLO（You Only Look Once）
print("\n5. YOLO（You Only Look Once）")

print("YOLO是一种单阶段目标检测方法")
print("- 原理：将目标检测问题转化为回归问题")
print("- 优点：速度快，适合实时应用")
print("- 缺点：小目标检测效果较差")

# 加载预训练的YOLO模型
def load_yolo():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers

def detect_objects(img, net, output_layers):
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return boxes, confidences, class_ids, indexes

def draw_labels(boxes, confidences, class_ids, indexes, classes, img):
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label + " " + str(round(confidence, 2)), (x, y + 30), font, 3, color, 3)
    return img

# 注意：YOLO模型文件较大，这里只是展示代码结构
print("YOLO模型代码结构已展示，实际使用需要下载模型文件")

# 6. Faster R-CNN
print("\n6. Faster R-CNN")

print("Faster R-CNN是一种基于区域的目标检测方法")
print("- 原理：使用Region Proposal Network (RPN)生成候选区域")
print("- 优点：精度高")
print("- 缺点：速度较慢")

# 7. SSD（Single Shot MultiBox Detector）
print("\n7. SSD（Single Shot MultiBox Detector）")

print("SSD是一种单阶段目标检测方法")
print("- 原理：在不同尺度的特征图上进行检测")
print("- 优点：速度快，精度较高")
print("- 缺点：小目标检测效果一般")

# 8. 目标检测的评估指标
print("\n8. 目标检测的评估指标")

print("目标检测的评估指标:")
print("- IoU（交并比）：预测边界框与真实边界框的重叠程度")
print("- mAP（平均精度）：不同IoU阈值下的平均精度")
print("- 召回率：正确检测的物体占总物体的比例")
print("- 精确率：正确检测的物体占总检测结果的比例")

# 9. 目标检测的挑战
print("\n9. 目标检测的挑战")

print("目标检测的挑战:")
print("- 小目标检测：小物体难以检测")
print("- 遮挡：物体被部分遮挡")
print("- 光照变化：光照条件影响检测效果")
print("- 视角变化：物体从不同角度观察")
print("- 类别不平衡：某些类别的样本较少")

# 10. 目标检测的应用
print("\n10. 目标检测的应用")

print("目标检测的主要应用:")
print("- 自动驾驶：检测车辆、行人、交通标志")
print("- 安防监控：检测异常行为、可疑人员")
print("- 人脸识别：检测和识别人脸")
print("- 零售分析：检测货架上的商品")
print("- 医学影像：检测病变区域")

# 11. 练习
print("\n11. 练习")

# 练习1: 使用Haar级联分类器
print("练习1: 使用Haar级联分类器")
print("- 使用Haar级联分类器检测人脸")
print("- 测试不同图像的检测效果")

# 练习2: 使用YOLO
print("\n练习2: 使用YOLO")
print("- 下载YOLO模型文件")
print("- 使用YOLO检测图像中的物体")

# 练习3: 目标检测模型训练
print("\n练习3: 目标检测模型训练")
print("- 准备自定义数据集")
print("- 训练目标检测模型")

# 练习4: 目标跟踪
print("\n练习4: 目标跟踪")
print("- 实现简单的目标跟踪算法")
print("- 测试跟踪效果")

# 练习5: 多目标检测
print("\n练习5: 多目标检测")
print("- 检测图像中的多个物体")
print("- 评估检测效果")

print("\n=== 第58天学习示例结束 ===")
