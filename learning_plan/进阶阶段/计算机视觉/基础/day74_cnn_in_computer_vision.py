#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第74天：卷积神经网络在计算机视觉中的应用
计算机视觉学习示例
内容：卷积神经网络的基本原理、常见架构和应用
"""

print("=== 第74天：卷积神经网络在计算机视觉中的应用 ===")

# 1. 卷积神经网络基本原理
print("\n1. 卷积神经网络基本原理")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

print("卷积神经网络（CNN）是专门为处理网格状数据（如图像）设计的神经网络")
print("- 卷积层：提取局部特征")
print("- 池化层：降低特征图维度")
print("- 全连接层：进行分类")
print("- 激活函数：引入非线性")

# 2. 简单CNN模型
print("\n2. 简单CNN模型")

# 定义一个简单的CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 第一层卷积
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        # 第二层卷积
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        # 第一层卷积 + 激活 + 池化
        x = self.pool(F.relu(self.conv1(x)))
        # 第二层卷积 + 激活 + 池化
        x = self.pool(F.relu(self.conv2(x)))
        # 展平
        x = x.view(-1, 64 * 7 * 7)
        # 全连接层 + 激活
        x = F.relu(self.fc1(x))
        # 输出层
        x = self.fc2(x)
        return x

# 创建模型
model = SimpleCNN()
print("简单CNN模型:")
print(model)

# 3. 数据加载和预处理
print("\n3. 数据加载和预处理")

# 定义数据变换
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize((0.1307,), (0.3081,))  # 归一化
])

# 加载MNIST数据集
train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)

# 创建数据加载器
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"训练数据集大小: {len(train_dataset)}")
print(f"测试数据集大小: {len(test_dataset)}")
print(f"批大小: {batch_size}")

# 4. 模型训练
print("\n4. 模型训练")

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练函数
def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # 零化梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # 每100个批次打印一次
        if (batch_idx + 1) % 100 == 0:
            print(f'[Epoch {epoch+1}, Batch {batch_idx+1}] Loss: {running_loss/(batch_idx+1):.3f}, Accuracy: {100.*correct/total:.3f}%')
    
    return running_loss / len(train_loader), 100.*correct/total

# 测试函数
def test(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return running_loss / len(test_loader), 100.*correct/total

# 训练模型
epochs = 5
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

for epoch in range(epochs):
    print(f"\n=== Epoch {epoch+1}/{epochs} ===")
    
    # 训练
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, epoch)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    
    # 测试
    test_loss, test_acc = test(model, test_loader, criterion)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)
    
    print(f"\n训练损失: {train_loss:.3f}, 训练准确率: {train_acc:.3f}%")
    print(f"测试损失: {test_loss:.3f}, 测试准确率: {test_acc:.3f}%")

# 5. 训练结果分析
print("\n5. 训练结果分析")

# 绘制训练曲线
plt.figure(figsize=(12, 4))

# 损失曲线
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), train_losses, label='训练损失')
plt.plot(range(1, epochs+1), test_losses, label='测试损失')
plt.title('损失曲线')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 准确率曲线
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), train_accuracies, label='训练准确率')
plt.plot(range(1, epochs+1), test_accuracies, label='测试准确率')
plt.title('准确率曲线')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.show()

# 6. 模型评估
print("\n6. 模型评估")

# 最终评估
test_loss, test_acc = test(model, test_loader, criterion)
print(f"最终测试损失: {test_loss:.3f}")
print(f"最终测试准确率: {test_acc:.3f}%")

# 查看预测结果
print("\n查看预测结果:")
model.eval()
dataiter = iter(test_loader)
images, labels = dataiter.next()
outputs = model(images)
_, predicted = outputs.max(1)

# 显示预测结果
plt.figure(figsize=(10, 4))
for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.imshow(images[i].squeeze().numpy(), cmap='gray')
    plt.title(f"真实: {labels[i].item()}, 预测: {predicted[i].item()}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# 7. 常见CNN架构
print("\n7. 常见CNN架构")

print("常见的CNN架构:")
print("1. LeNet: 最早的CNN架构之一，用于手写数字识别")
print("2. AlexNet: 2012年ImageNet竞赛冠军， deeper and wider")
print("3. VGGNet: 更深的网络，使用更小的卷积核")
print("4. GoogLeNet: 使用Inception模块")
print("5. ResNet: 使用残差连接，解决深层网络的梯度消失问题")
print("6. DenseNet: 密集连接网络")
print("7. EfficientNet: 高效网络设计")

# 8. 迁移学习
print("\n8. 迁移学习")

print("迁移学习是利用预训练模型进行新任务学习的方法")
print("- 特征提取: 使用预训练模型的特征提取部分")
print("- 微调: 调整预训练模型的部分层")
print("- 预训练模型: 如VGG、ResNet、EfficientNet等")

# 示例：使用预训练模型
print("\n使用预训练模型的示例:")
print("from torchvision import models")
print("\n# 加载预训练的ResNet18")
print("model = models.resnet18(pretrained=True)")
print("\n# 替换分类层")
print("num_ftrs = model.fc.in_features")
print("model.fc = nn.Linear(num_ftrs, 10)")

# 9. 数据增强
print("\n9. 数据增强")

print("数据增强是提高模型泛化能力的重要方法")
print("- 旋转、平移、缩放")
print("- 翻转、裁剪")
print("- 亮度、对比度调整")
print("- 随机擦除")

# 示例：数据增强
print("\n数据增强示例:")
data_augmentation = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

print("数据增强变换:")
print(data_augmentation)

# 10. 练习
print("\n10. 练习")

# 练习1: 构建不同的CNN架构
print("练习1: 构建不同的CNN架构")
print("- 尝试增加卷积层数量")
print("- 尝试不同的池化策略")
print("- 尝试不同的激活函数")

# 练习2: 调整超参数
print("\n练习2: 调整超参数")
print("- 尝试不同的学习率")
print("- 尝试不同的批量大小")
print("- 尝试不同的优化器")

# 练习3: 迁移学习
print("\n练习3: 迁移学习")
print("- 尝试使用不同的预训练模型")
print("- 尝试不同的微调策略")

# 练习4: 数据增强
print("\n练习4: 数据增强")
print("- 尝试不同的数据增强方法")
print("- 评估数据增强对模型性能的影响")

print("\n=== 第74天学习示例结束 ===")
