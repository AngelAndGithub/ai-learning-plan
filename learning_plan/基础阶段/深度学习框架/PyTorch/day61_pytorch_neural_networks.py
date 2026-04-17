#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第61天：PyTorch神经网络
深度学习框架学习示例
内容：PyTorch的神经网络构建、训练和评估
"""

print("=== 第61天：PyTorch神经网络 ===")

# 1. 神经网络基本概念
print("\n1. 神经网络基本概念")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

print("PyTorch中的神经网络是通过继承nn.Module类来构建的")
print("- 层 (Layer): 神经网络的基本构建块")
print("- 激活函数 (Activation Function): 引入非线性")
print("- 损失函数 (Loss Function): 衡量模型性能")
print("- 优化器 (Optimizer): 更新模型参数")
print("- 数据加载器 (DataLoader): 批量加载数据")

# 2. 构建神经网络
print("\n2. 构建神经网络")

# 定义一个简单的全连接神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 第一层：输入层到隐藏层
        self.fc1 = nn.Linear(784, 128)
        # 第二层：隐藏层
        self.fc2 = nn.Linear(128, 64)
        # 第三层：隐藏层到输出层
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        # 前向传播
        x = x.view(-1, 784)  # 展平图像
        x = F.relu(self.fc1(x))  # 激活函数
        x = F.relu(self.fc2(x))  # 激活函数
        x = self.fc3(x)  # 输出层
        return x

# 创建模型
model = Net()
print("神经网络模型:")
print(model)

# 查看模型参数
print("\n模型参数:")
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

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
print(f"训练批次数量: {len(train_loader)}")
print(f"测试批次数量: {len(test_loader)}")

# 查看数据
print("\n查看数据示例:")
dataiter = iter(train_loader)
images, labels = dataiter.next()
print(f"图像批次形状: {images.shape}")
print(f"标签批次形状: {labels.shape}")

# 显示图像
plt.figure(figsize=(10, 4))
for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.imshow(images[i].squeeze().numpy(), cmap='gray')
    plt.title(f"标签: {labels[i].item()}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# 4. 损失函数和优化器
print("\n4. 损失函数和优化器")

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 定义优化器
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

print("损失函数: CrossEntropyLoss")
print("优化器: SGD with momentum=0.9")

# 5. 模型训练
print("\n5. 模型训练")

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
        
        # 统计损失和准确率
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
epochs = 10
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

# 6. 训练结果分析
print("\n6. 训练结果分析")

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

# 7. 模型评估
print("\n7. 模型评估")

# 最终评估
test_loss, test_acc = test(model, test_loader, criterion)
print(f"最终测试损失: {test_loss:.3f}")
print(f"最终测试准确率: {test_acc:.3f}%")

# 查看错误分类的样本
print("\n查看错误分类的样本:")
model.eval()
incorrect = []
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        mask = predicted.ne(targets)
        if mask.sum() > 0:
            incorrect.extend([(inputs[i], targets[i], predicted[i]) for i in range(len(inputs)) if mask[i]])
        if len(incorrect) >= 8:
            break

# 显示错误分类的样本
plt.figure(figsize=(10, 4))
for i, (img, target, pred) in enumerate(incorrect[:8]):
    plt.subplot(2, 4, i+1)
    plt.imshow(img.squeeze().numpy(), cmap='gray')
    plt.title(f"真实: {target.item()}, 预测: {pred.item()}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# 8. 保存和加载模型
print("\n8. 保存和加载模型")

# 保存模型
torch.save(model.state_dict(), 'mnist_model.pth')
print("模型已保存到 'mnist_model.pth'")

# 加载模型
model = Net()
model.load_state_dict(torch.load('mnist_model.pth'))
print("模型已加载")

# 验证加载的模型
test_loss, test_acc = test(model, test_loader, criterion)
print(f"加载模型的测试准确率: {test_acc:.3f}%")

# 9. 卷积神经网络
print("\n9. 卷积神经网络")

# 定义卷积神经网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 第一层卷积
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # 池化
        self.pool = nn.MaxPool2d(2, 2)
        # 第二层卷积
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 创建CNN模型
cnn_model = CNN()
print("卷积神经网络模型:")
print(cnn_model)

# 训练CNN模型
optimizer = optim.SGD(cnn_model.parameters(), lr=0.01, momentum=0.9)

print("\n训练卷积神经网络...")
cnn_train_losses = []
cnn_train_accuracies = []
cnn_test_losses = []
cnn_test_accuracies = []

for epoch in range(5):  # 只训练5个epoch
    print(f"\n=== Epoch {epoch+1}/5 ===")
    
    # 训练
    train_loss, train_acc = train(cnn_model, train_loader, criterion, optimizer, epoch)
    cnn_train_losses.append(train_loss)
    cnn_train_accuracies.append(train_acc)
    
    # 测试
    test_loss, test_acc = test(cnn_model, test_loader, criterion)
    cnn_test_losses.append(test_loss)
    cnn_test_accuracies.append(test_acc)
    
    print(f"\n训练损失: {train_loss:.3f}, 训练准确率: {train_acc:.3f}%")
    print(f"测试损失: {test_loss:.3f}, 测试准确率: {test_acc:.3f}%")

# 评估CNN模型
print("\n卷积神经网络评估:")
test_loss, test_acc = test(cnn_model, test_loader, criterion)
print(f"CNN测试准确率: {test_acc:.3f}%")

# 10. 练习
print("\n10. 练习")

# 练习1: 调整模型架构
print("练习1: 调整模型架构")
print("- 尝试不同的隐藏层数量")
print("- 尝试不同的神经元数量")
print("- 尝试不同的激活函数")

# 练习2: 调整超参数
print("\n练习2: 调整超参数")
print("- 尝试不同的学习率")
print("- 尝试不同的批量大小")
print("- 尝试不同的优化器")

# 练习3: 数据增强
print("\n练习3: 数据增强")
print("- 添加数据增强变换")
print("- 尝试旋转、平移、缩放等变换")

# 示例：数据增强
print("\n数据增强示例:")
data_augmentation = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

print("数据增强变换:")
print(data_augmentation)

# 练习4: 迁移学习
print("\n练习4: 迁移学习")
print("- 使用预训练模型")
print("- 微调模型")

print("\n=== 第61天学习示例结束 ===")
