#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第60天：PyTorch基础
深度学习框架学习示例
内容：PyTorch的基本概念、张量操作和自动微分
"""

print("=== 第60天：PyTorch基础 ===")

# 1. PyTorch基本概念
print("\n1. PyTorch基本概念")

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA设备数量: {torch.cuda.device_count()}")
    print(f"当前CUDA设备: {torch.cuda.current_device()}")
    print(f"CUDA设备名称: {torch.cuda.get_device_name()}")

print("PyTorch是一个基于Python的科学计算库，主要用于深度学习")
print("- 张量 (Tensor): 多维数组，类似于NumPy数组")
print("- 自动微分 (Autograd): 自动计算梯度")
print("- 神经网络 (Neural Networks): 构建和训练神经网络")
print("- 优化器 (Optimizers): 优化模型参数")
print("- 数据集和数据加载器 (Datasets and DataLoaders): 处理数据")

# 2. 张量的创建
print("\n2. 张量的创建")

# 创建张量
x = torch.tensor(42)
print(f"标量张量: {x}")
print(f"张量类型: {x.dtype}")
print(f"张量设备: {x.device}")

# 创建多维张量
y = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(f"\n二维张量:")
print(y)
print(f"张量形状: {y.shape}")
print(f"张量维度: {y.ndim}")
print(f"张量元素数量: {y.numel()}")

# 创建特殊张量
# 全零张量
zeros = torch.zeros((2, 3))
print(f"\n全零张量:")
print(zeros)

# 全一张量
ones = torch.ones((3, 2))
print(f"\n全一张量:")
print(ones)

# 随机张量
random_tensor = torch.randn((2, 3))
print(f"\n随机张量:")
print(random_tensor)

# 从NumPy数组创建
numpy_array = np.array([[1, 2, 3], [4, 5, 6]])
torch_tensor = torch.from_numpy(numpy_array)
print(f"\n从NumPy数组创建:")
print(torch_tensor)

# 转换为NumPy数组
tensor_to_numpy = torch_tensor.numpy()
print(f"\n转换为NumPy数组:")
print(tensor_to_numpy)

# 3. 张量操作
print("\n3. 张量操作")

# 基本算术操作
a = torch.tensor(5)
b = torch.tensor(3)
print(f"a = {a}, b = {b}")
print(f"a + b = {a + b}")
print(f"a - b = {a - b}")
print(f"a * b = {a * b}")
print(f"a / b = {a / b}")
print(f"a // b = {a // b}")
print(f"a % b = {a % b}")
print(f"a ** b = {a ** b}")

# 张量运算
x = torch.tensor([[1, 2], [3, 4]])
y = torch.tensor([[5, 6], [7, 8]])
print(f"\nx =\n{x}")
print(f"y =\n{y}")
print(f"x + y =\n{x + y}")
print(f"x * y =\n{x * y}")  # 元素级乘法
print(f"矩阵乘法 x @ y =\n{x @ y}")
print(f"x的转置 =\n{x.t()}")
print(f"x的和 =\n{x.sum()}")
print(f"x的均值 =\n{x.mean()}")
print(f"x的最大值 =\n{x.max()}")
print(f"x的最小值 =\n{x.min()}")

# 索引和切片
print(f"\n索引和切片:")
print(f"x[0] =\n{x[0]}")
print(f"x[:, 1] =\n{x[:, 1]}")
print(f"x[1, 0] =\n{x[1, 0]}")

# 张量变形
print(f"\n张量变形:")
print(f"原始形状: {x.shape}")
print(f"变形为 (4,): {x.view(4)}")
print(f"变形为 (4, 1): {x.view(4, 1)}")
print(f"变形为 (1, 4): {x.view(1, 4)}")
print(f"使用reshape: {x.reshape(4)}")

# 4. 自动微分
print("\n4. 自动微分")

# 定义需要求导的张量
x = torch.tensor(3.0, requires_grad=True)
print(f"x = {x}")
print(f"x.requires_grad = {x.requires_grad}")

# 计算y = x^2
with torch.no_grad():
    y = x ** 2
print(f"\n在no_grad()中计算y = x^2: {y}")
print(f"y.requires_grad = {y.requires_grad}")

# 重新计算y，这次追踪梯度
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2
print(f"\n追踪梯度计算y = x^2: {y}")
print(f"y.requires_grad = {y.requires_grad}")

# 计算梯度
y.backward()
print(f"x的梯度: {x.grad}")
print(f"预期梯度: 2x = 6")

# 多变量梯度
w = torch.tensor([1.0, 2.0], requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)
x = torch.tensor([4.0, 5.0])

with torch.no_grad():
    y = torch.dot(w, x) + b
print(f"\n在no_grad()中计算y = w·x + b: {y}")

# 重新计算，追踪梯度
w = torch.tensor([1.0, 2.0], requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)
x = torch.tensor([4.0, 5.0])
y = torch.dot(w, x) + b
print(f"追踪梯度计算y = w·x + b: {y}")

# 计算梯度
y.backward()
print(f"w的梯度: {w.grad}")
print(f"b的梯度: {b.grad}")
print(f"预期w的梯度: x = [4, 5]")
print(f"预期b的梯度: 1")

# 5. 神经网络基础
print("\n5. 神经网络基础")

# 定义一个简单的神经网络
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 创建模型
model = SimpleNN(input_size=2, hidden_size=4, output_size=1)
print("神经网络模型:")
print(model)

# 查看模型参数
print("\n模型参数:")
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

# 测试模型
input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
output = model(input_tensor)
print(f"\n输入:")
print(input_tensor)
print(f"输出:")
print(output)

# 6. 损失函数和优化器
print("\n6. 损失函数和优化器")

# 定义损失函数
criterion = nn.MSELoss()

# 定义优化器
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 示例：前向传播和反向传播
input_tensor = torch.tensor([[1.0, 2.0]])
target = torch.tensor([[5.0]])

# 前向传播
output = model(input_tensor)
print(f"输入: {input_tensor}")
print(f"预测输出: {output}")
print(f"目标输出: {target}")

# 计算损失
loss = criterion(output, target)
print(f"损失: {loss}")

# 反向传播
optimizer.zero_grad()  # 清零梯度
loss.backward()        # 计算梯度
optimizer.step()       # 更新参数

# 再次前向传播
output = model(input_tensor)
print(f"更新后预测输出: {output}")

# 7. 数据加载
print("\n7. 数据加载")

from torch.utils.data import Dataset, DataLoader

# 自定义数据集
class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# 创建数据集
x = torch.randn(100, 2)
y = torch.randn(100, 1)
dataset = CustomDataset(x, y)

# 创建数据加载器
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 遍历数据加载器
print("数据加载器示例:")
for batch_idx, (data, target) in enumerate(dataloader):
    print(f"批次 {batch_idx+1}")
    print(f"数据形状: {data.shape}")
    print(f"目标形状: {target.shape}")
    if batch_idx == 2:  # 只显示前3个批次
        break

# 8. 线性回归示例
print("\n8. 线性回归示例")

# 生成模拟数据
np.random.seed(42)
x = np.random.randn(100, 1)
y = 2 * x + 1 + np.random.randn(100, 1) * 0.1

# 转换为张量
x_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# 创建线性回归模型
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.linear(x)

model = LinearRegression()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
epochs = 1000
for epoch in range(epochs):
    # 前向传播
    outputs = model(x_tensor)
    loss = criterion(outputs, y_tensor)
    
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 每100个epoch打印一次
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 查看模型参数
print(f"\n训练完成!")
for name, param in model.named_parameters():
    print(f"{name}: {param.item():.4f}")

# 9. GPU加速
print("\n9. GPU加速")

if torch.cuda.is_available():
    # 将模型移动到GPU
    model.to('cuda')
    
    # 将数据移动到GPU
    x_tensor = x_tensor.to('cuda')
    y_tensor = y_tensor.to('cuda')
    
    print("模型和数据已移动到GPU")
    print(f"模型设备: {next(model.parameters()).device}")
    print(f"数据设备: {x_tensor.device}")
else:
    print("CUDA不可用，使用CPU")

# 10. 练习
print("\n10. 练习")

# 练习1: 创建不同类型的张量
print("练习1: 创建不同类型的张量")

# 创建一个3x3的单位矩阵
identity_matrix = torch.eye(3)
print(f"3x3单位矩阵:")
print(identity_matrix)

# 创建一个形状为(2, 3, 4)的随机张量
random_3d = torch.randn(2, 3, 4)
print(f"\n2x3x4随机张量形状: {random_3d.shape}")
print(f"张量值:")
print(random_3d)

# 练习2: 张量运算
print("\n练习2: 张量运算")

# 创建两个张量
A = torch.tensor([[1, 2], [3, 4]])
B = torch.tensor([[5, 6], [7, 8]])

print(f"A =\n{A}")
print(f"B =\n{B}")
print(f"A + B =\n{A + B}")
print(f"A - B =\n{A - B}")
print(f"A * B =\n{A * B}")
print(f"A @ B =\n{A @ B}")
print(f"A的逆矩阵 =\n{torch.inverse(A)}")
print(f"A的行列式 =\n{torch.det(A)}")

# 练习3: 自动微分
print("\n练习3: 自动微分")

# 定义一个复杂函数
x = torch.tensor(2.0, requires_grad=True)

with torch.no_grad():
    y = x ** 3 + 2 * x ** 2 + 3 * x + 4
print(f"在no_grad()中计算y = x³ + 2x² + 3x + 4: {y}")
print(f"y.requires_grad = {y.requires_grad}")

# 重新计算，追踪梯度
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3 + 2 * x ** 2 + 3 * x + 4
y.backward()
print(f"\n追踪梯度计算y = x³ + 2x² + 3x + 4: {y}")
print(f"x的梯度: {x.grad}")
print(f"预期梯度: 3x² + 4x + 3 = {3*(2**2) + 4*2 + 3}")

print("\n=== 第60天学习示例结束 ===")
