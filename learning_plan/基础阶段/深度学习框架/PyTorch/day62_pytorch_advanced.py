#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第62天：PyTorch高级特性
深度学习框架学习示例
内容：PyTorch的高级特性、自定义层、模型部署等
"""

print("=== 第62天：PyTorch高级特性 ===")

# 1. 自定义层
print("\n1. 自定义层")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

# 自定义线性层
class LinearLayer(nn.Module):
    """自定义线性层"""
    
    def __init__(self, in_features, out_features):
        super(LinearLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 初始化权重和偏置
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
    
    def forward(self, x):
        """前向传播"""
        return F.linear(x, self.weight, self.bias)

# 测试自定义层
print("测试自定义线性层:")
linear_layer = LinearLayer(784, 10)
input_tensor = torch.randn(32, 784)
output_tensor = linear_layer(input_tensor)
print(f"输入形状: {input_tensor.shape}")
print(f"输出形状: {output_tensor.shape}")

# 2. 自定义激活函数
print("\n2. 自定义激活函数")

# 自定义激活函数
class CustomActivation(nn.Module):
    """自定义激活函数"""
    
    def forward(self, x):
        """前向传播"""
        return F.relu(x) + torch.sin(x)

# 测试自定义激活函数
print("测试自定义激活函数:")
custom_activation = CustomActivation()
x = torch.tensor([-1.0, 0.0, 1.0, 2.0])
y = custom_activation(x)
print(f"输入: {x.numpy()}")
print(f"输出: {y.numpy()}")

# 3. 自定义损失函数
print("\n3. 自定义损失函数")

# 自定义损失函数
class CustomLoss(nn.Module):
    """自定义损失函数"""
    
    def forward(self, y_true, y_pred):
        """前向传播"""
        mse_loss = F.mse_loss(y_true, y_pred)
        l1_loss = F.l1_loss(y_true, y_pred)
        return mse_loss + l1_loss

# 测试自定义损失函数
print("测试自定义损失函数:")
custom_loss = CustomLoss()
y_true = torch.tensor([1.0, 2.0, 3.0])
y_pred = torch.tensor([1.2, 1.8, 3.1])
loss = custom_loss(y_true, y_pred)
print(f"真实值: {y_true.numpy()}")
print(f"预测值: {y_pred.numpy()}")
print(f"损失值: {loss.item()}")

# 4. 模型保存和加载
print("\n4. 模型保存和加载")

# 定义一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 创建模型
model = SimpleModel()

# 保存模型
# 方法1: 保存整个模型
torch.save(model, 'entire_model.pth')
print("整个模型已保存")

# 方法2: 保存模型状态字典
torch.save(model.state_dict(), 'model_state_dict.pth')
print("模型状态字典已保存")

# 加载模型
# 方法1: 加载整个模型
loaded_model = torch.load('entire_model.pth')
print("整个模型已加载")

# 方法2: 加载模型状态字典
new_model = SimpleModel()
new_model.load_state_dict(torch.load('model_state_dict.pth'))
print("模型状态字典已加载")

# 5. 模型量化
print("\n5. 模型量化")

# 动态量化
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {nn.Linear},  # 要量化的层类型
    dtype=torch.qint8  # 量化类型
)
print("模型已动态量化")

# 6. 模型导出
print("\n6. 模型导出")

# 导出为ONNX格式
dummy_input = torch.randn(1, 1, 28, 28)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=['input'],
    output_names=['output']
)
print("模型已导出为ONNX格式")

# 7. 自动混合精度训练
print("\n7. 自动混合精度训练")

print("自动混合精度训练可以提高训练速度并减少内存使用")
print("- 使用torch.cuda.amp模块")
print("- 需要支持FP16的GPU")

# 示例代码
print("\n自动混合精度训练示例:")
print("from torch.cuda.amp import autocast, GradScaler")
print("\nscaler = GradScaler()")
print("\nfor epoch in range(epochs):")
print("    for batch in dataloader:")
print("        optimizer.zero_grad()")
print("        with autocast():")
print("            outputs = model(batch)")
print("            loss = criterion(outputs, targets)")
print("        scaler.scale(loss).backward()")
print("        scaler.step(optimizer)")
print("        scaler.update()")

# 8. 分布式训练
print("\n8. 分布式训练")

print("PyTorch支持多种分布式训练方法:")
print("- DataParallel: 单机多GPU")
print("- DistributedDataParallel: 多机多GPU")

# 示例：使用DataParallel
print("\n使用DataParallel的示例:")
print("if torch.cuda.device_count() > 1:")
print("    model = nn.DataParallel(model)")

# 9. 自定义训练循环
print("\n9. 自定义训练循环")

# 自定义训练循环示例
def custom_training_loop(model, train_loader, criterion, optimizer, epochs=5):
    """自定义训练循环"""
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
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
            
            if (batch_idx + 1) % 100 == 0:
                print(f'Batch {batch_idx+1}/{len(train_loader)}, Loss: {running_loss/(batch_idx+1):.3f}, Accuracy: {100.*correct/total:.3f}%')
        
        print(f'Epoch {epoch+1} finished. Loss: {running_loss/len(train_loader):.3f}, Accuracy: {100.*correct/total:.3f}%')

# 10. 模型部署
print("\n10. 模型部署")

print("PyTorch模型部署选项:")
print("1. PyTorch C++ API: 用于生产环境部署")
print("2. TorchScript: 将模型转换为序列化格式")
print("3. ONNX: 与其他框架和平台兼容")
print("4. TensorRT: NVIDIA的高性能推理库")
print("5. TorchServe: 模型服务框架")

# 示例：使用TorchScript
print("\n使用TorchScript的示例:")
print("# 跟踪模型")
print("traced_model = torch.jit.trace(model, dummy_input)")
print("\n# 保存跟踪模型")
print("traced_model.save('model.pt')")
print("\n# 加载跟踪模型")
print("loaded_model = torch.jit.load('model.pt')")

# 11. 练习
print("\n11. 练习")

# 练习1: 自定义更复杂的层
print("练习1: 自定义更复杂的层")
print("- 实现一个自定义卷积层")
print("- 实现一个自定义注意力层")

# 练习2: 模型优化
print("\n练习2: 模型优化")
print("- 尝试模型剪枝")
print("- 尝试知识蒸馏")
print("- 尝试模型量化")

# 练习3: 模型部署
print("\n练习3: 模型部署")
print("- 尝试使用TorchServe部署模型")
print("- 尝试使用Docker容器部署模型")

# 练习4: 高级训练技巧
print("\n练习4: 高级训练技巧")
print("- 尝试学习率调度器")
print("- 尝试早停")
print("- 尝试梯度裁剪")

print("\n=== 第62天学习示例结束 ===")
