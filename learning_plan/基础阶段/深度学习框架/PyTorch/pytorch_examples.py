import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 1. 张量操作示例
def tensor_operations():
    """张量操作示例"""
    print("=== 张量操作示例 ===")
    
    # 创建张量
    scalar = torch.tensor(5.0)
    vector = torch.tensor([1.0, 2.0, 3.0])
    matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    tensor_3d = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    
    print(f"标量: {scalar}")
    print(f"向量: {vector}")
    print(f"矩阵: {matrix}")
    print(f"3D张量: {tensor_3d}")
    
    # 张量属性
    print(f"标量形状: {scalar.shape}")
    print(f"向量形状: {vector.shape}")
    print(f"矩阵形状: {matrix.shape}")
    print(f"3D张量形状: {tensor_3d.shape}")
    
    # 张量运算
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])
    
    print(f"a + b: {torch.add(a, b)}")
    print(f"a - b: {torch.sub(a, b)}")
    print(f"a * b: {torch.mul(a, b)}")
    print(f"a / b: {torch.div(a, b)}")
    
    # 矩阵乘法
    c = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    d = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    print(f"矩阵乘法: {torch.matmul(c, d)}")
    
    # 形状操作
    e = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(f"原始形状: {e.shape}")
    print(f"reshape: {torch.reshape(e, (3, 2))}")
    print(f"转置: {torch.transpose(e, 0, 1)}")
    
    # 设备操作
    if torch.cuda.is_available():
        device = torch.device("cuda")
        tensor = torch.tensor([1.0, 2.0]).to(device)
        print(f"GPU张量: {tensor}")
        print(f"张量设备: {tensor.device}")
    
    return scalar, vector, matrix, tensor_3d

# 2. 自动微分示例
def automatic_differentiation():
    """自动微分示例"""
    print("\n=== 自动微分示例 ===")
    
    # 单个变量的梯度
    x = torch.tensor(3.0, requires_grad=True)
    y = x ** 2
    y.backward()
    print(f"dy/dx = {x.grad}")
    
    # 多个变量的梯度
    w = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(1.0, requires_grad=True)
    x = torch.tensor(5.0)
    y = w * x + b
    y.backward()
    print(f"dy/dw = {w.grad}")
    print(f"dy/db = {b.grad}")
    
    # 复杂计算的梯度
    x = torch.tensor(1.0, requires_grad=True)
    y = torch.tensor(2.0, requires_grad=True)
    z = x**2 + y**2
    z.backward()
    print(f"dz/dx = {x.grad}")
    print(f"dz/dy = {y.grad}")
    
    return x.grad, y.grad, w.grad, b.grad

# 3. 线性回归示例
def linear_regression():
    """线性回归示例"""
    print("\n=== 线性回归示例 ===")
    
    # 生成合成数据
    X = torch.randn(1000, 1)
    y = 2 * X + 3 + torch.randn(1000, 1) * 0.1
    
    # 可视化数据
    plt.scatter(X.numpy(), y.numpy(), alpha=0.5)
    plt.title('线性回归数据')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.savefig('pytorch_linear_regression_data.png')
    plt.close()
    
    # 定义模型
    class LinearRegression(nn.Module):
        def __init__(self):
            super(LinearRegression, self).__init__()
            self.linear = nn.Linear(1, 1)
        
        def forward(self, x):
            return self.linear(x)
    
    model = LinearRegression()
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # 训练模型
    num_epochs = 100
    for epoch in range(num_epochs):
        # 前向传播
        outputs = model(X)
        loss = criterion(outputs, y)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    # 评估模型
    with torch.no_grad():
        y_pred = model(X)
        loss = criterion(y_pred, y)
        print(f"训练损失: {loss.item():.4f}")
    
    # 可视化结果
    plt.scatter(X.numpy(), y.numpy(), alpha=0.5, label='真实值')
    plt.plot(X.numpy(), y_pred.numpy(), color='red', label='预测值')
    plt.title('线性回归结果')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.savefig('pytorch_linear_regression_result.png')
    plt.close()
    
    # 获取模型参数
    for name, param in model.named_parameters():
        print(f"{name}: {param.data}")
    
    return model

# 4. MNIST手写数字分类示例
def mnist_classification():
    """MNIST手写数字分类示例"""
    print("\n=== MNIST手写数字分类示例 ===")
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 加载数据集
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 定义模型
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(28*28, 128)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.2)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 10)
        
        def forward(self, x):
            x = self.flatten(x)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.relu(x)
            x = self.fc3(x)
            return x
    
    model = NeuralNetwork()
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 计算统计信息
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f'Batch [{batch_idx+1}/{len(train_loader)}], Loss: {running_loss/(batch_idx+1):.4f}, Accuracy: {100.*correct/total:.2f}%')
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100.*correct/total:.2f}%')
    
    # 评估模型
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    print(f"测试损失: {test_loss/len(test_loader):.4f}")
    print(f"测试准确率: {100.*correct/total:.2f}%")
    
    # 保存模型
    torch.save(model.state_dict(), 'mnist_model.pth')
    print("模型已保存为 mnist_model.pth")
    
    return model

# 5. CIFAR-10图像分类示例
def cifar10_classification():
    """CIFAR-10图像分类示例"""
    print("\n=== CIFAR-10图像分类示例 ===")
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 加载数据集
    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 定义模型
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
            self.fc1 = nn.Linear(64 * 4 * 4, 64)
            self.fc2 = nn.Linear(64, 10)
        
        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = F.relu(self.conv3(x))
            x = x.view(-1, 64 * 4 * 4)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    model = CNN()
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 计算统计信息
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f'Batch [{batch_idx+1}/{len(train_loader)}], Loss: {running_loss/(batch_idx+1):.4f}, Accuracy: {100.*correct/total:.2f}%')
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100.*correct/total:.2f}%')
    
    # 评估模型
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    print(f"测试损失: {test_loss/len(test_loader):.4f}")
    print(f"测试准确率: {100.*correct/total:.2f}%")
    
    # 保存模型
    torch.save(model.state_dict(), 'cifar10_model.pth')
    print("模型已保存为 cifar10_model.pth")
    
    return model

# 6. 自定义数据集示例
class CustomDataset(Dataset):
    """自定义数据集"""
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, target

def custom_dataset_example():
    """自定义数据集示例"""
    print("\n=== 自定义数据集示例 ===")
    
    # 生成合成数据
    data = torch.randn(100, 3, 32, 32)
    targets = torch.randint(0, 10, (100,))
    
    # 创建自定义数据集
    dataset = CustomDataset(data, targets)
    print(f"数据集大小: {len(dataset)}")
    
    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 遍历数据加载器
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        print(f"批次 {batch_idx+1}: 输入形状={inputs.shape}, 目标形状={targets.shape}")
    
    return dataset

# 7. 模型保存与加载示例
def model_save_load():
    """模型保存与加载示例"""
    print("\n=== 模型保存与加载示例 ===")
    
    # 定义模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc = nn.Linear(10, 1)
        
        def forward(self, x):
            return self.fc(x)
    
    # 创建模型
    model = SimpleModel()
    print("原始模型参数:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.data}")
    
    # 保存模型权重
    torch.save(model.state_dict(), 'model_weights.pth')
    print("模型权重已保存")
    
    # 创建新模型
    new_model = SimpleModel()
    print("\n新模型参数:")
    for name, param in new_model.named_parameters():
        print(f"{name}: {param.data}")
    
    # 加载模型权重
    new_model.load_state_dict(torch.load('model_weights.pth'))
    print("\n加载后的模型参数:")
    for name, param in new_model.named_parameters():
        print(f"{name}: {param.data}")
    
    # 保存整个模型
    torch.save(model, 'model.pth')
    print("\n整个模型已保存")
    
    # 加载整个模型
    loaded_model = torch.load('model.pth')
    print("整个模型已加载")
    
    return model, new_model, loaded_model

# 8. 迁移学习示例
def transfer_learning_example():
    """迁移学习示例"""
    print("\n=== 迁移学习示例 ===")
    
    # 导入预训练模型
    from torchvision import models
    
    # 加载预训练的ResNet18模型
    model = models.resnet18(pretrained=True)
    print("预训练模型已加载")
    
    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
    
    # 替换最后一层
    num_classes = 10
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    print("模型最后一层已替换")
    
    # 打印模型结构
    print("\n模型结构:")
    print(model)
    
    return model

# 9. 模型部署示例
def model_deployment():
    """模型部署示例"""
    print("\n=== 模型部署示例 ===")
    
    # 定义模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc = nn.Linear(10, 1)
        
        def forward(self, x):
            return self.fc(x)
    
    # 创建模型
    model = SimpleModel()
    
    # 创建示例输入
    dummy_input = torch.randn(1, 10)
    
    # 导出为ONNX
    torch.onnx.export(model, dummy_input, 'model.onnx')
    print("模型已导出为ONNX格式")
    
    # 使用TorchScript跟踪模型
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save('model.pt')
    print("模型已导出为TorchScript格式")
    
    # 加载TorchScript模型
    loaded_model = torch.jit.load('model.pt')
    print("TorchScript模型已加载")
    
    # 测试加载的模型
    test_input = torch.randn(1, 10)
    output = loaded_model(test_input)
    print(f"测试输入: {test_input}")
    print(f"模型输出: {output}")
    
    return model, traced_model, loaded_model

# 10. 混合精度训练示例
def mixed_precision_training():
    """混合精度训练示例"""
    print("\n=== 混合精度训练示例 ===")
    
    # 检查是否支持混合精度
    if torch.cuda.is_available():
        print("CUDA可用，支持混合精度训练")
        
        # 定义模型
        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.fc1 = nn.Linear(100, 50)
                self.fc2 = nn.Linear(50, 10)
            
            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                return x
        
        model = SimpleModel().cuda()
        
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # 创建AMP梯度缩放器
        scaler = torch.cuda.amp.GradScaler()
        
        # 生成合成数据
        X = torch.randn(100, 100).cuda()
        y = torch.randint(0, 10, (100,)).cuda()
        
        # 训练循环
        for epoch in range(5):
            # 前向传播
            with torch.cuda.amp.autocast():
                outputs = model(X)
                loss = criterion(outputs, y)
            
            # 反向传播和优化
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
        
        print("混合精度训练完成")
        return model
    else:
        print("CUDA不可用，无法进行混合精度训练")
        return None

if __name__ == "__main__":
    # 运行所有示例
    tensor_operations()
    automatic_differentiation()
    linear_regression()
    mnist_classification()
    cifar10_classification()
    custom_dataset_example()
    model_save_load()
    transfer_learning_example()
    model_deployment()
    mixed_precision_training()
    
    print("\n所有示例运行完成！")