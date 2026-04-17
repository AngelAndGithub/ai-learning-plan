import numpy as np
import matplotlib.pyplot as plt

# 1. 数值微分实现
def numerical_derivative(f, x, h=1e-6):
    """计算函数f在x处的数值导数"""
    return (f(x + h) - f(x - h)) / (2 * h)

def numerical_gradient(f, x, h=1e-6):
    """计算函数f在x处的数值梯度"""
    grad = np.zeros_like(x)
    for i in range(x.size):
        x_plus = x.copy()
        x_plus[i] += h
        x_minus = x.copy()
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad

# 2. 梯度下降算法实现
def gradient_descent(f, grad_f, x0, learning_rate=0.1, max_iter=1000, tol=1e-6):
    """梯度下降算法"""
    x = x0
    loss_history = []
    for i in range(max_iter):
        loss = f(x)
        loss_history.append(loss)
        grad = grad_f(x)
        x_new = x - learning_rate * grad
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return x, loss_history, i

# 3. 动量法实现
def momentum_optimizer(f, grad_f, x0, learning_rate=0.1, momentum=0.9, max_iter=1000, tol=1e-6):
    """动量优化器"""
    x = x0
    v = np.zeros_like(x)
    loss_history = []
    for i in range(max_iter):
        loss = f(x)
        loss_history.append(loss)
        grad = grad_f(x)
        v = momentum * v - learning_rate * grad
        x_new = x + v
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return x, loss_history, i

# 4. Adam优化器实现
def adam_optimizer(f, grad_f, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, max_iter=1000, tol=1e-6):
    """Adam优化器"""
    x = x0
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    loss_history = []
    for i in range(1, max_iter + 1):
        loss = f(x)
        loss_history.append(loss)
        grad = grad_f(x)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        m_hat = m / (1 - beta1**i)
        v_hat = v / (1 - beta2**i)
        x_new = x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return x, loss_history, i

# 5. 线性回归实现
def linear_regression(X, y, optimizer='gd', learning_rate=0.01, max_iter=1000, tol=1e-6):
    """线性回归模型"""
    # 添加偏置项
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    n_features = X_b.shape[1]
    
    # 定义损失函数和梯度函数
    def mse_loss(theta):
        y_pred = X_b.dot(theta)
        return np.mean((y_pred - y) ** 2)
    
    def mse_gradient(theta):
        y_pred = X_b.dot(theta)
        return 2 * X_b.T.dot(y_pred - y) / len(y)
    
    # 初始化参数
    theta0 = np.random.randn(n_features)
    
    # 选择优化器
    if optimizer == 'gd':
        theta, loss_history, iter_count = gradient_descent(mse_loss, mse_gradient, theta0, learning_rate, max_iter, tol)
    elif optimizer == 'momentum':
        theta, loss_history, iter_count = momentum_optimizer(mse_loss, mse_gradient, theta0, learning_rate, max_iter=max_iter, tol=tol)
    elif optimizer == 'adam':
        theta, loss_history, iter_count = adam_optimizer(mse_loss, mse_gradient, theta0, learning_rate, max_iter=max_iter, tol=tol)
    else:
        raise ValueError("Unsupported optimizer. Choose from 'gd', 'momentum', 'adam'")
    
    return theta, loss_history, iter_count

# 6. 神经网络中反向传播的简单实现
class SimpleNeuralNetwork:
    """简单的神经网络实现"""
    def __init__(self, input_size, hidden_size, output_size):
        # 初始化权重
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def sigmoid(self, z):
        """sigmoid激活函数"""
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        """sigmoid激活函数的导数"""
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    
    def forward(self, X):
        """前向传播"""
        self.z1 = X.dot(self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = self.a1.dot(self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def compute_loss(self, y):
        """计算损失"""
        m = y.shape[0]
        loss = -np.mean(y * np.log(self.a2) + (1 - y) * np.log(1 - self.a2))
        return loss
    
    def backward(self, X, y):
        """反向传播"""
        m = y.shape[0]
        
        # 计算输出层的梯度
        self.dz2 = self.a2 - y
        self.dW2 = (1 / m) * self.a1.T.dot(self.dz2)
        self.db2 = (1 / m) * np.sum(self.dz2, axis=0, keepdims=True)
        
        # 计算隐藏层的梯度
        self.dz1 = self.dz2.dot(self.W2.T) * self.sigmoid_derivative(self.z1)
        self.dW1 = (1 / m) * X.T.dot(self.dz1)
        self.db1 = (1 / m) * np.sum(self.dz1, axis=0, keepdims=True)
    
    def update_parameters(self, learning_rate):
        """更新参数"""
        self.W1 -= learning_rate * self.dW1
        self.b1 -= learning_rate * self.db1
        self.W2 -= learning_rate * self.dW2
        self.b2 -= learning_rate * self.db2
    
    def train(self, X, y, learning_rate=0.1, epochs=1000, tol=1e-6):
        """训练网络"""
        loss_history = []
        for i in range(epochs):
            # 前向传播
            self.forward(X)
            # 计算损失
            loss = self.compute_loss(y)
            loss_history.append(loss)
            # 反向传播
            self.backward(X, y)
            # 更新参数
            self.update_parameters(learning_rate)
            # 检查收敛
            if i > 0 and abs(loss_history[-1] - loss_history[-2]) < tol:
                break
        return loss_history
    
    def predict(self, X):
        """预测"""
        return self.forward(X) >= 0.5

# 7. 学习率对梯度下降的影响分析
def analyze_learning_rate():
    """分析不同学习率对梯度下降的影响"""
    # 定义目标函数
    def f(x):
        return x[0] ** 2 + 2 * x[1] ** 2
    
    def grad_f(x):
        return np.array([2 * x[0], 4 * x[1]])
    
    # 初始化参数
    x0 = np.array([1, 1])
    
    # 不同学习率
    learning_rates = [0.01, 0.1, 0.5, 0.9]
    
    plt.figure(figsize=(12, 8))
    
    for lr in learning_rates:
        x_opt, loss_history, iter_count = gradient_descent(f, grad_f, x0, learning_rate=lr)
        plt.plot(range(len(loss_history)), loss_history, label=f'学习率 = {lr}')
    
    plt.xlabel('迭代次数')
    plt.ylabel('损失值')
    plt.title('不同学习率对梯度下降的影响')
    plt.legend()
    plt.grid(True)
    plt.show()

# 8. 不同优化器的比较
def compare_optimizers():
    """比较不同优化器的性能"""
    # 定义目标函数
    def f(x):
        return x[0] ** 2 + 2 * x[1] ** 2
    
    def grad_f(x):
        return np.array([2 * x[0], 4 * x[1]])
    
    # 初始化参数
    x0 = np.array([1, 1])
    
    # 不同优化器
    optimizers = {
        'GD': gradient_descent,
        'Momentum': momentum_optimizer,
        'Adam': adam_optimizer
    }
    
    plt.figure(figsize=(12, 8))
    
    for name, optimizer in optimizers.items():
        if name == 'GD':
            x_opt, loss_history, iter_count = optimizer(f, grad_f, x0, learning_rate=0.1)
        elif name == 'Momentum':
            x_opt, loss_history, iter_count = optimizer(f, grad_f, x0, learning_rate=0.1)
        else:  # Adam
            x_opt, loss_history, iter_count = optimizer(f, grad_f, x0, learning_rate=0.01)
        
        plt.plot(range(len(loss_history)), loss_history, label=f'{name} (迭代次数: {iter_count})')
    
    plt.xlabel('迭代次数')
    plt.ylabel('损失值')
    plt.title('不同优化器的性能比较')
    plt.legend()
    plt.grid(True)
    plt.show()

# 运行示例
if __name__ == "__main__":
    print("=== 数值微分示例 ===")
    # 测试数值微分
    def f(x):
        return x ** 2
    
    derivative = numerical_derivative(f, 2)
    print(f"f'(2) = {derivative}")
    
    # 测试数值梯度
    def f2(x):
        return x[0] ** 2 + x[1] ** 2
    
    x = np.array([1, 2])
    gradient = numerical_gradient(f2, x)
    print(f"∇f(1, 2) = {gradient}")
    
    print("\n=== 线性回归示例 ===")
    # 生成模拟数据
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y = 2 * x + 1 + np.random.normal(0, 1, 100)
    X = x.reshape(-1, 1)
    
    # 使用不同优化器进行线性回归
    for optimizer in ['gd', 'momentum', 'adam']:
        theta, loss_history, iter_count = linear_regression(X, y, optimizer=optimizer)
        print(f"\n{optimizer.upper()}优化器:")
        print(f"最终参数: θ0 = {theta[0]:.4f}, θ1 = {theta[1]:.4f}")
        print(f"最小损失值: {loss_history[-1]:.4f}")
        print(f"迭代次数: {iter_count}")
    
    print("\n=== 神经网络示例 ===")
    # 生成二分类数据
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int).reshape(-1, 1)
    
    # 创建并训练神经网络
    model = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1)
    loss_history = model.train(X, y, learning_rate=0.1, epochs=1000)
    
    # 评估模型
    predictions = model.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"神经网络准确率: {accuracy:.4f}")
    print(f"训练迭代次数: {len(loss_history)}")
    
    # 分析学习率
    print("\n=== 学习率分析 ===")
    analyze_learning_rate()
    
    # 比较优化器
    print("\n=== 优化器比较 ===")
    compare_optimizers()
