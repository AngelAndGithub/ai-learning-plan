#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第57天：TensorFlow基础
深度学习框架学习示例
内容：TensorFlow的基本概念、张量操作和计算图
"""

print("=== 第57天：TensorFlow基础 ===")

# 1. TensorFlow基本概念
print("\n1. TensorFlow基本概念")

import tensorflow as tf
print(f"TensorFlow版本: {tf.__version__}")

print("TensorFlow是一个端到端的机器学习平台")
print("- 张量 (Tensor): 多维数组")
print("- 计算图 (Computation Graph): 表示计算的有向图")
print("- 会话 (Session): 执行计算图的环境")
print("- 变量 (Variable): 可训练的参数")
print("- 占位符 (Placeholder): 用于输入数据")

# 2. 张量的创建
print("\n2. 张量的创建")

# 创建常量张量
const_tensor = tf.constant(42)
print(f"常量张量: {const_tensor}")
print(f"张量值: {const_tensor.numpy()}")

# 创建多维张量
multi_dimensional = tf.constant([[1, 2, 3], [4, 5, 6]])
print(f"\n二维张量:")
print(multi_dimensional)
print(f"张量形状: {multi_dimensional.shape}")
print(f"张量值:")
print(multi_dimensional.numpy())

# 创建全零张量
zero_tensor = tf.zeros([2, 3])
print(f"\n全零张量:")
print(zero_tensor)

# 创建全一张量
ones_tensor = tf.ones([3, 2])
print(f"\n全一张量:")
print(ones_tensor)

# 创建随机张量
random_tensor = tf.random.normal([2, 3], mean=0, stddev=1)
print(f"\n随机张量:")
print(random_tensor)

# 3. 张量操作
print("\n3. 张量操作")

# 基本算术操作
a = tf.constant(5)
b = tf.constant(3)
print(f"a = {a.numpy()}, b = {b.numpy()}")
print(f"a + b = {(a + b).numpy()}")
print(f"a - b = {(a - b).numpy()}")
print(f"a * b = {(a * b).numpy()}")
print(f"a / b = {(a / b).numpy()}")
print(f"a // b = {(a // b).numpy()}")
print(f"a % b = {(a % b).numpy()}")
print(f"a ** b = {(a ** b).numpy()}")

# 张量运算
x = tf.constant([[1, 2], [3, 4]])
y = tf.constant([[5, 6], [7, 8]])
print(f"\nx =\n{x.numpy()}")
print(f"y =\n{y.numpy()}")
print(f"x + y =\n{(x + y).numpy()}")
print(f"x * y =\n{(x * y).numpy()}")  # 元素级乘法
print(f"矩阵乘法 x @ y =\n{(x @ y).numpy()}")
print(f"x的转置 =\n{tf.transpose(x).numpy()}")
print(f"x的和 =\n{tf.reduce_sum(x).numpy()}")
print(f"x的均值 =\n{tf.reduce_mean(x).numpy()}")
print(f"x的最大值 =\n{tf.reduce_max(x).numpy()}")
print(f"x的最小值 =\n{tf.reduce_min(x).numpy()}")

# 4. 自动微分
print("\n4. 自动微分")

# 定义变量
x = tf.Variable(3.0)

# 记录计算过程
with tf.GradientTape() as tape:
    y = x ** 2

# 计算梯度
gradient = tape.gradient(y, x)
print(f"y = x², x = {x.numpy()}")
print(f"dy/dx = {gradient.numpy()}")

# 多变量梯度
w = tf.Variable([1.0, 2.0])
b = tf.Variable(3.0)
x = tf.constant([4.0, 5.0])

with tf.GradientTape() as tape:
    y = tf.tensordot(w, x, axes=1) + b

[dw, db] = tape.gradient(y, [w, b])
print(f"\nw = {w.numpy()}, b = {b.numpy()}, x = {x.numpy()}")
print(f"y = w·x + b = {y.numpy()}")
print(f"dw = {dw.numpy()}")
print(f"db = {db.numpy()}")

# 5. 变量和模型参数
print("\n5. 变量和模型参数")

# 创建变量
weights = tf.Variable(tf.random.normal([2, 3]))
biases = tf.Variable(tf.zeros([3]))

print(f"权重变量形状: {weights.shape}")
print(f"偏置变量形状: {biases.shape}")
print(f"权重值:")
print(weights.numpy())
print(f"偏置值:")
print(biases.numpy())

# 更新变量
print("\n更新变量:")
new_weights = tf.random.normal([2, 3])
weights.assign(new_weights)
print(f"更新后的权重:")
print(weights.numpy())

# 增量更新
print("\n增量更新:")
weights.assign_add(tf.ones_like(weights))
print(f"增量更新后的权重:")
print(weights.numpy())

# 6. 计算图和函数
print("\n6. 计算图和函数")

# 使用tf.function装饰器创建计算图
@tf.function
def linear_model(x, w, b):
    return tf.tensordot(x, w, axes=1) + b

# 测试函数
x = tf.constant([1.0, 2.0])
w = tf.Variable([3.0, 4.0])
b = tf.Variable(5.0)

result = linear_model(x, w, b)
print(f"线性模型结果: {result.numpy()}")

# 7. 数据加载和处理
print("\n7. 数据加载和处理")

# 创建数据集
dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5])
print("原始数据集:")
for element in dataset:
    print(element.numpy())

# 数据集操作
dataset = dataset.map(lambda x: x * 2)  # 每个元素乘以2
dataset = dataset.batch(2)  # 批量处理

print("\n处理后的数据集:")
for batch in dataset:
    print(batch.numpy())

# 8. 线性回归示例
print("\n8. 线性回归示例")

# 生成模拟数据
np.random.seed(42)
x = np.random.randn(100, 1)
y = 2 * x + 1 + np.random.randn(100, 1) * 0.1

# 转换为张量
x_tensor = tf.constant(x, dtype=tf.float32)
y_tensor = tf.constant(y, dtype=tf.float32)

# 定义模型参数
w = tf.Variable(tf.random.normal([1, 1]))
b = tf.Variable(tf.zeros([1]))

# 定义损失函数
def loss_function(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# 定义优化器
optimizer = tf.optimizers.SGD(learning_rate=0.01)

# 训练模型
epochs = 1000
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        y_pred = tf.matmul(x_tensor, w) + b
        loss = loss_function(y_tensor, y_pred)
    
    # 计算梯度
    gradients = tape.gradient(loss, [w, b])
    
    # 更新参数
    optimizer.apply_gradients(zip(gradients, [w, b]))
    
    # 每100个epoch打印一次
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.numpy():.4f}")

print(f"\n训练完成!")
print(f"权重: {w.numpy()[0][0]:.4f}")
print(f"偏置: {b.numpy()[0]:.4f}")
print(f"预期权重: 2.0, 预期偏置: 1.0")

# 9. 保存和加载模型
print("\n9. 保存和加载模型")

# 保存模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))
])

model.compile(optimizer='sgd', loss='mse')
model.fit(x, y, epochs=100, verbose=0)

# 保存模型
model.save('linear_model')
print("模型已保存到 'linear_model' 目录")

# 加载模型
loaded_model = tf.keras.models.load_model('linear_model')
print("模型已加载")

# 测试加载的模型
test_x = np.array([[2.0]])
prediction = loaded_model.predict(test_x)
print(f"测试输入: {test_x[0][0]}")
print(f"预测输出: {prediction[0][0]:.4f}")
print(f"预期输出: {2*test_x[0][0] + 1}")

# 10. 练习
print("\n10. 练习")

# 练习1: 创建不同类型的张量
print("练习1: 创建不同类型的张量")

# 创建一个3x3的单位矩阵
identity_matrix = tf.eye(3)
print(f"3x3单位矩阵:")
print(identity_matrix.numpy())

# 创建一个形状为(2, 3, 4)的随机张量
random_3d = tf.random.normal([2, 3, 4])
print(f"\n2x3x4随机张量形状: {random_3d.shape}")
print(f"张量值:")
print(random_3d.numpy())

# 练习2: 张量运算
print("\n练习2: 张量运算")

# 创建两个张量
A = tf.constant([[1, 2], [3, 4]])
B = tf.constant([[5, 6], [7, 8]])

print(f"A =\n{A.numpy()}")
print(f"B =\n{B.numpy()}")
print(f"A + B =\n{(A + B).numpy()}")
print(f"A - B =\n{(A - B).numpy()}")
print(f"A * B =\n{(A * B).numpy()}")
print(f"A @ B =\n{(A @ B).numpy()}")
print(f"A的逆矩阵 =\n{tf.linalg.inv(A).numpy()}")
print(f"A的行列式 =\n{tf.linalg.det(A).numpy()}")

# 练习3: 自动微分
print("\n练习3: 自动微分")

# 定义一个复杂函数
x = tf.Variable(2.0)

with tf.GradientTape() as tape:
    y = x ** 3 + 2 * x ** 2 + 3 * x + 4

gradient = tape.gradient(y, x)
print(f"y = x³ + 2x² + 3x + 4, x = {x.numpy()}")
print(f"dy/dx = {gradient.numpy()}")
print(f"解析解: 3x² + 4x + 3 = {3*(2**2) + 4*2 + 3}")

print("\n=== 第57天学习示例结束 ===")
