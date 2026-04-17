import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Input, LSTM, Embedding
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
import time

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 1. 张量操作示例
def tensor_operations():
    """张量操作示例"""
    print("=== 张量操作示例 ===")
    
    # 创建张量
    scalar = tf.constant(5.0)
    vector = tf.constant([1.0, 2.0, 3.0])
    matrix = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    tensor_3d = tf.constant([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    
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
    a = tf.constant([1.0, 2.0])
    b = tf.constant([3.0, 4.0])
    
    print(f"a + b: {tf.add(a, b)}")
    print(f"a - b: {tf.subtract(a, b)}")
    print(f"a * b: {tf.multiply(a, b)}")
    print(f"a / b: {tf.divide(a, b)}")
    
    # 矩阵乘法
    c = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    d = tf.constant([[5.0, 6.0], [7.0, 8.0]])
    print(f"矩阵乘法: {tf.matmul(c, d)}")
    
    # 形状操作
    e = tf.constant([[1, 2, 3], [4, 5, 6]])
    print(f"原始形状: {e.shape}")
    print(f"reshape: {tf.reshape(e, (3, 2))}")
    print(f"转置: {tf.transpose(e)}")
    
    # 变量
    w = tf.Variable(0.0)
    print(f"初始变量值: {w}")
    
    # 修改变量值
    w.assign(1.0)
    print(f"修改后变量值: {w}")
    
    return scalar, vector, matrix, tensor_3d

# 2. 自动微分示例
def automatic_differentiation():
    """自动微分示例"""
    print("\n=== 自动微分示例 ===")
    
    # 单个变量的梯度
    x = tf.Variable(3.0)
    with tf.GradientTape() as tape:
        y = x ** 2
    dy_dx = tape.gradient(y, x)
    print(f"dy/dx = {dy_dx}")
    
    # 多个变量的梯度
    w = tf.Variable(2.0)
    b = tf.Variable(1.0)
    x = tf.constant(5.0)
    
    with tf.GradientTape() as tape:
        y = w * x + b
    dw, db = tape.gradient(y, [w, b])
    print(f"dy/dw = {dw}")
    print(f"dy/db = {db}")
    
    # 二阶导数
    x = tf.Variable(3.0)
    with tf.GradientTape() as tape1:
        with tf.GradientTape() as tape2:
            y = x ** 3
        dy_dx = tape2.gradient(y, x)
    d2y_dx2 = tape1.gradient(dy_dx, x)
    print(f"d²y/dx² = {d2y_dx2}")
    
    return dy_dx, dw, db, d2y_dx2

# 3. 线性回归示例
def linear_regression():
    """线性回归示例"""
    print("\n=== 线性回归示例 ===")
    
    # 生成合成数据
    X = tf.random.normal(shape=(1000, 1))
    y = 2 * X + 3 + tf.random.normal(shape=(1000, 1), stddev=0.1)
    
    # 可视化数据
    plt.scatter(X.numpy(), y.numpy(), alpha=0.5)
    plt.title('线性回归数据')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.savefig('linear_regression_data.png')
    plt.close()
    
    # 定义模型
    model = Sequential([
        Dense(1, input_shape=(1,))
    ])
    
    # 编译模型
    model.compile(optimizer='sgd', loss='mse')
    
    # 训练模型
    model.fit(X, y, epochs=100, batch_size=32, verbose=0)
    
    # 评估模型
    loss = model.evaluate(X, y, verbose=0)
    print(f"训练损失: {loss:.4f}")
    
    # 预测
    y_pred = model.predict(X)
    
    # 可视化结果
    plt.scatter(X.numpy(), y.numpy(), alpha=0.5, label='真实值')
    plt.plot(X.numpy(), y_pred, color='red', label='预测值')
    plt.title('线性回归结果')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.savefig('linear_regression_result.png')
    plt.close()
    
    # 获取模型参数
    weights = model.get_weights()
    print(f"权重: {weights[0][0][0]:.4f}")
    print(f"偏置: {weights[1][0]:.4f}")
    
    return model

# 4. MNIST手写数字分类示例
def mnist_classification():
    """MNIST手写数字分类示例"""
    print("\n=== MNIST手写数字分类示例 ===")
    
    # 加载数据集
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # 数据预处理
    x_train = x_train.reshape(-1, 28*28) / 255.0
    x_test = x_test.reshape(-1, 28*28) / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    # 定义模型
    model = Sequential([
        Dense(128, activation='relu', input_shape=(784,)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(10, activation='softmax')
    ])
    
    # 编译模型
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # 训练模型
    log_dir = f"logs/mnist/{int(time.time())}"
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[tensorboard_callback])
    
    # 评估模型
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"测试损失: {loss:.4f}")
    print(f"测试准确率: {accuracy:.4f}")
    
    # 保存模型
    model.save('mnist_model.h5')
    
    return model

# 5. CIFAR-10图像分类示例
def cifar10_classification():
    """CIFAR-10图像分类示例"""
    print("\n=== CIFAR-10图像分类示例 ===")
    
    # 加载数据集
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # 数据预处理
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    # 定义模型
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    
    # 编译模型
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # 训练模型
    log_dir = f"logs/cifar10/{int(time.time())}"
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[tensorboard_callback])
    
    # 评估模型
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"测试损失: {loss:.4f}")
    print(f"测试准确率: {accuracy:.4f}")
    
    # 保存模型
    model.save('cifar10_model.h5')
    
    return model

# 6. 函数式API示例
def functional_api_example():
    """函数式API示例"""
    print("\n=== 函数式API示例 ===")
    
    # 定义输入
    inputs = Input(shape=(784,))
    
    # 定义隐藏层
    x = Dense(128, activation='relu')(inputs)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # 定义输出层
    outputs = Dense(10, activation='softmax')(x)
    
    # 创建模型
    model = Model(inputs=inputs, outputs=outputs)
    
    # 编译模型
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # 打印模型结构
    model.summary()
    
    return model

# 7. 模型保存与加载示例
def model_save_load():
    """模型保存与加载示例"""
    print("\n=== 模型保存与加载示例 ===")
    
    # 加载MNIST数据集
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28*28) / 255.0
    y_train = to_categorical(y_train, 10)
    
    # 创建并训练模型
    model = Sequential([
        Dense(128, activation='relu', input_shape=(784,)),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=2, batch_size=32, verbose=0)
    
    # 保存模型
    model.save('model.h5')
    print("模型已保存为 model.h5")
    
    # 加载模型
    from tensorflow.keras.models import load_model
    loaded_model = load_model('model.h5')
    print("模型已加载")
    
    # 评估加载的模型
    x_test = x_test.reshape(-1, 28*28) / 255.0
    y_test = to_categorical(y_test, 10)
    loss, accuracy = loaded_model.evaluate(x_test, y_test, verbose=0)
    print(f"加载模型的测试准确率: {accuracy:.4f}")
    
    return model, loaded_model

# 8. 数据集API示例
def dataset_api_example():
    """数据集API示例"""
    print("\n=== 数据集API示例 ===")
    
    # 创建数据集
    X = tf.random.normal(shape=(1000, 10))
    y = tf.random.normal(shape=(1000, 1))
    
    # 从张量创建数据集
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    # 数据预处理
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(32)
    dataset = dataset.repeat()
    
    # 遍历数据集
    for batch in dataset.take(3):
        print(f"批次形状: X={batch[0].shape}, y={batch[1].shape}")
    
    return dataset

# 9. 自定义层示例
class CustomDense(tf.keras.layers.Layer):
    """自定义全连接层"""
    def __init__(self, units=32):
        super(CustomDense, self).__init__()
        self.units = units
    
    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                initializer='random_normal',
                                trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                initializer='zeros',
                                trainable=True)
    
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

def custom_layer_example():
    """自定义层示例"""
    print("\n=== 自定义层示例 ===")
    
    # 创建模型
    model = Sequential([
        CustomDense(64),
        tf.keras.layers.Activation('relu'),
        CustomDense(10),
        tf.keras.layers.Activation('softmax')
    ])
    
    # 编译模型
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # 打印模型结构
    model.summary()
    
    return model

# 10. 部署模型示例
def model_deployment():
    """模型部署示例"""
    print("\n=== 模型部署示例 ===")
    
    # 加载MNIST数据集
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28*28) / 255.0
    y_train = to_categorical(y_train, 10)
    
    # 创建并训练模型
    model = Sequential([
        Dense(128, activation='relu', input_shape=(784,)),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=2, batch_size=32, verbose=0)
    
    # 导出为SavedModel
    tf.saved_model.save(model, 'saved_model')
    print("模型已导出为SavedModel格式")
    
    # 加载SavedModel
    loaded_model = tf.saved_model.load('saved_model')
    print("SavedModel已加载")
    
    # 测试加载的模型
    x_test = x_test.reshape(-1, 28*28) / 255.0
    y_test = to_categorical(y_test, 10)
    
    # 使用加载的模型进行预测
    predictions = loaded_model(x_test[:5])
    print(f"预测结果: {tf.argmax(predictions, axis=1).numpy()}")
    print(f"真实标签: {tf.argmax(y_test[:5], axis=1).numpy()}")
    
    return model, loaded_model

if __name__ == "__main__":
    # 运行所有示例
    tensor_operations()
    automatic_differentiation()
    linear_regression()
    mnist_classification()
    cifar10_classification()
    functional_api_example()
    model_save_load()
    dataset_api_example()
    custom_layer_example()
    model_deployment()
    
    print("\n所有示例运行完成！")