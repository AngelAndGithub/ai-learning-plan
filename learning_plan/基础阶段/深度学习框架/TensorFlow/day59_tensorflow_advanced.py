#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第59天：TensorFlow高级特性
深度学习框架学习示例
内容：自定义层、自定义损失函数、模型部署等高级特性
"""

print("=== 第59天：TensorFlow高级特性 ===")

# 1. 自定义层
print("\n1. 自定义层")

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np

# 自定义线性层
class LinearLayer(Layer):
    """自定义线性层"""
    
    def __init__(self, units=32):
        super(LinearLayer, self).__init__()
        self.units = units
    
    def build(self, input_shape):
        """构建层的参数"""
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
    
    def call(self, inputs):
        """前向传播"""
        return tf.matmul(inputs, self.w) + self.b

# 测试自定义层
print("测试自定义线性层:")
linear_layer = LinearLayer(units=10)
input_tensor = tf.random.normal(shape=(32, 784))
output_tensor = linear_layer(input_tensor)
print(f"输入形状: {input_tensor.shape}")
print(f"输出形状: {output_tensor.shape}")

# 2. 自定义激活函数
print("\n2. 自定义激活函数")

# 自定义激活函数
def custom_activation(x):
    """自定义激活函数"""
    return tf.nn.relu(x) + tf.sin(x)

# 测试自定义激活函数
print("测试自定义激活函数:")
x = tf.constant([-1.0, 0.0, 1.0, 2.0])
y = custom_activation(x)
print(f"输入: {x.numpy()}")
print(f"输出: {y.numpy()}")

# 3. 自定义损失函数
print("\n3. 自定义损失函数")

# 自定义损失函数
def custom_loss(y_true, y_pred):
    """自定义损失函数"""
    return tf.reduce_mean(tf.square(y_true - y_pred)) + tf.reduce_mean(tf.abs(y_true - y_pred))

# 测试自定义损失函数
print("测试自定义损失函数:")
y_true = tf.constant([1.0, 2.0, 3.0])
y_pred = tf.constant([1.2, 1.8, 3.1])
loss = custom_loss(y_true, y_pred)
print(f"真实值: {y_true.numpy()}")
print(f"预测值: {y_pred.numpy()}")
print(f"损失值: {loss.numpy()}")

# 4. 自定义评估指标
print("\n4. 自定义评估指标")

from tensorflow.keras.metrics import Metric

class CustomAccuracy(Metric):
    """自定义准确率指标"""
    
    def __init__(self, name='custom_accuracy', **kwargs):
        super(CustomAccuracy, self).__init__(name=name, **kwargs)
        self.correct = self.add_weight(name='correct', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        """更新状态"""
        y_true = tf.argmax(y_true, axis=1)
        y_pred = tf.argmax(y_pred, axis=1)
        correct = tf.cast(tf.equal(y_true, y_pred), tf.float32)
        
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            correct = tf.multiply(correct, sample_weight)
        
        self.correct.assign_add(tf.reduce_sum(correct))
        self.total.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))
    
    def result(self):
        """计算结果"""
        return self.correct / self.total

# 5. 模型保存和加载
print("\n5. 模型保存和加载")

# 加载MNIST数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 数据预处理
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train_flat = x_train.reshape(-1, 784)
x_test_flat = x_test.reshape(-1, 784)
y_train_one_hot = to_categorical(y_train, 10)
y_test_one_hot = to_categorical(y_test, 10)

# 创建模型
model = Sequential([
    LinearLayer(units=64),
    tf.keras.layers.Activation(custom_activation),
    LinearLayer(units=32),
    tf.keras.layers.Activation(custom_activation),
    LinearLayer(units=10),
    tf.keras.layers.Activation('softmax')
])

# 编译模型
model.compile(
    optimizer='adam',
    loss=custom_loss,
    metrics=[CustomAccuracy()]
)

# 训练模型
model.fit(
    x_train_flat,
    y_train_one_hot,
    epochs=5,
    batch_size=32,
    validation_split=0.2
)

# 保存模型为SavedModel格式
model.save('custom_model')
print("模型已保存为SavedModel格式")

# 加载模型
loaded_model = tf.keras.models.load_model('custom_model', custom_objects={'LinearLayer': LinearLayer, 'custom_activation': custom_activation, 'custom_loss': custom_loss, 'CustomAccuracy': CustomAccuracy})
print("模型已加载")

# 评估模型
loss, accuracy = loaded_model.evaluate(x_test_flat, y_test_one_hot)
print(f"加载模型的测试集准确率: {accuracy:.4f}")

# 6. TensorFlow Lite
print("\n6. TensorFlow Lite")

# 转换为TensorFlow Lite模型
converter = tf.lite.TFLiteConverter.from_saved_model('custom_model')
tflite_model = converter.convert()

# 保存TensorFlow Lite模型
with open('custom_model.tflite', 'wb') as f:
    f.write(tflite_model)
print("模型已转换为TensorFlow Lite格式")

# 测试TensorFlow Lite模型
print("测试TensorFlow Lite模型:")

# 加载TFLite模型
interpreter = tf.lite.Interpreter(model_path='custom_model.tflite')
interpreter.allocate_tensors()

# 获取输入和输出张量
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 测试样本
test_sample = x_test_flat[0:1]

# 设置输入
interpreter.set_tensor(input_details[0]['index'], test_sample.astype(np.float32))

# 运行推理
interpreter.invoke()

# 获取输出
output = interpreter.get_tensor(output_details[0]['index'])
predicted_label = np.argmax(output)
true_label = np.argmax(y_test_one_hot[0])

print(f"预测标签: {predicted_label}")
print(f"真实标签: {true_label}")
print(f"预测正确: {predicted_label == true_label}")

# 7. 模型量化
print("\n7. 模型量化")

# 量化为INT8模型
converter = tf.lite.TFLiteConverter.from_saved_model('custom_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 提供校准数据
def representative_data_gen():
    for i in range(100):
        yield [x_train_flat[i:i+1].astype(np.float32)]

converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

quantized_tflite_model = converter.convert()

# 保存量化模型
with open('quantized_model.tflite', 'wb') as f:
    f.write(quantized_tflite_model)

print("模型已量化为INT8格式")

# 8. 自定义训练循环
print("\n8. 自定义训练循环")

# 重置模型权重
model = Sequential([
    LinearLayer(units=64),
    tf.keras.layers.Activation(custom_activation),
    LinearLayer(units=32),
    tf.keras.layers.Activation(custom_activation),
    LinearLayer(units=10),
    tf.keras.layers.Activation('softmax')
])

# 优化器
optimizer = tf.optimizers.Adam()

# 损失函数
loss_fn = custom_loss

# 评估指标
accuracy_metric = CustomAccuracy()

# 批次大小
batch_size = 32

# 训练循环
epochs = 5
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    
    # 打乱数据
    indices = np.random.permutation(len(x_train_flat))
    x_train_shuffled = x_train_flat[indices]
    y_train_shuffled = y_train_one_hot[indices]
    
    # 分批训练
    for i in range(0, len(x_train_flat), batch_size):
        x_batch = x_train_shuffled[i:i+batch_size]
        y_batch = y_train_shuffled[i:i+batch_size]
        
        with tf.GradientTape() as tape:
            y_pred = model(x_batch, training=True)
            loss = loss_fn(y_batch, y_pred)
        
        # 计算梯度
        gradients = tape.gradient(loss, model.trainable_variables)
        
        # 更新权重
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    # 评估
    accuracy_metric.reset_states()
    for i in range(0, len(x_test_flat), batch_size):
        x_batch = x_test_flat[i:i+batch_size]
        y_batch = y_test_one_hot[i:i+batch_size]
        y_pred = model(x_batch, training=False)
        accuracy_metric.update_state(y_batch, y_pred)
    
    accuracy = accuracy_metric.result().numpy()
    print(f"测试准确率: {accuracy:.4f}")

# 9. 分布式训练
print("\n9. 分布式训练")

print("TensorFlow支持多种分布式训练策略:")
print("- MirroredStrategy: 单机多GPU")
print("- MultiWorkerMirroredStrategy: 多机多GPU")
print("- TPUStrategy: TPU训练")
print("- ParameterServerStrategy: 参数服务器")

# 示例：使用MirroredStrategy
print("\n使用MirroredStrategy的示例:")
print("strategy = tf.distribute.MirroredStrategy()")
print("with strategy.scope():")
print("    model = create_model()")
print("    model.compile(...)")
print("model.fit(...)")

# 10. 自动混合精度训练
print("\n10. 自动混合精度训练")

print("自动混合精度训练可以提高训练速度并减少内存使用")
print("- 使用float16和float32混合精度")
print("- 需要支持FP16的GPU")

# 示例：使用自动混合精度
print("\n使用自动混合精度的示例:")
print("from tensorflow.keras.mixed_precision import experimental as mixed_precision")
print("policy = mixed_precision.Policy('mixed_float16')")
print("mixed_precision.set_global_policy(policy)")
print("# 然后正常创建和训练模型")

# 11. 练习
print("\n11. 练习")

# 练习1: 自定义更复杂的层
print("练习1: 自定义更复杂的层")
print("- 实现一个自定义卷积层")
print("- 实现一个自定义注意力层")

# 练习2: 模型部署
print("\n练习2: 模型部署")
print("- 尝试使用TensorFlow Serving部署模型")
print("- 尝试使用Docker容器部署模型")

# 练习3: 模型优化
print("\n练习3: 模型优化")
print("- 尝试不同的量化策略")
print("- 尝试模型剪枝")
print("- 尝试知识蒸馏")

print("\n=== 第59天学习示例结束 ===")
