import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist, cifar10

# 1. 图像处理基础
def image_processing_basics():
    """图像处理基础操作"""
    # 读取图像
    # 注意：由于是示例代码，这里假设当前目录有测试图像
    # 实际使用时需要替换为真实的图像路径
    try:
        img = cv2.imread('test_image.jpg')
        if img is None:
            raise FileNotFoundError("测试图像未找到，请替换为真实的图像路径")
    except:
        # 创建一个测试图像
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (150, 150), (0, 255, 0), -1)
        cv2.circle(img, (100, 100), 30, (0, 0, 255), -1)
    
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 图像缩放
    resized = cv2.resize(img, (100, 100))
    
    # 图像旋转
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1)
    rotated = cv2.warpAffine(img, M, (cols, rows))
    
    # 边缘检测
    edges = cv2.Canny(gray, 100, 200)
    
    # 高斯模糊
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    print("图像处理基础操作完成")
    return img, gray, resized, rotated, edges, blurred

# 2. 图像增强
def image_enhancement():
    """图像增强技术"""
    try:
        img = cv2.imread('test_image.jpg')
        if img is None:
            raise FileNotFoundError("测试图像未找到，请替换为真实的图像路径")
    except:
        # 创建一个测试图像
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (150, 150), (100, 100, 100), -1)
    
    # 直方图均衡化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    
    # 伽马校正
    gamma = 1.5
    gamma_corrected = np.power(img / 255.0, gamma) * 255.0
    gamma_corrected = gamma_corrected.astype(np.uint8)
    
    # 对比度调整
    alpha = 1.5  # 对比度增益
    beta = 0     # 亮度增益
    contrast = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    
    print("图像增强操作完成")
    return equalized, gamma_corrected, contrast

# 3. 卷积神经网络基础
def cnn_basics():
    """卷积神经网络基础"""
    # 创建一个简单的CNN模型
    model = models.Sequential([
        # 第一层卷积
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        # 第二层卷积
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        # 第三层卷积
        layers.Conv2D(64, (3, 3), activation='relu'),
        # 展平
        layers.Flatten(),
        # 全连接层
        layers.Dense(64, activation='relu'),
        # 输出层
        layers.Dense(10, activation='softmax')
    ])
    
    model.summary()
    print("CNN基础模型创建完成")
    return model

# 4. 训练MNIST分类模型
def train_mnist_model():
    """训练MNIST手写数字分类模型"""
    # 加载MNIST数据集
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    
    # 数据预处理
    train_images = train_images.reshape((60000, 28, 28, 1))
    train_images = train_images.astype('float32') / 255
    
    test_images = test_images.reshape((10000, 28, 28, 1))
    test_images = test_images.astype('float32') / 255
    
    # 创建模型
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    # 编译模型
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # 训练模型
    print("开始训练MNIST模型...")
    history = model.fit(train_images, train_labels, epochs=5, 
                        validation_data=(test_images, test_labels))
    
    # 评估模型
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"测试准确率: {test_acc}")
    
    print("MNIST模型训练完成")
    return model, history

# 5. 训练CIFAR-10分类模型
def train_cifar10_model():
    """训练CIFAR-10图像分类模型"""
    # 加载CIFAR-10数据集
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    
    # 数据预处理
    train_images = train_images.astype('float32') / 255
    test_images = test_images.astype('float32') / 255
    
    # 创建模型
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    # 编译模型
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # 训练模型
    print("开始训练CIFAR-10模型...")
    history = model.fit(train_images, train_labels, epochs=10, 
                        validation_data=(test_images, test_labels))
    
    # 评估模型
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"测试准确率: {test_acc}")
    
    print("CIFAR-10模型训练完成")
    return model, history

# 6. 目标检测基础 - 使用OpenCV的Haar级联分类器
def object_detection_basics():
    """目标检测基础"""
    # 加载预训练的人脸检测模型
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    try:
        img = cv2.imread('test_face.jpg')
        if img is None:
            raise FileNotFoundError("测试图像未找到，请替换为真实的图像路径")
    except:
        # 创建一个测试图像
        img = np.ones((400, 400, 3), dtype=np.uint8) * 255
        cv2.circle(img, (200, 200), 50, (0, 0, 0), -1)
    
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 检测人脸
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # 绘制检测结果
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    print(f"检测到 {len(faces)} 个人脸")
    print("目标检测基础操作完成")
    return img, faces

# 7. 图像分割基础
def image_segmentation_basics():
    """图像分割基础"""
    try:
        img = cv2.imread('test_image.jpg')
        if img is None:
            raise FileNotFoundError("测试图像未找到，请替换为真实的图像路径")
    except:
        # 创建一个测试图像
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (150, 150), (255, 255, 255), -1)
    
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 阈值分割
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # 轮廓检测
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 绘制轮廓
    contour_img = img.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    
    print(f"检测到 {len(contours)} 个轮廓")
    print("图像分割基础操作完成")
    return thresh, contour_img, contours

# 主函数
if __name__ == "__main__":
    print("=== 计算机视觉基础示例 ===")
    
    # 1. 图像处理基础
    print("\n1. 图像处理基础")
    try:
        img, gray, resized, rotated, edges, blurred = image_processing_basics()
        print("图像处理操作成功")
    except Exception as e:
        print(f"图像处理操作失败: {e}")
    
    # 2. 图像增强
    print("\n2. 图像增强")
    try:
        equalized, gamma_corrected, contrast = image_enhancement()
        print("图像增强操作成功")
    except Exception as e:
        print(f"图像增强操作失败: {e}")
    
    # 3. CNN基础
    print("\n3. CNN基础")
    try:
        model = cnn_basics()
        print("CNN模型创建成功")
    except Exception as e:
        print(f"CNN模型创建失败: {e}")
    
    # 4. 训练MNIST模型
    print("\n4. 训练MNIST模型")
    try:
        mnist_model, mnist_history = train_mnist_model()
        print("MNIST模型训练成功")
    except Exception as e:
        print(f"MNIST模型训练失败: {e}")
    
    # 5. 训练CIFAR-10模型
    print("\n5. 训练CIFAR-10模型")
    try:
        cifar_model, cifar_history = train_cifar10_model()
        print("CIFAR-10模型训练成功")
    except Exception as e:
        print(f"CIFAR-10模型训练失败: {e}")
    
    # 6. 目标检测基础
    print("\n6. 目标检测基础")
    try:
        face_img, faces = object_detection_basics()
        print("目标检测操作成功")
    except Exception as e:
        print(f"目标检测操作失败: {e}")
    
    # 7. 图像分割基础
    print("\n7. 图像分割基础")
    try:
        thresh, contour_img, contours = image_segmentation_basics()
        print("图像分割操作成功")
    except Exception as e:
        print(f"图像分割操作失败: {e}")
    
    print("\n=== 计算机视觉基础示例完成 ===")
