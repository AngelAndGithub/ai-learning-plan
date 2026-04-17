import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import json
import time
from flask import Flask, request, jsonify
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import docker
import os

# 1. 模型训练和保存
def train_and_save_model():
    """训练并保存模型"""
    # 加载数据集
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # 数据预处理
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 拆分数据集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # 构建模型
    model = models.Sequential([
        layers.Dense(10, activation='relu', input_shape=(4,)),
        layers.Dense(10, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])
    
    # 编译模型
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # 训练模型
    model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2)
    
    # 评估模型
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"模型准确率: {accuracy:.4f}")
    
    # 保存模型
    model.save('iris_classifier.h5')
    
    # 保存scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("模型和scaler保存完成")
    return model, scaler

# 2. 模型优化 - 量化
def optimize_model():
    """模型量化优化"""
    # 加载模型
    model = tf.keras.models.load_model('iris_classifier.h5')
    
    # 转换为TFLite模型
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # 启用量化
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # 转换模型
    tflite_model = converter.convert()
    
    # 保存量化后的模型
    with open('iris_classifier_quantized.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print("模型量化完成")
    return tflite_model

# 3. Flask API开发
def flask_api():
    """Flask API开发"""
    app = Flask(__name__)
    
    # 加载模型和scaler
    model = tf.keras.models.load_model('iris_classifier.h5')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # 类别映射
    class_names = ['setosa', 'versicolor', 'virginica']
    
    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            # 获取请求数据
            data = request.json
            features = data['features']
            
            # 数据预处理
            features_scaled = scaler.transform([features])
            
            # 预测
            predictions = model.predict(features_scaled)
            class_idx = np.argmax(predictions[0])
            class_name = class_names[class_idx]
            confidence = float(predictions[0][class_idx])
            
            # 返回结果
            return jsonify({
                'class': class_name,
                'confidence': confidence
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    
    print("Flask API创建完成")
    return app

# 4. FastAPI开发
def fastapi_api():
    """FastAPI开发"""
    app = FastAPI()
    
    # 加载模型和scaler
    model = tf.keras.models.load_model('iris_classifier.h5')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # 类别映射
    class_names = ['setosa', 'versicolor', 'virginica']
    
    # 请求模型
    class PredictionRequest(BaseModel):
        features: list
    
    @app.post('/predict')
    def predict(request: PredictionRequest):
        try:
            # 数据预处理
            features_scaled = scaler.transform([request.features])
            
            # 预测
            predictions = model.predict(features_scaled)
            class_idx = np.argmax(predictions[0])
            class_name = class_names[class_idx]
            confidence = float(predictions[0][class_idx])
            
            # 返回结果
            return {
                'class': class_name,
                'confidence': confidence
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    print("FastAPI创建完成")
    return app

# 5. Dockerfile生成
def generate_dockerfile():
    """生成Dockerfile"""
    dockerfile_content = """
FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
"""
    
    # 生成requirements.txt
    requirements_content = """
Flask
FastAPI
uvicorn
tensorflow
scikit-learn
numpy
pandas
"""
    
    # 生成app.py
    app_py_content = """
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import pickle
import numpy as np

app = FastAPI()

# 加载模型和scaler
model = tf.keras.models.load_model('iris_classifier.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# 类别映射
class_names = ['setosa', 'versicolor', 'virginica']

# 请求模型
class PredictionRequest(BaseModel):
    features: list

@app.post('/predict')
def predict(request: PredictionRequest):
    try:
        # 数据预处理
        features_scaled = scaler.transform([request.features])
        
        # 预测
        predictions = model.predict(features_scaled)
        class_idx = np.argmax(predictions[0])
        class_name = class_names[class_idx]
        confidence = float(predictions[0][class_idx])
        
        # 返回结果
        return {
            'class': class_name,
            'confidence': confidence
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
"""
    
    # 写入文件
    with open('Dockerfile', 'w') as f:
        f.write(dockerfile_content)
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements_content)
    
    with open('app.py', 'w') as f:
        f.write(app_py_content)
    
    print("Dockerfile和相关文件生成完成")
    return "Dockerfile生成完成"

# 6. Docker Compose配置
def generate_docker_compose():
    """生成Docker Compose配置"""
    docker_compose_content = """
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    restart: always
"""
    
    # 写入文件
    with open('docker-compose.yml', 'w') as f:
        f.write(docker_compose_content)
    
    print("Docker Compose配置生成完成")
    return "Docker Compose配置生成完成"

# 7. 模型监控
def model_monitoring():
    """模型监控"""
    # 模拟监控数据
    monitoring_data = []
    
    for i in range(10):
        # 模拟预测
        features = np.random.rand(4)
        prediction = np.random.rand(3)
        class_idx = np.argmax(prediction)
        
        # 模拟真实标签（这里使用随机值）
        true_label = np.random.randint(0, 3)
        
        # 计算准确率
        correct = 1 if class_idx == true_label else 0
        
        # 记录监控数据
        monitoring_data.append({
            'timestamp': time.time(),
            'features': features.tolist(),
            'prediction': prediction.tolist(),
            'predicted_class': class_idx,
            'true_label': true_label,
            'correct': correct
        })
    
    # 计算整体准确率
    accuracy = sum(item['correct'] for item in monitoring_data) / len(monitoring_data)
    print(f"模型监控准确率: {accuracy:.4f}")
    
    # 保存监控数据
    with open('monitoring_data.json', 'w') as f:
        json.dump(monitoring_data, f)
    
    print("模型监控数据保存完成")
    return monitoring_data

# 8. CI/CD配置
def generate_cicd_config():
    """生成CI/CD配置"""
    # GitHub Actions配置
    github_actions_content = """
name: CI/CD Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
    
    - name: Run tests
      run: |
        # 这里可以添加测试命令
        echo "Running tests..."
    
    - name: Build Docker image
      run: |
        docker build -t iris-classifier .
    
    - name: Push to Docker Hub
      run: |
        echo "${{ secrets.DOCKER_PASSWORD }}" | docker login -u "${{ secrets.DOCKER_USERNAME }}" --password-stdin
        docker tag iris-classifier ${{ secrets.DOCKER_USERNAME }}/iris-classifier:latest
        docker push ${{ secrets.DOCKER_USERNAME }}/iris-classifier:latest
"""
    
    # 写入文件
    os.makedirs('.github/workflows', exist_ok=True)
    with open('.github/workflows/ci-cd.yml', 'w') as f:
        f.write(github_actions_content)
    
    print("CI/CD配置生成完成")
    return "CI/CD配置生成完成"

# 9. 边缘部署准备
def edge_deployment_preparation():
    """边缘部署准备"""
    # 加载模型
    model = tf.keras.models.load_model('iris_classifier.h5')
    
    # 转换为TFLite模型
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    # 保存TFLite模型
    with open('iris_classifier.tflite', 'wb') as f:
        f.write(tflite_model)
    
    # 生成边缘设备推理代码
    edge_inference_code = """
import numpy as np
import tensorflow as tf
import pickle

# 加载TFLite模型
interpreter = tf.lite.Interpreter(model_path='iris_classifier.tflite')
interpreter.allocate_tensors()

# 获取输入和输出张量
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 加载scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# 类别映射
class_names = ['setosa', 'versicolor', 'virginica']

def predict(features):
    # 数据预处理
    features_scaled = scaler.transform([features])
    
    # 设置输入
    interpreter.set_tensor(input_details[0]['index'], features_scaled.astype(np.float32))
    
    # 执行推理
    interpreter.invoke()
    
    # 获取输出
    output = interpreter.get_tensor(output_details[0]['index'])
    
    # 处理结果
    class_idx = np.argmax(output[0])
    class_name = class_names[class_idx]
    confidence = float(output[0][class_idx])
    
    return class_name, confidence

# 测试
if __name__ == "__main__":
    # 测试数据
    test_features = [5.1, 3.5, 1.4, 0.2]  # setosa
    
    # 预测
    class_name, confidence = predict(test_features)
    print(f"预测结果: {class_name}, 置信度: {confidence:.4f}")
"""
    
    # 写入文件
    with open('edge_inference.py', 'w') as f:
        f.write(edge_inference_code)
    
    print("边缘部署准备完成")
    return "边缘部署准备完成"

# 主函数
if __name__ == "__main__":
    print("=== 部署与工程化示例 ===")
    
    # 1. 训练和保存模型
    print("\n1. 训练和保存模型")
    model, scaler = train_and_save_model()
    
    # 2. 模型优化 - 量化
    print("\n2. 模型优化 - 量化")
    tflite_model = optimize_model()
    
    # 3. Flask API开发
    print("\n3. Flask API开发")
    flask_app = flask_api()
    
    # 4. FastAPI开发
    print("\n4. FastAPI开发")
    fastapi_app = fastapi_api()
    
    # 5. Dockerfile生成
    print("\n5. Dockerfile生成")
    dockerfile_result = generate_dockerfile()
    
    # 6. Docker Compose配置
    print("\n6. Docker Compose配置")
    docker_compose_result = generate_docker_compose()
    
    # 7. 模型监控
    print("\n7. 模型监控")
    monitoring_data = model_monitoring()
    
    # 8. CI/CD配置
    print("\n8. CI/CD配置")
    cicd_result = generate_cicd_config()
    
    # 9. 边缘部署准备
    print("\n9. 边缘部署准备")
    edge_result = edge_deployment_preparation()
    
    print("\n=== 部署与工程化示例完成 ===")
