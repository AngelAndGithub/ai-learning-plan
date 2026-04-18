#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第49天：模型部署基础
机器学习基础学习示例
内容：模型部署的基本概念、模型保存与加载、API开发、Docker容器化
"""

print("=== 第49天：模型部署基础 ===")

import numpy as np
import pickle
import joblib
import json
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify
import os
import time

# 1. 模型部署概述
print("\n1. 模型部署概述")

print("模型部署是将训练好的模型应用到实际生产环境的过程")
print("- 目的：让模型能够处理实际数据，为业务提供价值")
print("- 步骤：模型训练、模型保存、模型加载、API开发、部署")
print("- 挑战：模型版本管理、性能优化、可扩展性")

# 2. 模型训练与保存
print("\n2. 模型训练与保存")

# 加载iris数据集
iris = load_iris()
X, y = iris.data, iris.target

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练模型
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

print(f"模型训练完成，准确率: {model.score(X_test_scaled, y_test):.4f}")

# 2.1 使用pickle保存模型
print("\n2.1 使用pickle保存模型")

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("模型已保存为 model.pkl")

# 2.2 使用joblib保存模型
print("\n2.2 使用joblib保存模型")

joblib.dump(model, 'model.joblib')
print("模型已保存为 model.joblib")

# 2.3 保存标准化器
print("\n2.3 保存标准化器")

joblib.dump(scaler, 'scaler.joblib')
print("标准化器已保存为 scaler.joblib")

# 3. 模型加载与预测
print("\n3. 模型加载与预测")

# 3.1 使用pickle加载模型
print("\n3.1 使用pickle加载模型")

with open('model.pkl', 'rb') as f:
    loaded_model_pickle = pickle.load(f)
print("使用pickle加载模型成功")

# 3.2 使用joblib加载模型
print("\n3.2 使用joblib加载模型")

loaded_model_joblib = joblib.load('model.joblib')
print("使用joblib加载模型成功")

# 3.3 加载标准化器
print("\n3.3 加载标准化器")

loaded_scaler = joblib.load('scaler.joblib')
print("加载标准化器成功")

# 3.4 模型预测
print("\n3.4 模型预测")

test_data = [[5.1, 3.5, 1.4, 0.2]]  # 山鸢尾花的特征
scaled_test_data = loaded_scaler.transform(test_data)
prediction = loaded_model_joblib.predict(scaled_test_data)
prediction_proba = loaded_model_joblib.predict_proba(scaled_test_data)

print(f"测试数据: {test_data}")
print(f"预测结果: {iris.target_names[prediction[0]]}")
print(f"预测概率: {prediction_proba[0]}")

# 4. API开发
print("\n4. API开发")

print("使用Flask开发模型API")

# 创建Flask应用
app = Flask(__name__)

# 加载模型和标准化器
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    """模型预测API"""
    try:
        # 获取请求数据
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({'error': '请提供特征数据'}), 400
        
        # 处理数据
        features = data['features']
        if not isinstance(features, list):
            return jsonify({'error': '特征数据必须是列表'}), 400
        
        # 转换为numpy数组
        features = np.array(features).reshape(1, -1)
        
        # 标准化
        scaled_features = scaler.transform(features)
        
        # 预测
        prediction = model.predict(scaled_features)
        prediction_proba = model.predict_proba(scaled_features)
        
        # 构建响应
        response = {
            'prediction': iris.target_names[prediction[0]],
            'prediction_index': int(prediction[0]),
            'probabilities': prediction_proba[0].tolist(),
            'class_names': iris.target_names.tolist()
        }
        
        return jsonify(response), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """健康检查API"""
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    # 注意：在实际生产环境中，应该使用WSGI服务器如Gunicorn
    print("启动Flask应用...")
    print("API地址: http://localhost:5000")
    print("预测API: POST http://localhost:5000/predict")
    print("健康检查API: GET http://localhost:5000/health")
    print("示例请求:")
    print('curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d "{\"features\": [5.1, 3.5, 1.4, 0.2]}"')
    # app.run(debug=True)

# 5. 模型部署的最佳实践
print("\n5. 模型部署的最佳实践")

print("模型部署的最佳实践:")
print("- 模型版本管理：使用版本控制系统")
print("- 模型监控：监控模型性能和数据漂移")
print("- 模型更新：定期重新训练模型")
print("- 错误处理：完善的错误处理机制")
print("- 安全性：保护API密钥和敏感数据")
print("- 性能优化：使用缓存和批处理")

# 6. Docker容器化
print("\n6. Docker容器化")

print("使用Docker容器化部署模型")

# 创建Dockerfile
with open('Dockerfile', 'w') as f:
    f.write('''FROM python:3.8-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir flask scikit-learn numpy joblib

EXPOSE 5000

CMD ["python", "app.py"]
''')
print("Dockerfile已创建")

# 创建app.py文件
with open('app.py', 'w') as f:
    f.write('''#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# 加载模型和标准化器
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

# 类别名称
class_names = ['setosa', 'versicolor', 'virginica']

@app.route('/predict', methods=['POST'])
def predict():
    """模型预测API"""
    try:
        # 获取请求数据
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({'error': '请提供特征数据'}), 400
        
        # 处理数据
        features = data['features']
        if not isinstance(features, list):
            return jsonify({'error': '特征数据必须是列表'}), 400
        
        # 转换为numpy数组
        features = np.array(features).reshape(1, -1)
        
        # 标准化
        scaled_features = scaler.transform(features)
        
        # 预测
        prediction = model.predict(scaled_features)
        prediction_proba = model.predict_proba(scaled_features)
        
        # 构建响应
        response = {
            'prediction': class_names[prediction[0]],
            'prediction_index': int(prediction[0]),
            'probabilities': prediction_proba[0].tolist(),
            'class_names': class_names
        }
        
        return jsonify(response), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """健康检查API"""
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
''')
print("app.py已创建")

# 创建docker-compose.yml文件
with open('docker-compose.yml', 'w') as f:
    f.write('''version: '3'
services:
  iris-classifier:
    build: .
    ports:
      - "5000:5000"
    restart: always
''')
print("docker-compose.yml已创建")

# 7. 模型部署的评估
print("\n7. 模型部署的评估")

print("模型部署的评估指标:")
print("- 响应时间：API的响应速度")
print("- 吞吐量：单位时间内处理的请求数")
print("- 准确率：模型预测的准确率")
print("- 稳定性：服务的稳定性和可靠性")
print("- 可扩展性：处理高并发的能力")

# 8. 模型部署的工具和框架
print("\n8. 模型部署的工具和框架")

print("常用的模型部署工具和框架:")
print("- Flask：轻量级Web框架")
print("- FastAPI：高性能Web框架")
print("- Django：全功能Web框架")
print("- Docker：容器化技术")
print("- Kubernetes：容器编排")
print("- AWS SageMaker：云服务")
print("- Google Cloud AI Platform：云服务")
print("- Microsoft Azure ML：云服务")

# 9. 模型部署的流程
print("\n9. 模型部署的流程")

print("模型部署的完整流程:")
print("1. 模型训练：使用训练数据训练模型")
print("2. 模型评估：评估模型性能")
print("3. 模型保存：将模型保存为文件")
print("4. API开发：开发模型预测API")
print("5. 容器化：使用Docker容器化应用")
print("6. 部署：部署到生产环境")
print("7. 监控：监控模型性能和服务状态")
print("8. 更新：定期更新模型")

# 10. 练习
print("\n10. 练习")

# 练习1: 实现模型部署
print("练习1: 实现模型部署")
print("- 使用Flask开发一个模型预测API")
print("- 测试API的功能")

# 练习2: Docker容器化
print("\n练习2: Docker容器化")
print("- 使用Docker构建和运行模型服务")
print("- 测试容器化服务的功能")

# 练习3: 模型监控
print("\n练习3: 模型监控")
print("- 实现简单的模型监控功能")
print("- 记录模型的预测结果和性能")

# 练习4: 模型版本管理
print("\n练习4: 模型版本管理")
print("- 实现模型版本管理")
print("- 支持多个模型版本的部署")

# 练习5: 高性能部署
print("\n练习5: 高性能部署")
print("- 使用Gunicorn和Nginx部署模型")
print("- 测试部署的性能")

print("\n=== 第49天学习示例结束 ===")
