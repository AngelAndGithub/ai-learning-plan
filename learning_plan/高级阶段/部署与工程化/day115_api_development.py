#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第115天：API开发
部署与工程化学习示例
内容：API开发的基本概念、Flask、FastAPI、API文档
"""

print("=== 第115天：API开发 ===")

import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import pickle
import os

# 1. API开发概述
print("\n1. API开发概述")

print("API（应用程序编程接口）是不同软件系统之间通信的桥梁")
print("- 目的：使不同系统能够相互交互")
print("- 类型：RESTful API、GraphQL API、gRPC等")
print("- 应用：微服务架构、移动应用后端、第三方集成等")

# 2. 准备模型
print("\n2. 准备模型")

# 创建示例模型
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 加载数据集
iris = load_iris()
X, y = iris.data, iris.target

# 数据预处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 训练模型
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_scaled, y)

# 创建模型保存目录
if not os.path.exists('models'):
    os.makedirs('models')

# 保存模型和标准化器
joblib.dump(model, 'models/iris_model.joblib')
joblib.dump(scaler, 'models/iris_scaler.joblib')
print("模型已保存到 models/ 目录")

# 3. Flask API
print("\n3. Flask API")

print("Flask是一个轻量级的Python Web框架")
print("- 优点：简单易用，灵活，适合小型应用")
print("- 缺点：性能相对较低，缺少自动API文档")

# 创建Flask应用文件
with open('flask_app.py', 'w') as f:
    f.write('''#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# 加载模型和标准化器
model = joblib.load('models/iris_model.joblib')
scaler = joblib.load('models/iris_scaler.joblib')

# 类别名称
class_names = ['setosa', 'versicolor', 'virginica']

@app.route('/predict', methods=['POST'])
def predict():
    """预测API"""
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
print("Flask应用已创建为 flask_app.py")

# 4. FastAPI
print("\n4. FastAPI")

print("FastAPI是一个现代、快速的Web框架")
print("- 优点：高性能，自动生成API文档，类型提示")
print("- 缺点：相对较新，生态系统不如Flask成熟")

# 创建FastAPI应用文件
with open('fastapi_app.py', 'w') as f:
    f.write('''#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Iris Classification API", description="使用机器学习模型预测鸢尾花类别")

# 加载模型和标准化器
model = joblib.load('models/iris_model.joblib')
scaler = joblib.load('models/iris_scaler.joblib')

# 类别名称
class_names = ['setosa', 'versicolor', 'virginica']

# 请求模型
class PredictionRequest(BaseModel):
    features: list

# 响应模型
class PredictionResponse(BaseModel):
    prediction: str
    prediction_index: int
    probabilities: list
    class_names: list

@app.post('/predict', response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """预测API"""
    try:
        # 处理数据
        features = request.features
        if not isinstance(features, list):
            raise HTTPException(status_code=400, detail="特征数据必须是列表")
        
        # 转换为numpy数组
        features = np.array(features).reshape(1, -1)
        
        # 标准化
        scaled_features = scaler.transform(features)
        
        # 预测
        prediction = model.predict(scaled_features)
        prediction_proba = model.predict_proba(scaled_features)
        
        # 构建响应
        response = PredictionResponse(
            prediction=class_names[prediction[0]],
            prediction_index=int(prediction[0]),
            probabilities=prediction_proba[0].tolist(),
            class_names=class_names
        )
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/health')
def health():
    """健康检查API"""
    return {'status': 'healthy'}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
''')
print("FastAPI应用已创建为 fastapi_app.py")

# 5. API文档
print("\n5. API文档")

print("API文档是API开发的重要组成部分")
print("- 目的：帮助开发者理解和使用API")
print("- 工具：Swagger UI、ReDoc、Postman等")
print("- 内容：API端点、请求参数、响应格式等")

print("\nFastAPI自动生成API文档:")
print("- Swagger UI: http://localhost:8000/docs")
print("- ReDoc: http://localhost:8000/redoc")

# 6. API测试
print("\n6. API测试")

print("API测试是确保API功能正常的重要步骤")
print("- 工具：Postman、curl、requests库等")
print("- 测试内容：功能测试、性能测试、安全测试等")

# 创建测试脚本
with open('test_api.py', 'w') as f:
    f.write('''#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import json

# 测试Flask API
def test_flask_api():
    print("测试Flask API...")
    url = "http://localhost:5000/predict"
    headers = {"Content-Type": "application/json"}
    data = {"features": [5.1, 3.5, 1.4, 0.2]}
    
    response = requests.post(url, headers=headers, data=json.dumps(data))
    print(f"Flask API响应状态码: {response.status_code}")
    print(f"Flask API响应内容: {response.json()}")

# 测试FastAPI
def test_fastapi_api():
    print("\n测试FastAPI API...")
    url = "http://localhost:8000/predict"
    headers = {"Content-Type": "application/json"}
    data = {"features": [5.1, 3.5, 1.4, 0.2]}
    
    response = requests.post(url, headers=headers, data=json.dumps(data))
    print(f"FastAPI API响应状态码: {response.status_code}")
    print(f"FastAPI API响应内容: {response.json()}")

# 测试健康检查
def test_health_check():
    print("\n测试健康检查...")
    
    # Flask健康检查
    flask_url = "http://localhost:5000/health"
    flask_response = requests.get(flask_url)
    print(f"Flask健康检查响应: {flask_response.json()}")
    
    # FastAPI健康检查
    fastapi_url = "http://localhost:8000/health"
    fastapi_response = requests.get(fastapi_url)
    print(f"FastAPI健康检查响应: {fastapi_response.json()}")

if __name__ == "__main__":
    test_flask_api()
    test_fastapi_api()
    test_health_check()
''')
print("API测试脚本已创建为 test_api.py")

# 7. API部署
print("\n7. API部署")

print("API部署是将API应用到生产环境的过程")
print("- 步骤：")
print("  1. 准备环境")
print("  2. 安装依赖")
print("  3. 配置服务")
print("  4. 启动服务")
print("- 平台：本地服务器、云服务（AWS、Azure、GCP）、容器平台（Docker、Kubernetes）")

# 创建requirements.txt
with open('requirements.txt', 'w') as f:
    f.write('''Flask==2.0.1
fastapi==0.68.1
uvicorn==0.15.0
scikit-learn==0.24.2
numpy==1.20.3
pandas==1.3.3
joblib==1.0.1
requests==2.26.0
''')
print("依赖文件已创建为 requirements.txt")

# 8. API安全
print("\n8. API安全")

print("API安全是保护API免受攻击的重要措施")
print("- 安全措施：")
print("  - 认证：API密钥、OAuth、JWT")
print("  - 授权：基于角色的访问控制")
print("  - 加密：HTTPS")
print("  - 速率限制：防止DoS攻击")
print("  - 输入验证：防止注入攻击")

# 9. API性能优化
print("\n9. API性能优化")

print("API性能优化是提高API响应速度的重要手段")
print("- 优化方法：")
print("  - 缓存：缓存频繁请求的结果")
print("  - 异步处理：处理耗时操作")
print("  - 数据库优化：索引、查询优化")
print("  - 代码优化：减少不必要的计算")
print("  - 负载均衡：分散请求压力")

# 10. 练习
print("\n10. 练习")

# 练习1: Flask API
print("练习1: Flask API")
print("- 实现一个Flask API")
print("- 测试API功能")

# 练习2: FastAPI
print("\n练习2: FastAPI")
print("- 实现一个FastAPI")
print("- 测试API功能")

# 练习3: API文档
print("\n练习3: API文档")
print("- 为API添加详细的文档")
print("- 使用Swagger UI测试API")

# 练习4: API安全
print("\n练习4: API安全")
print("- 为API添加认证机制")
print("- 测试认证功能")

# 练习5: API部署
print("\n练习5: API部署")
print("- 部署API到云服务")
print("- 测试部署的API")

print("\n=== 第115天学习示例结束 ===")
