#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第115天：API开发
部署与工程化学习示例
内容：API开发的基本概念、Flask和FastAPI框架
"""

print("=== 第115天：API开发 ===")

# 1. API开发基本概念
print("\n1. API开发基本概念")

import json
import requests
from flask import Flask, request, jsonify
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

print("API (Application Programming Interface) 是应用程序之间通信的接口")
print("- RESTful API：基于HTTP协议的API设计风格")
print("- GraphQL：一种用于API的查询语言")
print("- gRPC：基于HTTP/2的高性能RPC框架")
print("- 常见HTTP方法：GET、POST、PUT、DELETE")
print("- 状态码：200 (成功)、400 (请求错误)、404 (资源不存在)、500 (服务器错误)")

# 2. Flask框架
print("\n2. Flask框架")

print("Flask是一个轻量级的Python Web框架")
print("- 简单易用")
print("- 灵活可扩展")
print("- 适合构建小型API")

# 示例：Flask API
print("\nFlask API示例:")
print("from flask import Flask, request, jsonify")
print("")
print("app = Flask(__name__)")
print("")
print("# 简单的GET请求")
print("@app.route('/api/hello', methods=['GET'])")
print("def hello():")
print("    return jsonify({'message': 'Hello, World!'})")
print("")
print("# POST请求")
print("@app.route('/api/add', methods=['POST'])")
print("def add():")
print("    data = request.json")
print("    a = data.get('a')")
print("    b = data.get('b')")
print("    if a is None or b is None:")
print("        return jsonify({'error': 'Missing parameters'}), 400")
print("    result = a + b")
print("    return jsonify({'result': result})")
print("")
print("if __name__ == '__main__':")
print("    app.run(debug=True)")

# 3. FastAPI框架
print("\n3. FastAPI框架")

print("FastAPI是一个现代化的Python Web框架")
print("- 高性能")
print("- 自动生成API文档")
print("- 类型提示")
print("- 异步支持")

# 示例：FastAPI
print("\nFastAPI示例:")
print("from fastapi import FastAPI, HTTPException")
print("from pydantic import BaseModel")
print("")
print("app = FastAPI()")
print("")
print("# 数据模型")
print("class Item(BaseModel):")
print("    a: int")
print("    b: int")
print("")
print("# 简单的GET请求")
print("@app.get('/api/hello')")
print("def hello():")
print("    return {'message': 'Hello, World!'}")
print("")
print("# POST请求")
print("@app.post('/api/add')")
print("def add(item: Item):")
print("    result = item.a + item.b")
print("    return {'result': result}")
print("")
print("# 启动服务器")
print("# uvicorn main:app --reload")

# 4. API认证
print("\n4. API认证")

print("API认证是保护API安全的重要措施")
print("- API密钥")
print("- 基本认证 (Basic Auth)")
print("- 令牌认证 (Token Auth)")
print("- OAuth 2.0")
print("- JWT (JSON Web Token)")

# 示例：JWT认证
print("\nJWT认证示例:")
print("import jwt")
print("from flask import Flask, request, jsonify")
print("")
print("app = Flask(__name__)")
print("app.config['SECRET_KEY'] = 'your-secret-key'")
print("")
print("# 生成JWT令牌")
print("def generate_token(user_id):")
print("    token = jwt.encode({'user_id': user_id}, app.config['SECRET_KEY'], algorithm='HS256')")
print("    return token")
print("")
print("# 验证JWT令牌")
print("def verify_token(token):")
print("    try:")
print("        data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])")
print("        return data['user_id']")
print("    except:")
print("        return None")
print("")
print("# 登录接口")
print("@app.post('/api/login')")
print("def login():")
print("    data = request.json")
print("    username = data.get('username')")
print("    password = data.get('password')")
print("    # 简单验证")
print("    if username == 'admin' and password == 'password':")
print("        token = generate_token(1)")
print("        return jsonify({'token': token})")
print("    return jsonify({'error': 'Invalid credentials'}), 401")
print("")
print("# 需要认证的接口")
print("@app.get('/api/protected')")
print("def protected():")
print("    token = request.headers.get('Authorization')")
print("    if not token:")
print("        return jsonify({'error': 'Token required'}), 401")
print("    token = token.split(' ')[1]  # 去掉 'Bearer ' 前缀")
print("    user_id = verify_token(token)")
print("    if not user_id:")
print("        return jsonify({'error': 'Invalid token'}), 401")
print("    return jsonify({'message': f'Hello, user {user_id}!'})")

# 5. API文档
print("\n5. API文档")

print("API文档是API的重要组成部分")
print("- Swagger UI：自动生成的API文档")
print("- ReDoc：另一种API文档风格")
print("- 手动编写的API文档")

# 6. API测试
print("\n6. API测试")

print("API测试是确保API质量的重要手段")
print("- 单元测试")
print("- 集成测试")
print("- 端到端测试")
print("- 性能测试")

# 示例：使用requests测试API
print("\n使用requests测试API:")
print("import requests")
print("")
print("# 测试GET请求")
print("response = requests.get('http://localhost:5000/api/hello')")
print("print('GET response:', response.json())")
print("")
print("# 测试POST请求")
print("data = {'a': 10, 'b': 20}")
print("response = requests.post('http://localhost:5000/api/add', json=data)")
print("print('POST response:', response.json())")

# 7. API部署
print("\n7. API部署")

print("API部署是将API上线的过程")
print("- 本地开发服务器")
print("- 生产服务器")
print("- 容器化部署")
print("- 云服务部署")

# 8. API性能优化
print("\n8. API性能优化")

print("API性能优化的方法:")
print("1. 缓存")
print("2. 数据库优化")
print("3. 代码优化")
print("4. 负载均衡")
print("5. 异步处理")

# 9. API安全
print("\n9. API安全")

print("API安全的重要性:")
print("- 认证和授权")
print("- 输入验证")
print("- 防止SQL注入")
print("- 防止XSS攻击")
print("- 防止CSRF攻击")
print("- HTTPS")

# 10. 练习
print("\n10. 练习")

# 练习1: 构建Flask API
print("练习1: 构建Flask API")
print("- 实现基本的CRUD操作")
print("- 添加认证")
print("- 测试API")

# 练习2: 构建FastAPI
print("\n练习2: 构建FastAPI")
print("- 实现基本的CRUD操作")
print("- 使用Pydantic模型")
print("- 测试API")

# 练习3: API认证
print("\n练习3: API认证")
print("- 实现JWT认证")
print("- 添加权限控制")
print("- 测试认证")

# 练习4: API部署
print("\n练习4: API部署")
print("- 容器化部署")
print("- 云服务部署")
print("- 监控API")

print("\n=== 第115天学习示例结束 ===")
