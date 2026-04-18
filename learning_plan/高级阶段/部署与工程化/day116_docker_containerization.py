# 第116天：Docker容器化

"""
Docker容器化学习示例

今天的学习内容：
1. Docker基础概念
2. Dockerfile编写
3. Docker Compose配置
4. 多阶段构建
5. 容器化AI模型服务
6. 容器管理与优化
"""

import os
import subprocess

# 1. Docker基础概念
def docker_basics():
    """Docker基础概念介绍"""
    print("=== Docker基础概念 ===")
    print("Docker是一个开源的容器化平台，用于构建、部署和运行应用程序")
    print("核心概念：")
    print("- 镜像(Image)：应用程序的只读模板")
    print("- 容器(Container)：镜像的运行实例")
    print("- 仓库(Repository)：存储镜像的地方")
    print("- 网络(Network)：容器间通信的方式")
    print("- 卷(Volume)：持久化存储数据")

# 2. Dockerfile编写示例
def create_dockerfile():
    """创建Dockerfile示例"""
    print("\n=== Dockerfile编写 ===")
    dockerfile_content = '''
# 使用官方Python镜像作为基础
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 暴露端口
EXPOSE 8000

# 运行应用
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
'''
    
    # 写入Dockerfile
    with open('Dockerfile', 'w', encoding='utf-8') as f:
        f.write(dockerfile_content)
    
    print("创建了Dockerfile文件")
    print("Dockerfile内容：")
    print(dockerfile_content)

# 3. 创建requirements.txt文件
def create_requirements():
    """创建requirements.txt文件"""
    requirements_content = '''
fastapi
uvicorn
scikit-learn
pandas
numpy
'''
    
    with open('requirements.txt', 'w', encoding='utf-8') as f:
        f.write(requirements_content)
    
    print("\n创建了requirements.txt文件")

# 4. 创建简单的FastAPI应用
def create_app():
    """创建简单的FastAPI应用"""
    app_content = '''
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# 加载模型（示例，实际使用时需要训练模型）
# model = joblib.load('model.pkl')

class PredictionRequest(BaseModel):
    features: list

class PredictionResponse(BaseModel):
    prediction: int
    confidence: float

@app.post('/predict', response_model=PredictionResponse)
def predict(request: PredictionRequest):
    # 模拟预测
    features = np.array(request.features).reshape(1, -1)
    # prediction = model.predict(features)[0]
    # confidence = model.predict_proba(features).max()
    
    # 模拟结果
    prediction = 1
    confidence = 0.95
    
    return PredictionResponse(
        prediction=prediction,
        confidence=confidence
    )

@app.get('/')
def read_root():
    return {"message": "AI Model API"}
'''
    
    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(app_content)
    
    print("\n创建了app.py文件")

# 5. Docker Compose配置
def create_docker_compose():
    """创建Docker Compose配置文件"""
    print("\n=== Docker Compose配置 ===")
    docker_compose_content = '''
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - .:/app
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
'''
    
    with open('docker-compose.yml', 'w', encoding='utf-8') as f:
        f.write(docker_compose_content)
    
    print("创建了docker-compose.yml文件")
    print("Docker Compose内容：")
    print(docker_compose_content)

# 6. 多阶段构建示例
def create_multi_stage_dockerfile():
    """创建多阶段构建的Dockerfile"""
    print("\n=== 多阶段构建 ===")
    multi_stage_dockerfile = '''
# 第一阶段：构建环境
FROM python:3.9-slim as builder

WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装依赖到虚拟环境
RUN python -m venv venv && \
    venv/bin/pip install --no-cache-dir -r requirements.txt

# 第二阶段：运行环境
FROM python:3.9-slim

WORKDIR /app

# 从构建阶段复制虚拟环境
COPY --from=builder /app/venv /app/venv

# 复制应用代码
COPY . .

# 激活虚拟环境
ENV PATH="/app/venv/bin:$PATH"

# 暴露端口
EXPOSE 8000

# 运行应用
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
'''
    
    with open('Dockerfile.multi', 'w', encoding='utf-8') as f:
        f.write(multi_stage_dockerfile)
    
    print("创建了Dockerfile.multi文件")
    print("多阶段构建Dockerfile内容：")
    print(multi_stage_dockerfile)

# 7. 容器管理命令
def docker_commands():
    """Docker常用命令"""
    print("\n=== Docker常用命令 ===")
    commands = [
        "docker build -t ai-model-api .",  # 构建镜像
        "docker run -p 8000:8000 ai-model-api",  # 运行容器
        "docker ps",  # 查看运行中的容器
        "docker images",  # 查看镜像
        "docker stop <container_id>",  # 停止容器
        "docker rm <container_id>",  # 删除容器
        "docker rmi <image_id>",  # 删除镜像
        "docker-compose up -d",  # 使用Compose启动服务
        "docker-compose down",  # 停止Compose服务
        "docker logs <container_id>",  # 查看容器日志
    ]
    
    for cmd in commands:
        print(f"- {cmd}")

# 8. 容器优化建议
def docker_optimization():
    """Docker容器优化建议"""
    print("\n=== 容器优化建议 ===")
    tips = [
        "使用官方基础镜像",
        "使用多阶段构建减小镜像体积",
        "合理使用缓存层",
        "避免在容器中运行不必要的服务",
        "使用轻量级基础镜像（如alpine）",
        "合理配置资源限制（CPU、内存）",
        "使用健康检查确保容器状态",
        "定期清理未使用的镜像和容器",
    ]
    
    for tip in tips:
        print(f"- {tip}")

# 9. 实际操作示例
def docker_demo():
    """Docker操作示例"""
    print("\n=== Docker操作示例 ===")
    print("以下是Docker容器化的完整流程：")
    print("1. 创建应用代码和依赖文件")
    print("2. 编写Dockerfile")
    print("3. 构建镜像: docker build -t ai-model-api .")
    print("4. 运行容器: docker run -p 8000:8000 ai-model-api")
    print("5. 访问API: http://localhost:8000")
    print("6. 使用Docker Compose管理多容器应用")

if __name__ == "__main__":
    # 执行所有示例
    docker_basics()
    create_requirements()
    create_app()
    create_dockerfile()
    create_docker_compose()
    create_multi_stage_dockerfile()
    docker_commands()
    docker_optimization()
    docker_demo()
    
    print("\n=== 学习完成 ===")
    print("今天学习了Docker容器化的核心概念和实践应用")
    print("掌握了Dockerfile编写、Docker Compose配置和容器管理技巧")
    print("可以使用这些知识来容器化AI模型服务，实现更高效的部署和管理")
