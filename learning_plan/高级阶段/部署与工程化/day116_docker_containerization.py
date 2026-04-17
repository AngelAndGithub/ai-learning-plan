#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第116天：Docker和容器化
部署与工程化学习示例
内容：Docker的基本概念、容器化部署和Docker Compose
"""

print("=== 第116天：Docker和容器化 ===")

# 1. Docker基本概念
print("\n1. Docker基本概念")

print("Docker是一个开源的容器化平台")
print("- 容器：轻量级的隔离环境")
print("- 镜像：容器的模板")
print("- 仓库：存储镜像的地方")
print("- Dockerfile：构建镜像的脚本")
print("- Docker Compose：定义和运行多容器应用")

# 2. Docker安装
print("\n2. Docker安装")

print("Docker安装步骤:")
print("1. 访问Docker官网 (https://www.docker.com/)")
print("2. 下载适合你操作系统的Docker安装包")
print("3. 安装Docker")
print("4. 验证安装：docker --version")

# 3. Docker命令
print("\n3. Docker命令")

print("常用Docker命令:")
print("- docker --version：查看Docker版本")
print("- docker info：查看Docker信息")
print("- docker pull [image]：拉取镜像")
print("- docker run [image]：运行容器")
print("- docker ps：查看运行中的容器")
print("- docker ps -a：查看所有容器")
print("- docker stop [container]：停止容器")
print("- docker rm [container]：删除容器")
print("- docker rmi [image]：删除镜像")
print("- docker build -t [name] .：构建镜像")

# 4. Dockerfile
print("\n4. Dockerfile")

print("Dockerfile是构建Docker镜像的脚本")

# 示例：Dockerfile
print("\nDockerfile示例:")
print("# 使用官方Python镜像作为基础镜像")
print("FROM python:3.9-slim")
print("")
print("# 设置工作目录")
print("WORKDIR /app")
print("")
print("# 复制依赖文件")
print("COPY requirements.txt .")
print("")
print("# 安装依赖")
print("RUN pip install --no-cache-dir -r requirements.txt")
print("")
print("# 复制应用代码")
print("COPY . .")
print("")
print("# 暴露端口")
print("EXPOSE 5000")
print("")
print("# 运行应用")
print("CMD [\"python\", \"app.py\"]")

# 5. 构建和运行容器
print("\n5. 构建和运行容器")

print("构建和运行容器的步骤:")
print("1. 创建Dockerfile")
print("2. 构建镜像：docker build -t myapp .")
print("3. 运行容器：docker run -p 5000:5000 myapp")

# 6. Docker Compose
print("\n6. Docker Compose")

print("Docker Compose用于定义和运行多容器应用")

# 示例：docker-compose.yml
print("\ndocker-compose.yml示例:")
print("version: '3'")
print("services:")
print("  web:")
print("    build: .")
print("    ports:")
print("      - "5000:5000"")
print("    depends_on:")
print("      - redis")
print("  redis:")
print("    image: redis:alpine")

# 7. 容器化部署
print("\n7. 容器化部署")

print("容器化部署的优势:")
print("- 环境一致性")
print("- 快速部署")
print("- 资源隔离")
print("- 可扩展性")
print("- 简化管理")

# 8. Docker仓库
print("\n8. Docker仓库")

print("Docker仓库用于存储和分享镜像")
print("- Docker Hub：公共Docker仓库")
print("- 私有仓库：企业内部使用")
print("- 阿里云容器镜像服务")
print("- 腾讯云容器镜像服务")

# 9. 容器编排
print("\n9. 容器编排")

print("容器编排用于管理多个容器")
print("- Kubernetes (K8s)")
print("- Docker Swarm")
print("- Mesos")

# 10. 练习
print("\n10. 练习")

# 练习1: 构建Docker镜像
print("练习1: 构建Docker镜像")
print("- 创建Dockerfile")
print("- 构建镜像")
print("- 运行容器")

# 练习2: 使用Docker Compose
print("\n练习2: 使用Docker Compose")
print("- 创建docker-compose.yml")
print("- 启动多容器应用")
print("- 停止和清理")

# 练习3: 部署到云服务
print("\n练习3: 部署到云服务")
print("- 推送镜像到Docker Hub")
print("- 在云服务上运行容器")
print("- 配置服务")

# 练习4: 容器编排
print("\n练习4: 容器编排")
print("- 学习Kubernetes基本概念")
print("- 部署应用到Kubernetes")
print("- 管理Kubernetes集群")

print("\n=== 第116天学习示例结束 ===")
