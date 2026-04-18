# 第117天：CI/CD（持续集成/持续部署）

"""
CI/CD学习示例

今天的学习内容：
1. CI/CD基础概念
2. GitHub Actions配置
3. GitLab CI/CD配置
4. CI/CD流水线设计
5. 自动化测试集成
6. 部署策略
7. 监控与告警
8. CI/CD最佳实践
"""

# 1. CI/CD基础概念
def ci_cd_basics():
    """CI/CD基础概念介绍"""
    print("=== CI/CD基础概念 ===")
    print("CI/CD是持续集成(Continuous Integration)和持续部署(Continuous Deployment)的缩写")
    print("核心概念：")
    print("- 持续集成(CI)：频繁地将代码集成到主分支，自动进行构建和测试")
    print("- 持续部署(CD)：代码通过测试后自动部署到生产环境")
    print("- 持续交付(CD)：代码通过测试后准备好部署，但需要手动触发")
    print("CI/CD的好处：")
    print("- 减少人工错误")
    print("- 提高开发效率")
    print("- 加快交付速度")
    print("- 保证代码质量")

# 2. GitHub Actions配置
def github_actions_config():
    """GitHub Actions配置示例"""
    print("\n=== GitHub Actions配置 ===")
    workflow_content = '''
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Run tests
      run: |
        python -m pytest tests/ -v
    
  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to production
      run: |
        echo "Deploying to production..."
        # 部署命令
'''
    
    # 创建目录结构
    import os
    os.makedirs('.github/workflows', exist_ok=True)
    
    # 写入GitHub Actions配置文件
    with open('.github/workflows/ci-cd.yml', 'w', encoding='utf-8') as f:
        f.write(workflow_content)
    
    print("创建了.github/workflows/ci-cd.yml文件")
    print("GitHub Actions配置内容：")
    print(workflow_content)

# 3. GitLab CI/CD配置
def gitlab_ci_config():
    """GitLab CI/CD配置示例"""
    print("\n=== GitLab CI/CD配置 ===")
    gitlab_ci_content = '''
stages:
  - build
  - test
  - deploy

variables:
  PYTHON_VERSION: "3.9"

build:
  stage: build
  image: python:${PYTHON_VERSION}-slim
  script:
    - pip install --upgrade pip
    - pip install -r requirements.txt
    - echo "Build completed successfully"

test:
  stage: test
  image: python:${PYTHON_VERSION}-slim
  script:
    - pip install pytest
    - python -m pytest tests/ -v
  needs:
    - build

deploy_production:
  stage: deploy
  image: python:${PYTHON_VERSION}-slim
  script:
    - echo "Deploying to production..."
    # 部署命令
  environment:
    name: production
  only:
    - main
  needs:
    - test
'''
    
    # 写入GitLab CI配置文件
    with open('.gitlab-ci.yml', 'w', encoding='utf-8') as f:
        f.write(gitlab_ci_content)
    
    print("创建了.gitlab-ci.yml文件")
    print("GitLab CI/CD配置内容：")
    print(gitlab_ci_content)

# 4. CI/CD流水线设计
def pipeline_design():
    """CI/CD流水线设计"""
    print("\n=== CI/CD流水线设计 ===")
    print("典型的CI/CD流水线包含以下阶段：")
    stages = [
        "1. 代码检查：使用lint工具检查代码质量",
        "2. 依赖安装：安装项目所需的依赖包",
        "3. 单元测试：运行单元测试确保代码功能正常",
        "4. 集成测试：测试模块间的交互",
        "5. 构建：构建应用程序或容器镜像",
        "6. 部署：部署到测试/预生产/生产环境",
        "7. 监控：监控应用程序运行状态",
    ]
    
    for stage in stages:
        print(f"- {stage}")

# 5. 自动化测试集成
def test_integration():
    """自动化测试集成"""
    print("\n=== 自动化测试集成 ===")
    print("在CI/CD中集成自动化测试：")
    
    # 创建测试目录和示例测试文件
    test_content = '''
import pytest
from app import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "AI Model API"}

def test_predict():
    response = client.post("/predict", json={"features": [1.0, 2.0, 3.0, 4.0]})
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "confidence" in response.json()
'''
    
    # 创建tests目录
    import os
    os.makedirs('tests', exist_ok=True)
    
    # 写入测试文件
    with open('tests/test_app.py', 'w', encoding='utf-8') as f:
        f.write(test_content)
    
    print("创建了tests/test_app.py文件")
    print("测试文件内容：")
    print(test_content)

# 6. 部署策略
def deployment_strategies():
    """部署策略"""
    print("\n=== 部署策略 ===")
    strategies = [
        "1. 蓝绿部署：同时运行两个环境，切换流量",
        "2. 滚动部署：逐步替换旧版本",
        "3. 金丝雀部署：先部署到小部分用户",
        "4. A/B测试：同时运行多个版本进行比较",
        "5. 灰度发布：逐步扩大部署范围",
    ]
    
    for strategy in strategies:
        print(f"- {strategy}")

# 7. 监控与告警
def monitoring_and_alerting():
    """监控与告警"""
    print("\n=== 监控与告警 ===")
    print("CI/CD中的监控与告警：")
    monitoring = [
        "1. 构建状态监控：监控CI/CD流水线的执行状态",
        "2. 应用性能监控：监控部署后应用的性能指标",
        "3. 错误率监控：监控应用的错误率和异常情况",
        "4. 资源使用监控：监控服务器资源使用情况",
        "5. 告警机制：设置阈值触发告警通知",
    ]
    
    for item in monitoring:
        print(f"- {item}")

# 8. CI/CD最佳实践
def ci_cd_best_practices():
    """CI/CD最佳实践"""
    print("\n=== CI/CD最佳实践 ===")
    practices = [
        "1. 保持流水线简洁：每个任务只做一件事",
        "2. 快速反馈：确保构建和测试快速完成",
        "3. 环境一致性：确保各个环境配置一致",
        "4. 安全集成：在CI/CD中集成安全扫描",
        "5. 版本控制：对配置文件进行版本控制",
        "6. 自动化一切：尽可能自动化所有流程",
        "7. 回滚机制：确保出现问题时能够快速回滚",
        "8. 文档完善：详细记录CI/CD流程和配置",
    ]
    
    for practice in practices:
        print(f"- {practice}")

# 9. 实际操作示例
def ci_cd_demo():
    """CI/CD操作示例"""
    print("\n=== CI/CD操作示例 ===")
    print("以下是CI/CD的完整流程：")
    print("1. 开发者提交代码到版本控制系统")
    print("2. CI系统自动触发构建和测试")
    print("3. 测试通过后，自动部署到测试环境")
    print("4. 测试环境验证通过后，部署到预生产环境")
    print("5. 预生产环境验证通过后，部署到生产环境")
    print("6. 监控系统持续监控应用状态")

if __name__ == "__main__":
    # 执行所有示例
    ci_cd_basics()
    github_actions_config()
    gitlab_ci_config()
    pipeline_design()
    test_integration()
    deployment_strategies()
    monitoring_and_alerting()
    ci_cd_best_practices()
    ci_cd_demo()
    
    print("\n=== 学习完成 ===")
    print("今天学习了CI/CD的核心概念和实践应用")
    print("掌握了GitHub Actions和GitLab CI/CD的配置方法")
    print("了解了CI/CD流水线设计、自动化测试集成和部署策略")
    print("可以使用这些知识来构建自动化的软件交付流程，提高开发效率和代码质量")
