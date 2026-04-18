# 第118天：版本控制（Git和GitHub）

"""
版本控制学习示例

今天的学习内容：
1. Git基础概念
2. Git常用命令
3. GitHub操作
4. 分支管理
5. 远程仓库管理
6. 协作工作流
7. Git最佳实践
8. GitHub CLI使用
"""

import os
import subprocess

# 1. Git基础概念
def git_basics():
    """Git基础概念介绍"""
    print("=== Git基础概念 ===")
    print("Git是一个分布式版本控制系统，用于跟踪代码的变更")
    print("核心概念：")
    print("- 仓库(Repository)：存储代码的地方")
    print("- 提交(Commit)：代码的快照")
    print("- 分支(Branch)：代码的不同版本")
    print("- 合并(Merge)：将不同分支的代码合并")
    print("- 远程(Remote)：远程仓库地址")
    print("- 推送(Push)：将本地代码推送到远程仓库")
    print("- 拉取(Pull)：从远程仓库拉取代码")
    print("- 克隆(Clone)：复制远程仓库到本地")

# 2. Git常用命令
def git_commands():
    """Git常用命令"""
    print("\n=== Git常用命令 ===")
    commands = [
        "git init",  # 初始化Git仓库
        "git add <file>",  # 添加文件到暂存区
        "git add .",  # 添加所有文件到暂存区
        "git commit -m \"commit message\"",  # 提交代码
        "git status",  # 查看仓库状态
        "git log",  # 查看提交历史
        "git diff",  # 查看代码变更
        "git branch",  # 查看分支
        "git branch <branch-name>",  # 创建分支
        "git checkout <branch-name>",  # 切换分支
        "git checkout -b <branch-name>",  # 创建并切换分支
        "git merge <branch-name>",  # 合并分支
        "git remote -v",  # 查看远程仓库
        "git remote add origin <url>",  # 添加远程仓库
        "git push -u origin main",  # 推送代码到远程仓库
        "git pull",  # 从远程仓库拉取代码
        "git clone <url>",  # 克隆远程仓库
        "git reset HEAD <file>",  # 取消暂存文件
        "git checkout -- <file>",  # 撤销文件修改
        "git stash",  # 暂存未提交的修改
        "git stash pop",  # 恢复暂存的修改
    ]
    
    for cmd in commands:
        print(f"- {cmd}")

# 3. GitHub操作
def github_operations():
    """GitHub操作"""
    print("\n=== GitHub操作 ===")
    print("GitHub是一个基于Git的代码托管平台")
    operations = [
        "1. 创建仓库：在GitHub上创建新的代码仓库",
        "2. 克隆仓库：将GitHub上的仓库克隆到本地",
        "3. 推送代码：将本地代码推送到GitHub仓库",
        "4. 拉取请求(Pull Request)：请求将代码合并到主分支",
        "5. 问题(Issue)：跟踪bug和功能请求",
        "6. 项目(Project)：管理项目任务",
        "7. 动作(Action)：自动化工作流程",
        "8. Wiki：项目文档",
        "9. 发布(Release)：发布软件版本",
        "10. 分支保护：保护重要分支",
    ]
    
    for operation in operations:
        print(f"- {operation}")

# 4. 分支管理
def branch_management():
    """分支管理"""
    print("\n=== 分支管理 ===")
    print("分支管理策略：")
    strategies = [
        "1. 主分支(main/master)：稳定的生产代码",
        "2. 开发分支(develop)：集成所有功能开发",
        "3. 特性分支(feature/*)：开发新功能",
        "4. 发布分支(release/*)：准备发布版本",
        "5. 热修复分支(hotfix/*)：修复生产环境bug",
    ]
    
    for strategy in strategies:
        print(f"- {strategy}")

# 5. 远程仓库管理
def remote_repository_management():
    """远程仓库管理"""
    print("\n=== 远程仓库管理 ===")
    management = [
        "1. 添加远程仓库：git remote add origin <url>",
        "2. 查看远程仓库：git remote -v",
        "3. 重命名远程仓库：git remote rename <old-name> <new-name>",
        "4. 删除远程仓库：git remote remove <name>",
        "5. 推送代码：git push <remote> <branch>",
        "6. 拉取代码：git pull <remote> <branch>",
        "7. 强制推送：git push -f <remote> <branch>",
        "8. 推送标签：git push --tags",
    ]
    
    for item in management:
        print(f"- {item}")

# 6. 协作工作流
def collaboration_workflow():
    """协作工作流"""
    print("\n=== 协作工作流 ===")
    print("常见的Git工作流：")
    workflows = [
        "1. 集中式工作流：所有开发者直接推送到主分支",
        "2. 功能分支工作流：每个功能创建独立分支",
        "3. GitFlow工作流：严格的分支管理策略",
        "4. Forking工作流：通过Fork和Pull Request协作",
    ]
    
    for workflow in workflows:
        print(f"- {workflow}")

# 7. Git最佳实践
def git_best_practices():
    """Git最佳实践"""
    print("\n=== Git最佳实践 ===")
    practices = [
        "1. 提交信息清晰：使用有意义的提交信息",
        "2. 小而频繁的提交：每次提交只包含相关的变更",
        "3. 分支命名规范：使用有意义的分支名称",
        "4. 定期拉取：保持本地代码与远程同步",
        "5. 解决冲突：及时解决代码冲突",
        "6. 使用.gitignore：忽略不需要版本控制的文件",
        "7. 定期备份：确保代码安全",
        "8. 代码审查：通过Pull Request进行代码审查",
    ]
    
    for practice in practices:
        print(f"- {practice}")

# 8. GitHub CLI使用
def github_cli():
    """GitHub CLI使用"""
    print("\n=== GitHub CLI使用 ===")
    cli_commands = [
        "gh auth login",  # 登录GitHub
        "gh repo create",  # 创建仓库
        "gh repo clone <owner/repo>",  # 克隆仓库
        "gh pr create",  # 创建Pull Request
        "gh pr list",  # 列出Pull Request
        "gh issue create",  # 创建Issue
        "gh issue list",  # 列出Issue
        "gh release create",  # 创建发布
        "gh workflow run",  # 运行工作流
    ]
    
    for cmd in cli_commands:
        print(f"- {cmd}")

# 9. 实际操作示例
def git_demo():
    """Git操作示例"""
    print("\n=== Git操作示例 ===")
    print("以下是Git和GitHub的完整操作流程：")
    print("1. 初始化仓库：git init")
    print("2. 添加文件：git add .")
    print("3. 提交代码：git commit -m \"Initial commit\"")
    print("4. 在GitHub上创建仓库")
    print("5. 添加远程仓库：git remote add origin <url>")
    print("6. 推送代码：git push -u origin main")
    print("7. 创建分支：git checkout -b feature/new-feature")
    print("8. 开发新功能并提交")
    print("9. 推送分支：git push origin feature/new-feature")
    print("10. 在GitHub上创建Pull Request")
    print("11. 合并Pull Request")
    print("12. 拉取最新代码：git pull")

# 10. 创建.gitignore文件
def create_gitignore():
    """创建.gitignore文件"""
    print("\n=== 创建.gitignore文件 ===")
    gitignore_content = '''
# Python
__pycache__/
*.py[cod]
*$py.class

# Environment variables
.env
.env.local
.env.development.local
.env.test.local
.env.production.local

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Docker
Dockerfile*
!Dockerfile.example
.dockerignore

# CI/CD
.github/workflows/
.gitlab-ci.yml

# Testing
coverage/
.pytest_cache/
'''
    
    # 写入.gitignore文件
    with open('.gitignore', 'w', encoding='utf-8') as f:
        f.write(gitignore_content)
    
    print("创建了.gitignore文件")
    print(".gitignore文件内容：")
    print(gitignore_content)

if __name__ == "__main__":
    # 执行所有示例
    git_basics()
    git_commands()
    github_operations()
    branch_management()
    remote_repository_management()
    collaboration_workflow()
    git_best_practices()
    github_cli()
    git_demo()
    create_gitignore()
    
    print("\n=== 学习完成 ===")
    print("今天学习了Git和GitHub的核心概念和实践应用")
    print("掌握了Git常用命令、分支管理和远程仓库操作")
    print("了解了GitHub操作、协作工作流和最佳实践")
    print("可以使用这些知识来有效地管理代码版本，与团队协作开发")
