---
name: "git-commit"
description: "Git代码提交助手，帮助用户执行Git代码提交相关操作，包括检查状态、添加文件、提交代码、推送远程等。当用户需要管理Git版本控制时调用。"
---

# Git 代码提交助手

## 功能说明

此工具用于帮助用户执行Git代码提交相关操作，包括：
- 检查Git仓库状态
- 添加文件到暂存区
- 提交代码
- 推送代码到远程仓库
- 管理分支
- 处理合并冲突

## 使用场景

当用户需要：
- 提交代码到Git仓库
- 推送代码到远程仓库（如GitHub、GitLab等）
- 检查代码变更状态
- 管理Git分支
- 解决代码合并冲突

## 操作流程

### 1. 检查仓库状态

```bash
git status
```

### 2. 添加文件到暂存区

```bash
# 添加单个文件
git add <file_path>

# 添加所有文件
git add .

# 添加特定类型的文件
git add *.py
```

### 3. 提交代码

```bash
git commit -m "提交信息"

# 提交并添加修改的文件
git commit -am "提交信息"
```

### 4. 推送代码到远程仓库

```bash
# 推送当前分支到默认远程仓库
git push

# 推送指定分支到指定远程仓库
git push <remote> <branch>

# 首次推送设置上游分支
git push -u <remote> <branch>
```

### 5. 分支管理

```bash
# 查看分支
git branch

# 创建分支
git branch <branch_name>

# 切换分支
git checkout <branch_name>

# 创建并切换分支
git checkout -b <branch_name>

# 合并分支
git merge <branch_name>

# 删除分支
git branch -d <branch_name>
```

### 6. 远程仓库管理

```bash
# 查看远程仓库
git remote -v

# 添加远程仓库
git remote add <name> <url>

# 拉取远程代码
git pull

# 克隆远程仓库
git clone <url>
```

## 最佳实践

1. **提交信息规范**：使用清晰、简洁的提交信息，描述代码变更的内容和原因
2. **小而频繁的提交**：每次提交只包含相关的变更，便于代码审查和回滚
3. **分支命名规范**：使用有意义的分支名称，如 `feature/xxx`、`bugfix/xxx` 等
4. **定期拉取**：保持本地代码与远程仓库同步，减少合并冲突
5. **使用 .gitignore**：忽略不需要版本控制的文件，如日志、依赖包等

## 常见问题处理

### 1. 合并冲突

当出现合并冲突时，需要手动解决冲突，然后重新提交：

```bash
# 解决冲突后
git add .
git commit -m "Resolve merge conflict"
```

### 2. 撤销提交

```bash
# 撤销最后一次提交（保留更改）
git reset HEAD~1

# 撤销最后一次提交（丢弃更改）
git reset --hard HEAD~1
```

### 3. 强制推送

```bash
# 强制推送（谨慎使用）
git push -f <remote> <branch>
```

## 示例使用

### 示例1：基本提交流程

```bash
# 检查状态
git status

# 添加所有文件
git add .

# 提交代码
git commit -m "feat: 添加新功能"

# 推送到远程
git push
```

### 示例2：分支操作

```bash
# 创建并切换到新分支
git checkout -b feature/new-feature

# 进行开发并提交
git add .
git commit -m "feat: 实现新功能"

# 推送到远程
git push -u origin feature/new-feature

# 切换回主分支
git checkout main

# 拉取最新代码
git pull

# 合并新功能分支
git merge feature/new-feature

# 推送到远程
git push
```

## 注意事项

### 1. 安全注意事项
- **避免提交敏感信息**：不要提交密码、API密钥、数据库连接字符串等敏感信息
- **使用 .gitignore**：忽略包含敏感信息的文件
- **定期检查**：定期检查仓库中是否存在敏感信息
- **使用环境变量**：将敏感配置放在环境变量中，而不是硬编码

### 2. 性能注意事项
- **避免提交大文件**：不要提交超过100MB的文件
- **使用Git LFS**：对于大文件，使用Git Large File Storage (LFS)
- **定期清理**：定期清理仓库历史，减少仓库大小
- **合理使用分支**：避免创建过多分支，定期清理不需要的分支

### 3. 协作注意事项
- **遵循团队规范**：严格遵循团队的Git工作流程和命名规范
- **及时沟通**：在进行重要操作前与团队成员沟通
- **代码审查**：通过Pull Request进行代码审查
- **保持同步**：定期拉取远程代码，保持本地代码与团队同步

### 4. 操作注意事项
- **提交前测试**：确保代码通过测试后再提交
- **检查代码质量**：使用代码质量工具检查代码
- **提交信息清晰**：使用规范化的提交信息
- **避免强制推送**：除非必要，否则不要使用强制推送

### 5. 故障处理
- **定期备份**：定期备份Git仓库
- **灾难恢复**：了解如何从备份中恢复仓库
- **错误处理**：遇到错误时，先了解原因再进行操作
- **寻求帮助**：遇到问题时，及时寻求团队成员或社区的帮助

### 6. 工具使用
- **使用Git客户端**：对于复杂操作，使用图形化Git客户端
- **配置Git**：合理配置Git参数，如用户名、邮箱、编辑器等
- **使用Git hooks**：利用Git hooks自动化一些操作
- **集成CI/CD**：将Git操作与CI/CD流程集成