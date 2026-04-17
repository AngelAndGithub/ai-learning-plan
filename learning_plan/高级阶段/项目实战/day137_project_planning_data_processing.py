#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第137天：项目规划和数据处理
项目实战学习示例
内容：项目规划的基本步骤、数据收集、数据清洗和数据预处理
"""

print("=== 第137天：项目规划和数据处理 ===")

# 1. 项目规划
print("\n1. 项目规划")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

print("项目规划是项目成功的关键")
print("- 项目目标：明确项目的目的和预期结果")
print("- 项目范围：确定项目的边界和包含的内容")
print("- 项目时间线：制定项目的时间计划")
print("- 项目资源：确定所需的资源")
print("- 项目风险：识别可能的风险和应对策略")

# 2. 数据收集
print("\n2. 数据收集")

print("数据收集是项目的基础")
print("- 数据源：公开数据集、内部数据、爬虫获取")
print("- 数据格式：CSV、JSON、数据库")
print("- 数据质量：完整性、准确性、一致性")

# 示例：加载数据集
print("\n示例：加载数据集")

# 生成示例数据
data = {
    'id': range(1, 101),
    'age': np.random.randint(18, 65, 100),
    'gender': np.random.choice(['Male', 'Female'], 100),
    'income': np.random.randint(30000, 100000, 100),
    'expenses': np.random.randint(10000, 80000, 100),
    'credit_score': np.random.randint(300, 850, 100),
    'default': np.random.choice([0, 1], 100, p=[0.8, 0.2])
}

df = pd.DataFrame(data)
print("数据集基本信息:")
print(df.info())
print("\n数据集前5行:")
print(df.head())

# 3. 数据清洗
print("\n3. 数据清洗")

print("数据清洗是确保数据质量的重要步骤")
print("- 缺失值处理")
print("- 异常值处理")
print("- 重复值处理")
print("- 数据类型转换")

# 示例：数据清洗
print("\n示例：数据清洗")

# 添加缺失值
df.loc[0, 'age'] = np.nan
df.loc[1, 'income'] = np.nan

print("数据集缺失值情况:")
print(df.isnull().sum())

# 处理缺失值
df['age'].fillna(df['age'].mean(), inplace=True)
df['income'].fillna(df['income'].median(), inplace=True)

print("\n处理后缺失值情况:")
print(df.isnull().sum())

# 处理重复值
df = df.drop_duplicates()
print(f"\n去重后数据行数: {len(df)}")

# 处理异常值
print("\n处理异常值:")
print(f"原始数据收入最大值: {df['income'].max()}")
print(f"原始数据收入最小值: {df['income'].min()}")

# 移除收入异常值
Q1 = df['income'].quantile(0.25)
Q3 = df['income'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['income'] >= Q1 - 1.5 * IQR) & (df['income'] <= Q3 + 1.5 * IQR)]

print(f"处理后数据行数: {len(df)}")
print(f"处理后收入最大值: {df['income'].max()}")
print(f"处理后收入最小值: {df['income'].min()}")

# 4. 数据预处理
print("\n4. 数据预处理")

print("数据预处理是为模型训练做准备")
print("- 特征工程")
print("- 特征选择")
print("- 特征缩放")
print("- 编码分类变量")

# 示例：数据预处理
print("\n示例：数据预处理")

# 特征工程
df['savings'] = df['income'] - df['expenses']
df['debt_ratio'] = df['expenses'] / df['income']

print("\n添加新特征后的数据:")
print(df.head())

# 编码分类变量
le = LabelEncoder()
df['gender_encoded'] = le.fit_transform(df['gender'])

print("\n编码分类变量后的数据:")
print(df[['gender', 'gender_encoded']].head())

# 特征缩放
scaler = StandardScaler()
df[['age', 'income', 'expenses', 'credit_score', 'savings', 'debt_ratio']] = scaler.fit_transform(
    df[['age', 'income', 'expenses', 'credit_score', 'savings', 'debt_ratio']]
)

print("\n特征缩放后的数据:")
print(df.head())

# 5. 数据可视化
print("\n5. 数据可视化")

print("数据可视化有助于理解数据")
print("- 直方图：查看数据分布")
print("- 散点图：查看变量关系")
print("- 箱线图：查看数据分布和异常值")
print("- 热力图：查看变量相关性")

# 示例：数据可视化
print("\n示例：数据可视化")

# 直方图
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
sns.histplot(df['age'], kde=True)
plt.title('Age Distribution')

plt.subplot(2, 2, 2)
sns.histplot(df['income'], kde=True)
plt.title('Income Distribution')

plt.subplot(2, 2, 3)
sns.histplot(df['credit_score'], kde=True)
plt.title('Credit Score Distribution')

plt.subplot(2, 2, 4)
sns.histplot(df['default'], kde=True)
plt.title('Default Distribution')

plt.tight_layout()
plt.show()

# 散点图
plt.figure(figsize=(10, 6))
sns.scatterplot(x='income', y='expenses', hue='default', data=df)
plt.title('Income vs Expenses')
plt.show()

# 箱线图
plt.figure(figsize=(10, 6))
sns.boxplot(x='default', y='credit_score', data=df)
plt.title('Credit Score by Default Status')
plt.show()

# 热力图
plt.figure(figsize=(10, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# 6. 数据划分
print("\n6. 数据划分")

print("数据划分是模型训练的重要步骤")
print("- 训练集：用于模型训练")
print("- 验证集：用于模型调优")
print("- 测试集：用于模型评估")

# 示例：数据划分
print("\n示例：数据划分")

X = df.drop(['id', 'gender', 'default'], axis=1)
y = df['default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)  # 25% of 80% = 20% of total

print(f"训练集大小: {len(X_train)}")
print(f"验证集大小: {len(X_val)}")
print(f"测试集大小: {len(X_test)}")

# 7. 项目文档
print("\n7. 项目文档")

print("项目文档是项目的重要组成部分")
print("- 项目计划")
print("- 数据文档")
print("- 模型文档")
print("- 部署文档")
print("- 用户文档")

# 8. 项目管理
print("\n8. 项目管理")

print("项目管理是确保项目成功的关键")
print("- 敏捷开发")
print("- Scrum")
print("- Kanban")
print("- 项目跟踪工具：JIRA、Trello、Asana")

# 9. 练习
print("\n9. 练习")

# 练习1: 项目规划
print("练习1: 项目规划")
print("- 确定一个项目目标")
print("- 制定项目计划")
print("- 识别项目风险")

# 练习2: 数据收集和清洗
print("\n练习2: 数据收集和清洗")
print("- 收集一个公开数据集")
print("- 清洗数据")
print("- 预处理数据")

# 练习3: 数据可视化
print("\n练习3: 数据可视化")
print("- 制作不同类型的可视化图表")
print("- 分析数据分布")
print("- 发现数据中的模式")

# 练习4: 项目文档
print("\n练习4: 项目文档")
print("- 编写项目计划")
print("- 编写数据文档")
print("- 编写模型文档")

print("\n=== 第137天学习示例结束 ===")
