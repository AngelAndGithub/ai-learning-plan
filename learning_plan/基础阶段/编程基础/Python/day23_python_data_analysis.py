#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第23天：Python数据分析库
Python编程基础学习示例
内容：NumPy、Pandas和Matplotlib等数据分析库的使用
"""

print("=== 第23天：Python数据分析库 ===")

# 1. NumPy
print("\n1. NumPy")

import numpy as np

# 创建数组
print("创建数组:")

# 从列表创建
arr1 = np.array([1, 2, 3, 4, 5])
print(f"从列表创建: {arr1}")
print(f"形状: {arr1.shape}")
print(f"类型: {arr1.dtype}")

# 从元组创建
arr2 = np.array((6, 7, 8, 9, 10))
print(f"\n从元组创建: {arr2}")

# 创建多维数组
arr3 = np.array([[1, 2, 3], [4, 5, 6]])
print(f"\n二维数组:")
print(arr3)
print(f"形状: {arr3.shape}")

# 创建特殊数组
print("\n创建特殊数组:")

# 全零数组
zeros = np.zeros((2, 3))
print(f"全零数组:")
print(zeros)

# 全一数组
ones = np.ones((3, 2))
print(f"\n全一数组:")
print(ones)

# 单位矩阵
identity = np.eye(3)
print(f"\n单位矩阵:")
print(identity)

# 等差数列
arange = np.arange(0, 10, 2)
print(f"\n等差数列: {arange}")

# 等间隔数组
linspace = np.linspace(0, 1, 5)
print(f"等间隔数组: {linspace}")

# 2. NumPy数组操作
print("\n2. NumPy数组操作")

# 基本运算
print("基本运算:")

arr = np.array([1, 2, 3, 4, 5])
print(f"原数组: {arr}")
print(f"数组 + 2: {arr + 2}")
print(f"数组 * 2: {arr * 2}")
print(f"数组 ** 2: {arr ** 2}")

# 数组间运算
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
print(f"\n数组1: {arr1}")
print(f"数组2: {arr2}")
print(f"数组1 + 数组2: {arr1 + arr2}")
print(f"数组1 * 数组2: {arr1 * arr2}")

# 矩阵乘法
matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])
print(f"\n矩阵1:")
print(matrix1)
print(f"矩阵2:")
print(matrix2)
print(f"矩阵乘法:")
print(np.dot(matrix1, matrix2))
print(f"矩阵乘法 (使用 @ 运算符):")
print(matrix1 @ matrix2)

# 索引和切片
print("\n索引和切片:")

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
print(f"原数组: {arr}")
print(f"索引 2: {arr[2]}")
print(f"切片 2:5: {arr[2:5]}")
print(f"切片 :3: {arr[:3]}")
print(f"切片 6:: {arr[6:]}")
print(f"步长 2: {arr[::2]}")

# 二维数组索引
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"\n二维数组:")
print(matrix)
print(f"索引 (0, 1): {matrix[0, 1]}")
print(f"切片 [0:2, 1:3]:")
print(matrix[0:2, 1:3])

# 3. NumPy数学函数
print("\n3. NumPy数学函数")

arr = np.array([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
print(f"角度数组 (弧度): {arr}")
print(f"sin: {np.sin(arr)}")
print(f"cos: {np.cos(arr)}")
print(f"tan: {np.tan(arr)}")

# 统计函数
arr = np.array([1, 2, 3, 4, 5])
print(f"\n数组: {arr}")
print(f"平均值: {np.mean(arr)}")
print(f"总和: {np.sum(arr)}")
print(f"最大值: {np.max(arr)}")
print(f"最小值: {np.min(arr)}")
print(f"标准差: {np.std(arr)}")
print(f"方差: {np.var(arr)}")

# 4. Pandas
print("\n4. Pandas")

import pandas as pd

# 创建Series
print("创建Series:")
data = [1, 2, 3, 4, 5]
index = ['a', 'b', 'c', 'd', 'e']
series = pd.Series(data, index=index)
print(series)

# 创建DataFrame
print("\n创建DataFrame:")
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'age': [25, 30, 35, 40, 45],
    'city': ['New York', 'London', 'Paris', 'Tokyo', 'Sydney']
}
df = pd.DataFrame(data)
print(df)

# 查看DataFrame信息
print("\nDataFrame信息:")
print(df.info())
print(f"\nDataFrame形状: {df.shape}")
print(f"\nDataFrame列: {list(df.columns)}")
print(f"\nDataFrame索引: {list(df.index)}")

# 访问数据
print("\n访问数据:")
print(f"访问'name'列:")
print(df['name'])
print(f"\n访问前3行:")
print(df.head(3))
print(f"\n访问后2行:")
print(df.tail(2))
print(f"\n访问特定行 (iloc):")
print(df.iloc[1:3])
print(f"\n访问特定行 (loc):")
print(df.loc[1:3, ['name', 'age']])

# 数据过滤
print("\n数据过滤:")
filtered = df[df['age'] > 30]
print(filtered)

# 数据排序
print("\n数据排序:")
sorted_df = df.sort_values('age', ascending=False)
print(sorted_df)

# 数据聚合
print("\n数据聚合:")
print(f"年龄平均值: {df['age'].mean()}")
print(f"年龄最大值: {df['age'].max()}")
print(f"年龄最小值: {df['age'].min()}")

# 5. Matplotlib
print("\n5. Matplotlib")

import matplotlib.pyplot as plt

# 基本折线图
print("基本折线图:")
x = np.linspace(0, 10, 100)
y = np.sin(x)

print(f"x 形状: {x.shape}")
print(f"y 形状: {y.shape}")
print(f"前5个x值: {x[:5]}")
print(f"前5个y值: {y[:5]}")

# 基本散点图
print("\n基本散点图:")
x = np.random.rand(50)
y = np.random.rand(50)

print(f"x 形状: {x.shape}")
print(f"y 形状: {y.shape}")
print(f"前5个x值: {x[:5]}")
print(f"前5个y值: {y[:5]}")

# 基本柱状图
print("\n基本柱状图:")
categories = ['A', 'B', 'C', 'D', 'E']
values = [10, 20, 15, 25, 30]

print(f"类别: {categories}")
print(f"值: {values}")

# 6. 综合应用
print("\n6. 综合应用")

# 创建模拟数据
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=30)
prices = np.random.randn(30) + 100
prices = np.cumsum(prices)
volumes = np.random.randint(1000, 5000, 30)

# 创建DataFrame
data = {
    'date': dates,
    'price': prices,
    'volume': volumes
}
df = pd.DataFrame(data)
df.set_index('date', inplace=True)

print("\n股票数据:")
print(df.head())

# 计算移动平均线
df['MA5'] = df['price'].rolling(window=5).mean()
df['MA10'] = df['price'].rolling(window=10).mean()

print("\n添加移动平均线后:")
print(df.head(15))

# 计算收益率
df['return'] = df['price'].pct_change() * 100

print("\n添加收益率后:")
print(df.head())

# 7. 练习
print("\n7. 练习")

# 练习1: 生成正态分布数据并分析
print("练习1: 生成正态分布数据并分析")

# 生成正态分布数据
np.random.seed(42)
data = np.random.normal(loc=100, scale=10, size=1000)

print(f"数据形状: {data.shape}")
print(f"均值: {np.mean(data):.2f}")
print(f"标准差: {np.std(data):.2f}")
print(f"最小值: {np.min(data):.2f}")
print(f"最大值: {np.max(data):.2f}")
print(f"中位数: {np.median(data):.2f}")

# 练习2: 分析CSV数据
print("\n练习2: 分析CSV数据")

# 创建测试CSV文件
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'math': [85, 90, 75, 95, 80],
    'english': [90, 85, 80, 75, 95],
    'science': [80, 75, 90, 85, 95]
}
df = pd.DataFrame(data)
df.to_csv('students.csv', index=False)

print("创建了students.csv文件")

# 读取CSV文件
students_df = pd.read_csv('students.csv')
print("\n读取CSV文件:")
print(students_df)

# 计算每个学生的平均分
students_df['average'] = students_df[['math', 'english', 'science']].mean(axis=1)
print("\n添加平均分后:")
print(students_df)

# 按平均分排序
top_students = students_df.sort_values('average', ascending=False)
print("\n按平均分排序:")
print(top_students)

# 计算每门课程的平均分
subject_averages = students_df[['math', 'english', 'science']].mean()
print("\n每门课程的平均分:")
print(subject_averages)

# 8. 清理临时文件
print("\n8. 清理临时文件")

import os

# 列出所有临时文件
temp_files = ["students.csv"]

# 删除临时文件
for file in temp_files:
    if os.path.exists(file):
        os.remove(file)
        print(f"删除文件: {file}")

print("\n=== 第23天学习示例结束 ===")
