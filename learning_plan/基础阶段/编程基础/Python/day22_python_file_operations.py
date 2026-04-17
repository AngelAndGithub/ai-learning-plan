#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第22天：Python文件操作和异常处理
Python编程基础学习示例
内容：文件读写、异常处理和上下文管理器
"""

print("=== 第22天：Python文件操作和异常处理 ===")

# 1. 文件操作
print("\n1. 文件操作")

# 创建一个测试文件
with open("test.txt", "w") as f:
    f.write("Hello, World!\n")
    f.write("This is a test file.\n")
    f.write("Python file operations.\n")

print("创建了测试文件 test.txt")

# 读取文件内容
print("\n读取文件内容:")
with open("test.txt", "r") as f:
    content = f.read()
    print(content)

# 逐行读取
print("\n逐行读取:")
with open("test.txt", "r") as f:
    for line in f:
        print(line.strip())

# 读取指定数量的字符
print("\n读取指定数量的字符:")
with open("test.txt", "r") as f:
    first_10_chars = f.read(10)
    print(f"前10个字符: '{first_10_chars}'")
    next_10_chars = f.read(10)
    print(f"接下来10个字符: '{next_10_chars}'")

# 追加内容
print("\n追加内容:")
with open("test.txt", "a") as f:
    f.write("Appended line 1.\n")
    f.write("Appended line 2.\n")

# 验证追加结果
with open("test.txt", "r") as f:
    print(f.read())

# 2. 异常处理
print("\n2. 异常处理")

# 基本异常处理
try:
    with open("nonexistent.txt", "r") as f:
        content = f.read()
except FileNotFoundError:
    print("错误: 文件不存在")

# 多个异常
try:
    # 尝试打开文件
    with open("test.txt", "r") as f:
        content = f.read()
    # 尝试除以零
    result = 10 / 0
except FileNotFoundError:
    print("错误: 文件不存在")
except ZeroDivisionError:
    print("错误: 除数不能为零")
except Exception as e:
    print(f"错误: {e}")

# 异常处理的else和finally
print("\n异常处理的else和finally:")
try:
    with open("test.txt", "r") as f:
        content = f.read()
except FileNotFoundError:
    print("错误: 文件不存在")
else:
    print("文件读取成功")
finally:
    print("无论是否发生异常，都会执行这里")

# 3. 上下文管理器
print("\n3. 上下文管理器")

# 使用with语句（上下文管理器）
print("使用with语句:")
with open("test.txt", "r") as f:
    content = f.read()
    print("文件内容读取完成")
# 这里文件已经自动关闭
print("文件已自动关闭")

# 自定义上下文管理器
print("\n自定义上下文管理器:")

class Timer:
    """计时上下文管理器"""
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        self.end_time = time.time()
        print(f"执行时间: {self.end_time - self.start_time:.4f} 秒")

with Timer() as timer:
    # 模拟耗时操作
    import time
    time.sleep(0.5)
    print("耗时操作完成")

# 4. 文件系统操作
print("\n4. 文件系统操作")

import os

# 获取当前目录
print(f"当前目录: {os.getcwd()}")

# 列出目录内容
print("\n目录内容:")
for item in os.listdir("."):
    print(item)

# 检查文件是否存在
print(f"\ntest.txt 是否存在: {os.path.exists('test.txt')}")
print(f"nonexistent.txt 是否存在: {os.path.exists('nonexistent.txt')}")

# 检查是否为文件
print(f"test.txt 是否为文件: {os.path.isfile('test.txt')}")

# 检查是否为目录
print(f". 是否为目录: {os.path.isdir('.')}")

# 获取文件大小
print(f"test.txt 文件大小: {os.path.getsize('test.txt')} 字节")

# 5. 处理不同文件格式
print("\n5. 处理不同文件格式")

# 处理CSV文件
print("处理CSV文件:")
import csv

# 写入CSV文件
with open("data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Name", "Age", "City"])
    writer.writerow(["Alice", 30, "New York"])
    writer.writerow(["Bob", 25, "London"])
    writer.writerow(["Charlie", 35, "Paris"])

print("创建了CSV文件 data.csv")

# 读取CSV文件
print("\n读取CSV文件:")
with open("data.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)

# 处理JSON文件
print("\n处理JSON文件:")
import json

# 写入JSON文件
data = {
    "name": "Alice",
    "age": 30,
    "city": "New York",
    "hobbies": ["reading", "coding", "traveling"]
}

with open("data.json", "w") as f:
    json.dump(data, f, indent=2)

print("创建了JSON文件 data.json")

# 读取JSON文件
print("\n读取JSON文件:")
with open("data.json", "r") as f:
    loaded_data = json.load(f)
    print(loaded_data)

# 6. 高级文件操作
print("\n6. 高级文件操作")

# 二进制文件操作
print("二进制文件操作:")

# 写入二进制文件
with open("binary.bin", "wb") as f:
    f.write(b"Hello, Binary World!")

print("创建了二进制文件 binary.bin")

# 读取二进制文件
with open("binary.bin", "rb") as f:
    content = f.read()
    print(f"二进制内容: {content}")
    print(f"解码后: {content.decode('utf-8')}")

# 文件位置操作
print("\n文件位置操作:")
with open("test.txt", "r") as f:
    print(f"初始位置: {f.tell()}")
    # 读取5个字符
    content = f.read(5)
    print(f"读取了: '{content}'")
    print(f"当前位置: {f.tell()}")
    # 移动到文件开头
    f.seek(0)
    print(f"移动到开头后的位置: {f.tell()}")
    # 再次读取
    content = f.read(5)
    print(f"再次读取: '{content}'")

# 7. 练习
print("\n7. 练习")

# 练习1: 统计文件中的单词数
def count_words(filename):
    """统计文件中的单词数"""
    try:
        with open(filename, "r") as f:
            content = f.read()
            # 简单的单词统计
            words = content.split()
            return len(words)
    except FileNotFoundError:
        print(f"错误: 文件 {filename} 不存在")
        return 0

word_count = count_words("test.txt")
print(f"test.txt 中的单词数: {word_count}")

# 练习2: 复制文件
def copy_file(source, destination):
    """复制文件"""
    try:
        with open(source, "rb") as src:
            content = src.read()
        with open(destination, "wb") as dst:
            dst.write(content)
        print(f"文件 {source} 已复制到 {destination}")
    except FileNotFoundError:
        print(f"错误: 文件 {source} 不存在")
    except Exception as e:
        print(f"错误: {e}")

copy_file("test.txt", "test_copy.txt")

# 验证复制结果
print("\n复制后的文件内容:")
with open("test_copy.txt", "r") as f:
    print(f.read())

# 练习3: 读取日志文件并统计错误
def count_errors(log_file):
    """统计日志文件中的错误数"""
    try:
        error_count = 0
        with open(log_file, "r") as f:
            for line in f:
                if "ERROR" in line.upper():
                    error_count += 1
        return error_count
    except FileNotFoundError:
        print(f"错误: 文件 {log_file} 不存在")
        return 0

# 创建一个测试日志文件
with open("test.log", "w") as f:
    f.write("INFO: Application started\n")
    f.write("ERROR: Database connection failed\n")
    f.write("INFO: User logged in\n")
    f.write("ERROR: API request failed\n")
    f.write("INFO: Application stopped\n")

error_count = count_errors("test.log")
print(f"test.log 中的错误数: {error_count}")

# 8. 清理临时文件
print("\n8. 清理临时文件")

import os

# 列出所有临时文件
temp_files = ["test.txt", "test_copy.txt", "data.csv", "data.json", "binary.bin", "test.log"]

# 删除临时文件
for file in temp_files:
    if os.path.exists(file):
        os.remove(file)
        print(f"删除文件: {file}")

print("\n=== 第22天学习示例结束 ===")
