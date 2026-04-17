#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第24天：Python高级特性和最佳实践
Python编程基础学习示例
内容：Python的高级特性、装饰器、生成器、上下文管理器等
"""

print("=== 第24天：Python高级特性和最佳实践 ===")

# 1. 装饰器
print("\n1. 装饰器")

# 基本装饰器
def simple_decorator(func):
    """简单的装饰器"""
    def wrapper(*args, **kwargs):
        print("Before function call")
        result = func(*args, **kwargs)
        print("After function call")
        return result
    return wrapper

@simple_decorator
def greet(name):
    """问候函数"""
    print(f"Hello, {name}!")
    return f"Greeted {name}"

result = greet("Alice")
print(f"Function result: {result}")

# 带参数的装饰器
def repeat(n):
    """重复执行函数n次的装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            results = []
            for i in range(n):
                print(f"Execution {i+1}")
                results.append(func(*args, **kwargs))
            return results
        return wrapper
    return decorator

@repeat(3)
def say_hello(name):
    """说 hello"""
    return f"Hello, {name}!"

results = say_hello("Bob")
print(f"Results: {results}")

# 2. 生成器
print("\n2. 生成器")

# 生成器函数
def fibonacci(n):
    """生成斐波那契数列"""
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

print("前10个斐波那契数:")
for num in fibonacci(10):
    print(num, end=" ")
print()

# 生成器表达式
squares = (x**2 for x in range(10))
print("\n生成器表达式:")
print(f"类型: {type(squares)}")
print("平方数:")
for square in squares:
    print(square, end=" ")
print()

# 3. 上下文管理器
print("\n3. 上下文管理器")

# 使用contextmanager装饰器
from contextlib import contextmanager

@contextmanager
def timer():
    """计时上下文管理器"""
    import time
    start = time.time()
    yield
    end = time.time()
    print(f"执行时间: {end - start:.4f} 秒")

with timer():
    # 模拟耗时操作
    import time
    time.sleep(0.5)
    print("耗时操作完成")

# 4. 闭包
print("\n4. 闭包")

def make_counter():
    """创建计数器"""
    count = 0
    
    def counter():
        nonlocal count
        count += 1
        return count
    
    return counter

counter1 = make_counter()
counter2 = make_counter()

print(f"counter1(): {counter1()}")
print(f"counter1(): {counter1()}")
print(f"counter2(): {counter2()}")
print(f"counter1(): {counter1()}")

# 5. 迭代器
print("\n5. 迭代器")

# 自定义迭代器
class Countdown:
    """倒计时迭代器"""
    
    def __init__(self, start):
        self.start = start
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.start <= 0:
            raise StopIteration
        self.start -= 1
        return self.start + 1

print("倒计时:")
for i in Countdown(5):
    print(i, end=" ")
print()

# 6. 函数式编程
print("\n6. 函数式编程")

# map
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))
print(f"map: {squared}")

# filter
even = list(filter(lambda x: x % 2 == 0, numbers))
print(f"filter: {even}")

# reduce
from functools import reduce
sum_result = reduce(lambda x, y: x + y, numbers)
print(f"reduce: {sum_result}")

# 7. 列表推导式和生成器表达式
print("\n7. 列表推导式和生成器表达式")

# 列表推导式
squares = [x**2 for x in range(10)]
print(f"列表推导式: {squares}")

# 带条件的列表推导式
even_squares = [x**2 for x in range(10) if x % 2 == 0]
print(f"带条件的列表推导式: {even_squares}")

# 嵌套列表推导式
matrix = [[i*j for j in range(3)] for i in range(3)]
print(f"嵌套列表推导式:")
for row in matrix:
    print(row)

# 生成器表达式
squares_gen = (x**2 for x in range(10))
print(f"\n生成器表达式:")
print(f"类型: {type(squares_gen)}")
print("值:")
for square in squares_gen:
    print(square, end=" ")
print()

# 8. 最佳实践
print("\n8. 最佳实践")

# 代码风格
print("代码风格:")
print("- 使用4个空格进行缩进")
print("- 行长度不超过79个字符")
print("- 模块级常量使用全大写")
print("- 函数和变量使用小写字母加下划线")
print("- 类名使用驼峰命名法")

# 异常处理
print("\n异常处理:")
print("- 只捕获特定的异常")
print("- 提供有意义的错误消息")
print("- 使用finally清理资源")

# 性能优化
print("\n性能优化:")
print("- 使用生成器处理大量数据")
print("- 使用局部变量而不是全局变量")
print("- 避免在循环中进行字符串拼接")
print("- 使用列表推导式代替循环")

# 9. 练习
print("\n9. 练习")

# 练习1: 装饰器记录函数执行时间
def timing_decorator(func):
    """记录函数执行时间的装饰器"""
    import time
    
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} 执行时间: {end_time - start_time:.4f} 秒")
        return result
    
    return wrapper

@timing_decorator
def slow_function():
    """模拟耗时函数"""
    import time
    time.sleep(1)
    return "Done"

result = slow_function()
print(f"结果: {result}")

# 练习2: 生成器生成素数
def is_prime(n):
    """判断是否为素数"""
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def prime_generator(n):
    """生成前n个素数"""
    count = 0
    num = 2
    while count < n:
        if is_prime(num):
            yield num
            count += 1
        num += 1

print("\n前10个素数:")
for prime in prime_generator(10):
    print(prime, end=" ")
print()

# 练习3: 上下文管理器管理文件
def read_file_safely(filename):
    """安全读取文件"""
    try:
        with open(filename, "r") as f:
            content = f.read()
        return content
    except FileNotFoundError:
        print(f"错误: 文件 {filename} 不存在")
        return ""
    except Exception as e:
        print(f"错误: {e}")
        return ""

# 创建测试文件
with open("test.txt", "w") as f:
    f.write("Hello, World!")

content = read_file_safely("test.txt")
print(f"\n文件内容: '{content}'")

content = read_file_safely("nonexistent.txt")
print(f"不存在的文件内容: '{content}'")

# 练习4: 函数式编程示例
def process_numbers(numbers):
    """处理数字列表"""
    # 过滤出偶数
    even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
    # 计算平方
    squared = list(map(lambda x: x**2, even_numbers))
    # 计算总和
    total = reduce(lambda x, y: x + y, squared, 0)
    return total

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
result = process_numbers(numbers)
print(f"\n偶数平方和: {result}")

# 10. 清理临时文件
print("\n10. 清理临时文件")

import os

# 列出所有临时文件
temp_files = ["test.txt"]

# 删除临时文件
for file in temp_files:
    if os.path.exists(file):
        os.remove(file)
        print(f"删除文件: {file}")

print("\n=== 第24天学习示例结束 ===")
