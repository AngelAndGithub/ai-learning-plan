#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python编程基础示例代码
涵盖Python基础语法、数据结构、函数、模块与包、文件操作等内容
"""

# 1. 基础语法
print("=== 1. 基础语法 ===")

# 变量与数据类型
x = 10  # 整数
print(f"整数: {x}, 类型: {type(x)}")

y = 3.14  # 浮点数
print(f"浮点数: {y}, 类型: {type(y)}")

z = "Hello, Python!"  # 字符串
print(f"字符串: {z}, 类型: {type(z)}")

is_true = True  # 布尔值
print(f"布尔值: {is_true}, 类型: {type(is_true)}")

# 控制流：条件语句
print("\n=== 条件语句 ===")
a = 10
b = 20
if a > b:
    print(f"{a} 大于 {b}")
elif a < b:
    print(f"{a} 小于 {b}")
else:
    print(f"{a} 等于 {b}")

# 控制流：循环语句
print("\n=== 循环语句 ===")

# for循环
print("For循环:")
for i in range(5):
    print(f"i = {i}")

# while循环
print("\nWhile循环:")
count = 0
while count < 5:
    print(f"count = {count}")
    count += 1

# 2. 数据结构
print("\n=== 2. 数据结构 ===")

# 列表
print("\n列表:")
my_list = [1, 2, 3, 4, 5]
print(f"原始列表: {my_list}")
my_list.append(6)  # 添加元素
print(f"添加元素后: {my_list}")
my_list.remove(3)  # 删除元素
print(f"删除元素后: {my_list}")
my_list.sort(reverse=True)  # 排序
print(f"排序后: {my_list}")

# 元组
print("\n元组:")
my_tuple = (1, 2, 3, 4, 5)
print(f"元组: {my_tuple}")
print(f"元组长度: {len(my_tuple)}")
print(f"第二个元素: {my_tuple[1]}")

# 字典
print("\n字典:")
my_dict = {"name": "Alice", "age": 25, "city": "New York"}
print(f"原始字典: {my_dict}")
print(f"姓名: {my_dict['name']}")
my_dict["age"] = 26  # 修改值
print(f"修改后: {my_dict}")
my_dict["country"] = "USA"  # 添加键值对
print(f"添加后: {my_dict}")

# 集合
print("\n集合:")
my_set = {1, 2, 3, 4, 5}
print(f"原始集合: {my_set}")
my_set.add(6)  # 添加元素
print(f"添加元素后: {my_set}")
my_set.remove(3)  # 删除元素
print(f"删除元素后: {my_set}")

# 3. 函数
print("\n=== 3. 函数 ===")

def greet(name):
    """问候函数"""
    return f"Hello, {name}!"

print(greet("Bob"))

def calculate_area(length, width):
    """计算面积"""
    return length * width

print(f"面积: {calculate_area(10, 5)}")

# 带默认参数的函数
def calculate_volume(length, width, height=1):
    """计算体积"""
    return length * width * height

print(f"体积: {calculate_volume(10, 5)}")
print(f"体积: {calculate_volume(10, 5, 2)}")

# 4. 模块与包
print("\n=== 4. 模块与包 ===")

# 导入模块
import math
print(f"π的值: {math.pi}")
print(f"平方根: {math.sqrt(16)}")

# 导入模块的特定函数
from math import sin, cos
print(f"sin(π/2): {sin(math.pi/2)}")
print(f"cos(π): {cos(math.pi)}")

# 5. 文件操作
print("\n=== 5. 文件操作 ===")

# 写入文件
with open("example.txt", "w", encoding="utf-8") as f:
    f.write("Hello, Python!\n")
    f.write("This is a test file.\n")
print("文件写入完成")

# 读取文件
with open("example.txt", "r", encoding="utf-8") as f:
    content = f.read()
print("文件内容:")
print(content)

# 6. 异常处理
print("\n=== 6. 异常处理 ===")

try:
    result = 10 / 0
except ZeroDivisionError as e:
    print(f"错误: {e}")
finally:
    print("无论是否有异常，都会执行这里")

try:
    with open("non_existent_file.txt", "r") as f:
        content = f.read()
except FileNotFoundError as e:
    print(f"错误: {e}")

# 7. 面向对象编程
print("\n=== 7. 面向对象编程 ===")

class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def greet(self):
        return f"Hello, my name is {self.name}, I'm {self.age} years old."
    
    def celebrate_birthday(self):
        self.age += 1
        return f"Happy birthday! Now I'm {self.age} years old."

# 创建对象
person = Person("Alice", 25)
print(person.greet())
print(person.celebrate_birthday())

# 继承
class Student(Person):
    def __init__(self, name, age, student_id):
        super().__init__(name, age)
        self.student_id = student_id
    
    def study(self, subject):
        return f"{self.name} is studying {subject}."

student = Student("Bob", 20, "S12345")
print(student.greet())
print(student.study("Mathematics"))

# 8. 列表推导式
print("\n=== 8. 列表推导式 ===")

# 生成1-10的平方
squares = [x**2 for x in range(1, 11)]
print(f"平方列表: {squares}")

# 生成偶数的平方
even_squares = [x**2 for x in range(1, 11) if x % 2 == 0]
print(f"偶数平方列表: {even_squares}")

# 9. 生成器
print("\n=== 9. 生成器 ===")

def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

print("斐波那契数列:")
for num in fibonacci(10):
    print(num, end=" ")
print()

# 10. 装饰器
print("\n=== 10. 装饰器 ===")

def timer(func):
    import time
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"函数 {func.__name__} 执行时间: {end_time - start_time:.4f} 秒")
        return result
    return wrapper

@timer
def slow_function():
    import time
    time.sleep(1)
    return "函数执行完成"

print(slow_function())

print("\n=== Python编程基础示例代码结束 ===")
