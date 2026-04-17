#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第20天：Python函数和模块
Python编程基础学习示例
内容：函数定义、参数、返回值和模块导入
"""

print("=== 第20天：Python函数和模块 ===")

# 1. 函数定义
print("\n1. 函数定义")

# 基本函数定义
def greet():
    """打印问候信息"""
    print("Hello, World!")

# 调用函数
greet()

# 带参数的函数
def greet_name(name):
    """向指定名称的人问候"""
    print(f"Hello, {name}!")

greet_name("Alice")

# 带默认参数的函数
def greet_with_default(name="World"):
    """带默认参数的问候函数"""
    print(f"Hello, {name}!")

greet_with_default()
greet_with_default("Bob")

# 2. 函数参数
print("\n2. 函数参数")

# 位置参数
def add(a, b):
    """计算两个数的和"""
    return a + b

result = add(3, 5)
print(f"3 + 5 = {result}")

# 关键字参数
def describe_person(name, age, city):
    """描述一个人"""
    print(f"Name: {name}, Age: {age}, City: {city}")

describe_person(name="Alice", age=30, city="New York")
describe_person(city="London", name="Bob", age=25)  # 可以改变参数顺序

# 可变长度参数
# *args 用于接收任意数量的位置参数
def sum_numbers(*args):
    """计算任意数量数字的和"""
    return sum(args)

print(f"sum_numbers(1, 2, 3, 4) = {sum_numbers(1, 2, 3, 4)}")
print(f"sum_numbers(10, 20) = {sum_numbers(10, 20)}")

# **kwargs 用于接收任意数量的关键字参数
def print_info(**kwargs):
    """打印任意数量的关键字参数"""
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_info(name="Alice", age=30, city="New York")

# 混合使用参数类型
def mixed_params(a, b, *args, c=10, **kwargs):
    """混合使用不同类型的参数"""
    print(f"a: {a}, b: {b}")
    print(f"args: {args}")
    print(f"c: {c}")
    print(f"kwargs: {kwargs}")

mixed_params(1, 2, 3, 4, 5, c=20, name="Alice", age=30)

# 3. 函数返回值
print("\n3. 函数返回值")

# 返回单个值
def square(x):
    """计算平方"""
    return x ** 2

print(f"square(5) = {square(5)}")

# 返回多个值
def calculate(a, b):
    """计算和与差"""
    return a + b, a - b

add_result, subtract_result = calculate(10, 3)
print(f"10 + 3 = {add_result}, 10 - 3 = {subtract_result}")

# 返回None
def no_return():
    """不返回任何值"""
    print("This function returns nothing")

result = no_return()
print(f"no_return() 返回: {result}")

# 4. 作用域
print("\n4. 作用域")

# 全局变量
x = 10

def print_global():
    """打印全局变量"""
    print(f"全局变量 x = {x}")

print_global()

# 局部变量
def print_local():
    """打印局部变量"""
    y = 20
    print(f"局部变量 y = {y}")

print_local()
# print(y)  # 这会引发错误，因为y是局部变量

# 修改全局变量
def modify_global():
    """修改全局变量"""
    global x
    x = 30
    print(f"修改后的全局变量 x = {x}")

modify_global()
print(f"全局变量 x = {x}")

# 5. 模块
print("\n5. 模块")

# 导入整个模块
import math
print(f"math.pi = {math.pi}")
print(f"math.sqrt(16) = {math.sqrt(16)}")

# 导入模块中的特定函数
from math import sin, cos
print(f"sin(0) = {sin(0)}")
print(f"cos(0) = {cos(0)}")

# 导入模块并重命名
import math as m
print(f"m.pi = {m.pi}")

# 导入模块中的所有函数
from math import *
print(f"tan(0) = {tan(0)}")

# 6. 包
print("\n6. 包")

# 导入包
# 注意：这里只是演示导入包的语法，实际运行时可能需要安装相应的包
"""
import numpy as np
print(f"numpy版本: {np.__version__}")

import pandas as pd
print(f"pandas版本: {pd.__version__}")
"""
print("包导入语法示例:")
print("import numpy as np")
print("import pandas as pd")

# 7. 内置函数
print("\n7. 内置函数")

# 常用内置函数
print(f"len([1, 2, 3, 4, 5]) = {len([1, 2, 3, 4, 5])}")
print(f"max([1, 5, 3, 9, 2]) = {max([1, 5, 3, 9, 2])}")
print(f"min([1, 5, 3, 9, 2]) = {min([1, 5, 3, 9, 2])}")
print(f"sum([1, 2, 3, 4, 5]) = {sum([1, 2, 3, 4, 5])}")
print(f"sorted([5, 2, 8, 1, 3]) = {sorted([5, 2, 8, 1, 3])}")
print(f"list('Hello') = {list('Hello')}")
print(f"tuple([1, 2, 3]) = {tuple([1, 2, 3])}")
print(f"set([1, 2, 2, 3, 3, 3]) = {set([1, 2, 2, 3, 3, 3])}")
print(f"dict(a=1, b=2) = {dict(a=1, b=2)}")
print(f"abs(-10) = {abs(-10)}")
print(f"round(3.14159, 2) = {round(3.14159, 2)}")
print(f"int('123') = {int('123')}")
print(f"float('3.14') = {float('3.14')}")
print(f"str(123) = {str(123)}")

# 8. 递归函数
print("\n8. 递归函数")

# 计算阶乘
def factorial(n):
    """递归计算阶乘"""
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)

print(f"factorial(5) = {factorial(5)}")

# 斐波那契数列
def fibonacci(n):
    """递归计算斐波那契数"""
    if n <= 1:
        return n
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)

print("前10个斐波那契数:")
for i in range(10):
    print(fibonacci(i), end=" ")
print()

# 9. 练习
print("\n9. 练习")

# 练习1: 计算列表的平均值
def calculate_average(lst):
    """计算列表的平均值"""
    if not lst:
        return 0
    return sum(lst) / len(lst)

numbers = [1, 2, 3, 4, 5]
average = calculate_average(numbers)
print(f"列表 {numbers} 的平均值: {average}")

# 练习2: 检查字符串是否为回文
def is_palindrome(s):
    """检查字符串是否为回文"""
    s = s.lower().replace(" ", "")
    return s == s[::-1]

print(f"'racecar' 是回文: {is_palindrome('racecar')}")
print(f"'hello' 是回文: {is_palindrome('hello')}")
print(f"'A man a plan a canal Panama' 是回文: {is_palindrome('A man a plan a canal Panama')}")

# 练习3: 生成指定长度的随机密码
def generate_password(length=8):
    """生成指定长度的随机密码"""
    import random
    import string
    
    characters = string.ascii_letters + string.digits + string.punctuation
    password = ''.join(random.choice(characters) for _ in range(length))
    return password

print(f"生成的8位密码: {generate_password()}")
print(f"生成的12位密码: {generate_password(12)}")

# 练习4: 计算两个数的最大公约数
def gcd(a, b):
    """计算两个数的最大公约数"""
    while b:
        a, b = b, a % b
    return a

print(f"12和18的最大公约数: {gcd(12, 18)}")
print(f"25和35的最大公约数: {gcd(25, 35)}")

print("\n=== 第20天学习示例结束 ===")
