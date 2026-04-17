#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第17天：Python基础
Python编程基础学习示例
内容：Python的基本语法、变量、数据类型和操作符
"""

print("=== 第17天：Python基础 ===")

# 1. 基本语法
print("\n1. 基本语法")

# 注释
# 这是单行注释
"""
这是多行注释
可以跨越多行
"""

# 语句和缩进
if True:
    print("这是缩进的代码块")
    print("Python使用缩进来表示代码块")
else:
    print("这是else分支")

# 2. 变量
print("\n2. 变量")

# 变量赋值
x = 10
y = "Hello"
print(f"x = {x}, y = {y}")

# 变量类型
print(f"x的类型: {type(x)}")
print(f"y的类型: {type(y)}")

# 多重赋值
a, b, c = 1, 2, 3
print(f"a = {a}, b = {b}, c = {c}")

# 交换变量值
a, b = b, a
print(f"交换后: a = {a}, b = {b}")

# 3. 数据类型
print("\n3. 数据类型")

# 数字类型
integer = 42
float_num = 3.14
complex_num = 1 + 2j
print(f"整数: {integer}, 类型: {type(integer)}")
print(f"浮点数: {float_num}, 类型: {type(float_num)}")
print(f"复数: {complex_num}, 类型: {type(complex_num)}")

# 字符串
string1 = "Hello"
string2 = 'World'
string3 = """多行
字符串"""
print(f"字符串1: {string1}")
print(f"字符串2: {string2}")
print(f"多行字符串: {string3}")

# 布尔值
true_value = True
false_value = False
print(f"True: {true_value}, 类型: {type(true_value)}")
print(f"False: {false_value}, 类型: {type(false_value)}")

# 空值
none_value = None
print(f"None: {none_value}, 类型: {type(none_value)}")

# 4. 操作符
print("\n4. 操作符")

# 算术操作符
print("算术操作符:")
print(f"10 + 5 = {10 + 5}")
print(f"10 - 5 = {10 - 5}")
print(f"10 * 5 = {10 * 5}")
print(f"10 / 5 = {10 / 5}")
print(f"10 // 3 = {10 // 3} (整除)")
print(f"10 % 3 = {10 % 3} (取模)")
print(f"2 ** 3 = {2 ** 3} (幂)")

# 比较操作符
print("\n比较操作符:")
print(f"10 > 5: {10 > 5}")
print(f"10 < 5: {10 < 5}")
print(f"10 == 5: {10 == 5}")
print(f"10 != 5: {10 != 5}")
print(f"10 >= 5: {10 >= 5}")
print(f"10 <= 5: {10 <= 5}")

# 逻辑操作符
print("\n逻辑操作符:")
print(f"True and False: {True and False}")
print(f"True or False: {True or False}")
print(f"not True: {not True}")

# 成员操作符
print("\n成员操作符:")
list_example = [1, 2, 3, 4, 5]
print(f"3 in {list_example}: {3 in list_example}")
print(f"6 not in {list_example}: {6 not in list_example}")

# 身份操作符
print("\n身份操作符:")
x = 10
y = 10
z = [1, 2, 3]
w = [1, 2, 3]
print(f"x is y: {x is y}")
print(f"z is w: {z is w}")
print(f"z == w: {z == w}")

# 5. 字符串操作
print("\n5. 字符串操作")

# 字符串拼接
str1 = "Hello"
str2 = "World"
print(f"字符串拼接: {str1 + ' ' + str2}")

# 字符串重复
print(f"字符串重复: {'Hello' * 3}")

# 字符串索引和切片
text = "Python"
print(f"字符串: {text}")
print(f"第一个字符: {text[0]}")
print(f"最后一个字符: {text[-1]}")
print(f"切片 [1:4]: {text[1:4]}")
print(f"切片 [:3]: {text[:3]}")
print(f"切片 [2:]: {text[2:]}")

# 字符串方法
print(f"大写: {text.upper()}")
print(f"小写: {text.lower()}")
print(f"首字母大写: {text.capitalize()}")
print(f"替换: {text.replace('P', 'p')}")
print(f"分割: {'a,b,c'.split(',')}")
print(f"连接: '-'.join(['a', 'b', 'c'])")

# 6. 输入输出
print("\n6. 输入输出")

# 输出
print("Hello, World!")
print("The value of x is", x)
print(f"The value of x is {x}")
print("The value of x is {}, y is {}".format(x, y))

# 输入 (注释掉，避免运行时阻塞)
# user_input = input("请输入一个值: ")
# print(f"你输入的值是: {user_input}")

# 7. 类型转换
print("\n7. 类型转换")

# 转换为整数
print(f"int('123'): {int('123')}")
print(f"int(3.9): {int(3.9)}")

# 转换为浮点数
print(f"float('3.14'): {float('3.14')}")
print(f"float(42): {float(42)}")

# 转换为字符串
print(f"str(42): {str(42)}")
print(f"str(3.14): {str(3.14)}")

# 8. 基本数学函数
print("\n8. 基本数学函数")

import math
print(f"math.pi: {math.pi}")
print(f"math.e: {math.e}")
print(f"math.sqrt(16): {math.sqrt(16)}")
print(f"math.pow(2, 3): {math.pow(2, 3)}")
print(f"math.sin(math.pi/2): {math.sin(math.pi/2)}")
print(f"math.cos(math.pi): {math.cos(math.pi)}")
print(f"math.log(10): {math.log(10)}")
print(f"math.log10(100): {math.log10(100)}")

# 9. 随机数
print("\n9. 随机数")

import random
random.seed(42)  # 设置随机种子
print(f"random.random(): {random.random()}")  # 0-1之间的随机数
print(f"random.randint(1, 10): {random.randint(1, 10)}")  # 1-10之间的整数
print(f"random.choice(['a', 'b', 'c']): {random.choice(['a', 'b', 'c'])}")  # 随机选择

# 10. 练习
print("\n10. 练习")

# 练习1: 计算圆的面积
def calculate_circle_area(radius):
    return math.pi * radius ** 2

radius = 5
area = calculate_circle_area(radius)
print(f"半径为 {radius} 的圆的面积: {area:.2f}")

# 练习2: 温度转换 (摄氏度转华氏度)
def celsius_to_fahrenheit(celsius):
    return celsius * 9/5 + 32

celsius = 25
fahrenheit = celsius_to_fahrenheit(celsius)
print(f"{celsius}°C = {fahrenheit}°F")

# 练习3: 字符串反转
def reverse_string(text):
    return text[::-1]

text = "Python"
reversed_text = reverse_string(text)
print(f"原字符串: {text}")
print(f"反转后: {reversed_text}")

print("\n=== 第17天学习示例结束 ===")
