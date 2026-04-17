#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第19天：Python控制流
Python编程基础学习示例
内容：条件语句、循环语句和控制流工具
"""

print("=== 第19天：Python控制流 ===")

# 1. 条件语句
print("\n1. 条件语句")

# if 语句
x = 10
if x > 5:
    print("x 大于 5")

# if-else 语句
x = 3
if x > 5:
    print("x 大于 5")
else:
    print("x 小于或等于 5")

# if-elif-else 语句
x = 7
if x > 10:
    print("x 大于 10")
elif x > 5:
    print("x 大于 5 但小于或等于 10")
else:
    print("x 小于或等于 5")

# 嵌套条件语句
x = 8
y = 4
if x > 5:
    if y > 5:
        print("x 和 y 都大于 5")
    else:
        print("x 大于 5，但 y 小于或等于 5")
else:
    print("x 小于或等于 5")

# 2. 循环语句
print("\n2. 循环语句")

# for 循环
print("for 循环:")
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(f"水果: {fruit}")

# 使用 range()
print("\n使用 range():")
for i in range(5):
    print(f"数字: {i}")

for i in range(2, 10, 2):
    print(f"偶数: {i}")

# 遍历字典
print("\n遍历字典:")
person = {"name": "Alice", "age": 30, "city": "New York"}
for key, value in person.items():
    print(f"{key}: {value}")

# while 循环
print("\nwhile 循环:")
i = 0
while i < 5:
    print(f"计数: {i}")
    i += 1

# 3. 控制流工具
print("\n3. 控制流工具")

# break 语句
print("break 语句:")
for i in range(10):
    if i == 5:
        break
    print(f"数字: {i}")

# continue 语句
print("\ncontinue 语句:")
for i in range(10):
    if i % 2 == 0:
        continue
    print(f"奇数: {i}")

# pass 语句
print("\npass 语句:")
for i in range(5):
    if i == 2:
        pass  # 占位符，什么都不做
    print(f"数字: {i}")

# 4. 列表推导式中的条件
print("\n4. 列表推导式中的条件")

# 基本条件
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even_numbers = [x for x in numbers if x % 2 == 0]
print(f"偶数: {even_numbers}")

# 多个条件
between_3_and_8 = [x for x in numbers if x > 3 and x < 8]
print(f"3到8之间的数字: {between_3_and_8}")

# if-else 在列表推导式中
transformed = [x if x % 2 == 0 else x*2 for x in numbers]
print(f"转换后的列表: {transformed}")

# 5. 异常处理
print("\n5. 异常处理")

# try-except 语句
try:
    result = 10 / 0
except ZeroDivisionError:
    print("错误: 除数不能为零")

try:
    value = int("abc")
except ValueError:
    print("错误: 无法将字符串转换为整数")

# 多个异常
try:
    value = int("abc")
    result = 10 / 0
except ValueError:
    print("错误: 无法将字符串转换为整数")
except ZeroDivisionError:
    print("错误: 除数不能为零")

# 捕获所有异常
try:
    value = int("abc")
except Exception as e:
    print(f"错误: {e}")

# try-except-else-finally
try:
    result = 10 / 2
except ZeroDivisionError:
    print("错误: 除数不能为零")
else:
    print(f"计算结果: {result}")
finally:
    print("无论如何都会执行的代码")

# 6. 断言
print("\n6. 断言")

# assert 语句
def divide(a, b):
    assert b != 0, "除数不能为零"
    return a / b

try:
    result = divide(10, 2)
    print(f"除法结果: {result}")
    result = divide(10, 0)
except AssertionError as e:
    print(f"断言错误: {e}")

# 7. 练习
print("\n7. 练习")

# 练习1: 猜数字游戏
print("练习1: 猜数字游戏")
import random

# 生成1-100之间的随机数
number = random.randint(1, 100)
guesses = 0

# 注释掉循环，避免运行时阻塞
"""
while True:
    guess = int(input("请猜一个1-100之间的数字: "))
    guesses += 1
    
    if guess < number:
        print("太小了！")
    elif guess > number:
        print("太大了！")
    else:
        print(f"恭喜你，猜对了！用了{guesses}次。")
        break
"""
print("猜数字游戏逻辑已实现，实际运行时会提示用户输入")

# 练习2: 计算阶乘
def factorial(n):
    """计算n的阶乘"""
    if n < 0:
        return "错误: 负数没有阶乘"
    elif n == 0:
        return 1
    else:
        result = 1
        for i in range(1, n + 1):
            result *= i
        return result

print(f"5的阶乘: {factorial(5)}")
print(f"0的阶乘: {factorial(0)}")
print(f"-1的阶乘: {factorial(-1)}")

# 练习3: 判断素数
def is_prime(n):
    """判断n是否为素数"""
    if n <= 1:
        return False
    elif n <= 3:
        return True
    elif n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

print("100以内的素数:")
primes = [x for x in range(100) if is_prime(x)]
print(primes)

# 练习4: 斐波那契数列
def fibonacci(n):
    """生成前n个斐波那契数列"""
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    else:
        fib = [0, 1]
        for i in range(2, n):
            fib.append(fib[i-1] + fib[i-2])
        return fib

print(f"前10个斐波那契数: {fibonacci(10)}")

print("\n=== 第19天学习示例结束 ===")
