#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第18天：Python数据结构
Python编程基础学习示例
内容：列表、元组、字典、集合等数据结构
"""

print("=== 第18天：Python数据结构 ===")

# 1. 列表 (List)
print("\n1. 列表 (List)")

# 创建列表
fruits = ["apple", "banana", "cherry"]
numbers = [1, 2, 3, 4, 5]
mixed = [1, "apple", 3.14, True]
print(f" fruits = {fruits}")
print(f" numbers = {numbers}")
print(f" mixed = {mixed}")

# 访问列表元素
print(f"\n访问列表元素:")
print(f" fruits[0] = {fruits[0]}")
print(f" fruits[-1] = {fruits[-1]}")
print(f" numbers[1:3] = {numbers[1:3]}")

# 修改列表
print(f"\n修改列表:")
fruits[1] = "orange"
print(f" 修改后 fruits = {fruits}")

# 列表方法
print(f"\n列表方法:")

# 添加元素
fruits.append("grape")
print(f" append('grape') 后: {fruits}")

fruits.insert(1, "pear")
print(f" insert(1, 'pear') 后: {fruits}")

# 移除元素
fruits.remove("orange")
print(f" remove('orange') 后: {fruits}")

popped = fruits.pop()
print(f" pop() 后: {fruits}, 弹出的元素: {popped}")

# 其他方法
print(f"\n其他列表方法:")
print(f" len(fruits) = {len(fruits)}")
print(f" 'apple' in fruits: {'apple' in fruits}")
print(f" fruits.count('apple') = {fruits.count('apple')}")
print(f" fruits.index('cherry') = {fruits.index('cherry')}")

# 排序
numbers = [3, 1, 4, 1, 5, 9, 2, 6]
numbers.sort()
print(f" 排序后 numbers = {numbers}")

numbers.sort(reverse=True)
print(f" 倒序排序后 numbers = {numbers}")

# 复制列表
fruits_copy = fruits.copy()
print(f" fruits_copy = {fruits_copy}")

# 2. 元组 (Tuple)
print("\n2. 元组 (Tuple)")

# 创建元组
tuple1 = (1, 2, 3)
tuple2 = ("apple", "banana", "cherry")
tuple3 = (1, "apple", 3.14)
print(f" tuple1 = {tuple1}")
print(f" tuple2 = {tuple2}")
print(f" tuple3 = {tuple3}")

# 访问元组元素
print(f"\n访问元组元素:")
print(f" tuple1[0] = {tuple1[0]}")
print(f" tuple2[-1] = {tuple2[-1]}")

# 元组是不可变的
# tuple1[0] = 10  # 这会引发错误

# 元组方法
print(f"\n元组方法:")
print(f" len(tuple1) = {len(tuple1)}")
print(f" tuple2.count('apple') = {tuple2.count('apple')}")
print(f" tuple2.index('banana') = {tuple2.index('banana')}")

# 3. 字典 (Dictionary)
print("\n3. 字典 (Dictionary)")

# 创建字典
person = {
    "name": "Alice",
    "age": 30,
    "city": "New York"
}

print(f" person = {person}")

# 访问字典值
print(f"\n访问字典值:")
print(f" person['name'] = {person['name']}")
print(f" person.get('age') = {person.get('age')}")
print(f" person.get('email', 'Not available') = {person.get('email', 'Not available')}")

# 修改字典
print(f"\n修改字典:")
person["age"] = 31
print(f" 修改年龄后: {person}")

person["email"] = "alice@example.com"
print(f" 添加邮箱后: {person}")

# 移除字典项
print(f"\n移除字典项:")
del person["city"]
print(f" 删除city后: {person}")

popped_value = person.pop("age")
print(f" 弹出age后: {person}, 弹出的值: {popped_value}")

# 字典方法
print(f"\n字典方法:")
print(f" person.keys() = {list(person.keys())}")
print(f" person.values() = {list(person.values())}")
print(f" person.items() = {list(person.items())}")

# 4. 集合 (Set)
print("\n4. 集合 (Set)")

# 创建集合
fruits_set = {"apple", "banana", "cherry", "apple"}  # 自动去重
numbers_set = {1, 2, 3, 4, 5}
print(f" fruits_set = {fruits_set}")
print(f" numbers_set = {numbers_set}")

# 添加元素
print(f"\n添加元素:")
fruits_set.add("orange")
print(f" add('orange') 后: {fruits_set}")

# 移除元素
print(f"\n移除元素:")
fruits_set.remove("banana")  # 如果元素不存在会引发错误
print(f" remove('banana') 后: {fruits_set}")

fruits_set.discard("grape")  # 如果元素不存在不会引发错误
print(f" discard('grape') 后: {fruits_set}")

# 集合操作
print(f"\n集合操作:")
set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}

print(f" set1 = {set1}")
print(f" set2 = {set2}")
print(f" 并集: {set1 | set2}")
print(f" 交集: {set1 & set2}")
print(f" 差集: {set1 - set2}")
print(f" 对称差集: {set1 ^ set2}")

# 5. 嵌套数据结构
print("\n5. 嵌套数据结构")

# 列表中的列表
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print(f" 矩阵: {matrix}")
print(f" matrix[1][2] = {matrix[1][2]}")

# 字典中的列表
student = {
    "name": "Bob",
    "grades": [85, 90, 95]
}
print(f" 学生: {student}")
print(f" 平均成绩: {sum(student['grades']) / len(student['grades'])}")

# 字典中的字典
company = {
    "name": "Tech Corp",
    "employees": {
        "Alice": {"position": "Engineer", "salary": 100000},
        "Bob": {"position": "Manager", "salary": 120000}
    }
}
print(f" 公司: {company}")
print(f" Alice的职位: {company['employees']['Alice']['position']}")

# 6. 列表推导式
print("\n6. 列表推导式")

# 基本列表推导式
squares = [x**2 for x in range(10)]
print(f" 平方数: {squares}")

# 带条件的列表推导式
even_squares = [x**2 for x in range(10) if x % 2 == 0]
print(f" 偶数平方: {even_squares}")

# 嵌套列表推导式
matrix = [[i*j for j in range(3)] for i in range(3)]
print(f" 矩阵: {matrix}")

# 7. 字典推导式
print("\n7. 字典推导式")

# 基本字典推导式
square_dict = {x: x**2 for x in range(5)}
print(f" 平方字典: {square_dict}")

# 带条件的字典推导式
even_square_dict = {x: x**2 for x in range(10) if x % 2 == 0}
print(f" 偶数平方字典: {even_square_dict}")

# 8. 集合推导式
print("\n8. 集合推导式")

# 基本集合推导式
square_set = {x**2 for x in range(10)}
print(f" 平方集合: {square_set}")

# 带条件的集合推导式
even_square_set = {x**2 for x in range(10) if x % 2 == 0}
print(f" 偶数平方集合: {even_square_set}")

# 9. 练习
print("\n9. 练习")

# 练习1: 统计字符串中每个字符出现的次数
def count_characters(text):
    char_count = {}
    for char in text:
        if char in char_count:
            char_count[char] += 1
        else:
            char_count[char] = 1
    return char_count

text = "Hello, World!"
char_count = count_characters(text)
print(f" 字符计数: {char_count}")

# 练习2: 找出列表中的重复元素
def find_duplicates(lst):
    seen = set()
    duplicates = set()
    for item in lst:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)
    return duplicates

numbers = [1, 2, 3, 2, 4, 5, 5, 6]
duplicates = find_duplicates(numbers)
print(f" 重复元素: {duplicates}")

# 练习3: 合并两个字典
def merge_dicts(dict1, dict2):
    merged = dict1.copy()
    for key, value in dict2.items():
        if key in merged:
            if isinstance(merged[key], list) and isinstance(value, list):
                merged[key].extend(value)
            else:
                merged[key] = value
        else:
            merged[key] = value
    return merged

dict1 = {"a": 1, "b": 2, "c": [1, 2]}
dict2 = {"b": 3, "d": 4, "c": [3, 4]}
merged = merge_dicts(dict1, dict2)
print(f" 合并后的字典: {merged}")

print("\n=== 第18天学习示例结束 ===")
