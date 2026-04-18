#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第32天：算法复杂度分析
数据结构与算法学习示例
内容：时间复杂度、空间复杂度、大O表示法
"""

print("=== 第32天：算法复杂度分析 ===")

# 1. 算法复杂度概述
print("\n1. 算法复杂度概述")

print("算法复杂度是衡量算法效率的重要指标")
print("- 时间复杂度：算法执行时间随输入规模增长的变化趋势")
print("- 空间复杂度：算法所需存储空间随输入规模增长的变化趋势")
print("- 大O表示法：用于描述算法复杂度的渐近行为")

# 2. 大O表示法
print("\n2. 大O表示法")

print("大O表示法是一种用于描述算法复杂度的数学符号")
print("- O(1): 常数时间复杂度")
print("- O(log n): 对数时间复杂度")
print("- O(n): 线性时间复杂度")
print("- O(n log n): 线性对数时间复杂度")
print("- O(n²): 平方时间复杂度")
print("- O(n³): 立方时间复杂度")
print("- O(2^n): 指数时间复杂度")
print("- O(n!): 阶乘时间复杂度")

# 3. 时间复杂度分析
print("\n3. 时间复杂度分析")

# 3.1 常数时间复杂度 O(1)
print("\n3.1 常数时间复杂度 O(1)")

def constant_time_operation(arr, index):
    """常数时间复杂度的操作"""
    return arr[index] if 0 <= index < len(arr) else None

print("常数时间复杂度的操作:")
print("- 访问数组元素")
print("- 基本算术运算")
print("- 赋值操作")
print("- 条件判断")

# 3.2 线性时间复杂度 O(n)
print("\n3.2 线性时间复杂度 O(n)")

def linear_time_operation(arr):
    """线性时间复杂度的操作"""
    sum = 0
    for element in arr:
        sum += element
    return sum

print("线性时间复杂度的操作:")
print("- 线性搜索")
print("- 遍历数组")
print("- 计算数组和")
print("- 单层循环")

# 3.3 对数时间复杂度 O(log n)
print("\n3.3 对数时间复杂度 O(log n)")

def binary_search(arr, target):
    """二分搜索 - 对数时间复杂度"""
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

print("对数时间复杂度的操作:")
print("- 二分搜索")
print("- 二叉搜索树的查找")
print("- 分治算法的某些步骤")

# 3.4 线性对数时间复杂度 O(n log n)
print("\n3.4 线性对数时间复杂度 O(n log n)")

def merge_sort(arr):
    """归并排序 - 线性对数时间复杂度"""
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    """合并两个有序数组"""
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

print("线性对数时间复杂度的操作:")
print("- 归并排序")
print("- 快速排序（平均情况）")
print("- 堆排序")
print("- 某些分治算法")

# 3.5 平方时间复杂度 O(n²)
print("\n3.5 平方时间复杂度 O(n²)")

def bubble_sort(arr):
    """冒泡排序 - 平方时间复杂度"""
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

print("平方时间复杂度的操作:")
print("- 冒泡排序")
print("- 选择排序")
print("- 插入排序")
print("- 双层循环")
print("- 矩阵乘法")

# 3.6 立方时间复杂度 O(n³)
print("\n3.6 立方时间复杂度 O(n³)")

def matrix_multiply(a, b):
    """矩阵乘法 - 立方时间复杂度"""
    n = len(a)
    result = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                result[i][j] += a[i][k] * b[k][j]
    return result

print("立方时间复杂度的操作:")
print("- 矩阵乘法")
print("- 三层循环")
print("- 某些动态规划算法")

# 3.7 指数时间复杂度 O(2^n)
print("\n3.7 指数时间复杂度 O(2^n)")

def fibonacci_recursive(n):
    """递归斐波那契 - 指数时间复杂度"""
    if n <= 1:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)

print("指数时间复杂度的操作:")
print("- 递归斐波那契")
print("- 暴力破解子集问题")
print("- 旅行商问题的暴力解法")

# 4. 空间复杂度分析
print("\n4. 空间复杂度分析")

# 4.1 常数空间复杂度 O(1)
print("\n4.1 常数空间复杂度 O(1)")

def constant_space_operation(arr):
    """常数空间复杂度的操作"""
    max_value = float('-inf')
    for element in arr:
        if element > max_value:
            max_value = element
    return max_value

print("常数空间复杂度的操作:")
print("- 只使用固定数量的变量")
print("- 不使用与输入规模相关的额外空间")

# 4.2 线性空间复杂度 O(n)
print("\n4.2 线性空间复杂度 O(n)")

def linear_space_operation(arr):
    """线性空间复杂度的操作"""
    new_arr = arr.copy()
    return new_arr

print("线性空间复杂度的操作:")
print("- 复制数组")
print("- 使用与输入规模相关的线性额外空间")
print("- 递归调用栈（深度为n）")

# 4.3 平方空间复杂度 O(n²)
print("\n4.3 平方空间复杂度 O(n²)")

def square_space_operation(n):
    """平方空间复杂度的操作"""
    matrix = [[0] * n for _ in range(n)]
    return matrix

print("平方空间复杂度的操作:")
print("- 创建二维数组")
print("- 某些动态规划算法")

# 5. 最好、最坏和平均时间复杂度
print("\n5. 最好、最坏和平均时间复杂度")

print("- 最好时间复杂度：在最理想情况下的时间复杂度")
print("- 最坏时间复杂度：在最糟糕情况下的时间复杂度")
print("- 平均时间复杂度：在所有可能输入下的期望时间复杂度")

# 6. 算法复杂度的比较
print("\n6. 算法复杂度的比较")

print("算法复杂度的增长顺序:")
print("O(1) < O(log n) < O(n) < O(n log n) < O(n²) < O(n³) < O(2^n) < O(n!)")

# 7. 实际性能测试
print("\n7. 实际性能测试")

import time
import random

def generate_random_array(size):
    """生成随机数组"""
    return [random.randint(0, 10000) for _ in range(size)]

def test_algorithm_performance():
    """测试不同算法的性能"""
    sizes = [1000, 5000, 10000]
    for size in sizes:
        print(f"\n测试数组大小: {size}")
        arr = generate_random_array(size)
        
        # 测试线性搜索
        start_time = time.time()
        for _ in range(10):
            linear_search(arr, size // 2)
        end_time = time.time()
        print(f"线性搜索: {(end_time - start_time) / 10:.6f} 秒")
        
        # 测试二分搜索（需要先排序）
        sorted_arr = sorted(arr)
        start_time = time.time()
        for _ in range(10):
            binary_search(sorted_arr, size // 2)
        end_time = time.time()
        print(f"二分搜索: {(end_time - start_time) / 10:.6f} 秒")
        
        # 测试冒泡排序
        start_time = time.time()
        bubble_sort(arr.copy())
        end_time = time.time()
        print(f"冒泡排序: {end_time - start_time:.6f} 秒")
        
        # 测试归并排序
        start_time = time.time()
        merge_sort(arr.copy())
        end_time = time.time()
        print(f"归并排序: {end_time - start_time:.6f} 秒")

def linear_search(arr, target):
    """线性搜索"""
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1

# 运行性能测试
test_algorithm_performance()

# 8. 算法优化策略
print("\n8. 算法优化策略")

print("算法优化的常见策略:")
print("- 减少时间复杂度：使用更高效的算法")
print("- 减少空间复杂度：优化数据结构，使用原地算法")
print("- 记忆化：缓存重复计算的结果")
print("- 剪枝：提前终止不必要的计算")
print("- 并行计算：利用多核处理器")
print("- 空间换时间：使用额外空间来提高时间效率")

# 9. 实际应用中的复杂度考虑
print("\n9. 实际应用中的复杂度考虑")

print("在实际应用中，需要根据具体情况选择合适的算法:")
print("- 对于小规模数据，简单算法可能比复杂算法更高效")
print("- 对于大规模数据，必须选择时间复杂度低的算法")
print("- 内存有限的情况下，需要考虑空间复杂度")
print("- 实时系统中，需要保证最坏时间复杂度")

# 10. 练习
print("\n10. 练习")

# 练习1: 分析算法复杂度
print("练习1: 分析算法复杂度")
print("- 分析以下代码的时间复杂度和空间复杂度:")
print("  def example(n):")
print("      for i in range(n):")
print("          for j in range(i):")
print("              print(i, j)")

# 练习2: 优化算法
print("\n练习2: 优化算法")
print("- 优化以下代码，降低时间复杂度:")
print("  def find_duplicates(arr):")
print("      duplicates = []")
print("      for i in range(len(arr)):")
print("          for j in range(i + 1, len(arr)):")
print("              if arr[i] == arr[j] and arr[i] not in duplicates:")
print("                  duplicates.append(arr[i])")
print("      return duplicates")

# 练习3: 设计高效算法
print("\n练习3: 设计高效算法")
print("- 设计一个算法，在O(n)时间复杂度内找到数组中的多数元素")
print("- 多数元素是指在数组中出现次数大于⌊n/2⌋的元素")

# 练习4: 空间复杂度优化
print("\n练习4: 空间复杂度优化")
print("- 优化以下代码，降低空间复杂度:")
print("  def reverse_array(arr):")
print("      reversed_arr = []")
print("      for i in range(len(arr) - 1, -1, -1):")
print("          reversed_arr.append(arr[i])")
print("      return reversed_arr")

# 练习5: 分析递归算法的复杂度
print("\n练习5: 分析递归算法的复杂度")
print("- 分析以下递归算法的时间复杂度和空间复杂度:")
print("  def factorial(n):")
print("      if n <= 1:")
print("          return 1")
print("      return n * factorial(n - 1)")

# 11. 代码优化建议
print("\n11. 代码优化建议")

print("- 选择合适的算法和数据结构")
print("- 避免不必要的计算")
print("- 合理使用缓存")
print("- 注意算法的边界情况")
print("- 进行性能测试和基准测试")
print("- 考虑算法的可维护性")

print("\n=== 第32天学习示例结束 ===")
