#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第29天：排序算法
数据结构与算法学习示例
内容：冒泡排序、选择排序、插入排序、快速排序、归并排序
"""

print("=== 第29天：排序算法 ===")

import time
import random

# 1. 排序算法概述
print("\n1. 排序算法概述")

print("排序算法是将一组数据按照特定顺序排列的算法")
print("- 内部排序：所有数据在内存中处理")
print("- 外部排序：数据量较大，需要使用外部存储")
print("- 稳定排序：相同值的相对顺序保持不变")
print("- 不稳定排序：相同值的相对顺序可能改变")

# 2. 冒泡排序
print("\n2. 冒泡排序")

print("冒泡排序是一种简单的排序算法，通过重复遍历要排序的数组，比较相邻元素并交换")


def bubble_sort(arr):
    """冒泡排序"""
    n = len(arr)
    for i in range(n):
        # 标记是否发生交换
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        # 如果没有交换，说明数组已经有序
        if not swapped:
            break
    return arr

# 测试冒泡排序
print("测试冒泡排序:")
test_arr = [64, 34, 25, 12, 22, 11, 90]
print(f"原始数组: {test_arr}")
sorted_arr = bubble_sort(test_arr.copy())
print(f"排序后数组: {sorted_arr}")

# 3. 选择排序
print("\n3. 选择排序")

print("选择排序是一种简单的排序算法，每次从未排序部分选择最小元素放到已排序部分的末尾")


def selection_sort(arr):
    """选择排序"""
    n = len(arr)
    for i in range(n):
        # 找到未排序部分的最小元素
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        # 交换元素
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr

# 测试选择排序
print("测试选择排序:")
test_arr = [64, 34, 25, 12, 22, 11, 90]
print(f"原始数组: {test_arr}")
sorted_arr = selection_sort(test_arr.copy())
print(f"排序后数组: {sorted_arr}")

# 4. 插入排序
print("\n4. 插入排序")

print("插入排序是一种简单的排序算法，将元素插入到已排序部分的适当位置")


def insertion_sort(arr):
    """插入排序"""
    n = len(arr)
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        # 将大于key的元素向右移动
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

# 测试插入排序
print("测试插入排序:")
test_arr = [64, 34, 25, 12, 22, 11, 90]
print(f"原始数组: {test_arr}")
sorted_arr = insertion_sort(test_arr.copy())
print(f"排序后数组: {sorted_arr}")

# 5. 快速排序
print("\n5. 快速排序")

print("快速排序是一种高效的排序算法，使用分治策略")


def quick_sort(arr):
    """快速排序"""
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

# 测试快速排序
print("测试快速排序:")
test_arr = [64, 34, 25, 12, 22, 11, 90]
print(f"原始数组: {test_arr}")
sorted_arr = quick_sort(test_arr.copy())
print(f"排序后数组: {sorted_arr}")

# 6. 归并排序
print("\n6. 归并排序")

print("归并排序是一种稳定的排序算法，使用分治策略")


def merge_sort(arr):
    """归并排序"""
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

# 测试归并排序
print("测试归并排序:")
test_arr = [64, 34, 25, 12, 22, 11, 90]
print(f"原始数组: {test_arr}")
sorted_arr = merge_sort(test_arr.copy())
print(f"排序后数组: {sorted_arr}")

# 7. 堆排序
print("\n7. 堆排序")

print("堆排序是一种基于二叉堆的排序算法")


def heap_sort(arr):
    """堆排序"""
    n = len(arr)
    # 构建最大堆
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    # 逐个提取元素
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)
    return arr

def heapify(arr, n, i):
    """堆化操作"""
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2
    if left < n and arr[left] > arr[largest]:
        largest = left
    if right < n and arr[right] > arr[largest]:
        largest = right
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

# 测试堆排序
print("测试堆排序:")
test_arr = [64, 34, 25, 12, 22, 11, 90]
print(f"原始数组: {test_arr}")
sorted_arr = heap_sort(test_arr.copy())
print(f"排序后数组: {sorted_arr}")

# 8. 时间复杂度分析
print("\n8. 时间复杂度分析")

print("各排序算法的时间复杂度:")
print("- 冒泡排序: O(n²)")
print("- 选择排序: O(n²)")
print("- 插入排序: O(n²)，但在接近有序的情况下性能较好")
print("- 快速排序: O(n log n) 平均情况，O(n²) 最坏情况")
print("- 归并排序: O(n log n)")
print("- 堆排序: O(n log n)")

# 9. 空间复杂度分析
print("\n9. 空间复杂度分析")

print("各排序算法的空间复杂度:")
print("- 冒泡排序: O(1)")
print("- 选择排序: O(1)")
print("- 插入排序: O(1)")
print("- 快速排序: O(log n) 递归栈空间")
print("- 归并排序: O(n) 额外空间")
print("- 堆排序: O(1)")

# 10. 排序算法比较
print("\n10. 排序算法比较")

print("各排序算法的优缺点:")
print("- 冒泡排序: 简单易实现，但效率低")
print("- 选择排序: 简单易实现，交换次数少，但效率低")
print("- 插入排序: 对于小规模数据或接近有序的数据效率较高")
print("- 快速排序: 平均情况下效率高，但最坏情况性能差")
print("- 归并排序: 稳定，效率高，但需要额外空间")
print("- 堆排序: 效率高，不需要额外空间，但不稳定")

# 11. 性能测试
print("\n11. 性能测试")

# 生成随机数组
def generate_random_array(size):
    return [random.randint(0, 10000) for _ in range(size)]

# 测试排序算法性能
def test_sort_performance():
    sizes = [1000, 5000, 10000]
    for size in sizes:
        print(f"\n测试数组大小: {size}")
        arr = generate_random_array(size)
        
        # 冒泡排序
        start_time = time.time()
        bubble_sort(arr.copy())
        end_time = time.time()
        print(f"冒泡排序: {end_time - start_time:.4f} 秒")
        
        # 选择排序
        start_time = time.time()
        selection_sort(arr.copy())
        end_time = time.time()
        print(f"选择排序: {end_time - start_time:.4f} 秒")
        
        # 插入排序
        start_time = time.time()
        insertion_sort(arr.copy())
        end_time = time.time()
        print(f"插入排序: {end_time - start_time:.4f} 秒")
        
        # 快速排序
        start_time = time.time()
        quick_sort(arr.copy())
        end_time = time.time()
        print(f"快速排序: {end_time - start_time:.4f} 秒")
        
        # 归并排序
        start_time = time.time()
        merge_sort(arr.copy())
        end_time = time.time()
        print(f"归并排序: {end_time - start_time:.4f} 秒")
        
        # 堆排序
        start_time = time.time()
        heap_sort(arr.copy())
        end_time = time.time()
        print(f"堆排序: {end_time - start_time:.4f} 秒")

# 运行性能测试
test_sort_performance()

# 12. 排序算法的应用场景
print("\n12. 排序算法的应用场景")

print("不同排序算法的适用场景:")
print("- 冒泡排序: 小规模数据，或作为教学示例")
print("- 选择排序: 小规模数据，或对交换次数有要求的场景")
print("- 插入排序: 小规模数据，或接近有序的数据")
print("- 快速排序: 一般情况下的首选排序算法")
print("- 归并排序: 需要稳定排序的场景")
print("- 堆排序: 内存有限的场景")

# 13. 练习
print("\n13. 练习")

# 练习1: 实现希尔排序
print("练习1: 实现希尔排序")
print("- 希尔排序是插入排序的改进版，通过分组插入提高效率")
print("- 实现希尔排序并测试性能")

# 练习2: 实现计数排序
print("\n练习2: 实现计数排序")
print("- 计数排序是一种非比较排序算法，适用于整数排序")
print("- 实现计数排序并测试性能")

# 练习3: 实现桶排序
print("\n练习3: 实现桶排序")
print("- 桶排序是一种分治排序算法，将数据分到不同的桶中")
print("- 实现桶排序并测试性能")

# 练习4: 实现基数排序
print("\n练习4: 实现基数排序")
print("- 基数排序是一种非比较排序算法，按位排序")
print("- 实现基数排序并测试性能")

# 练习5: 优化快速排序
print("\n练习5: 优化快速排序")
print("- 实现随机选择 pivot 的快速排序")
print("- 测试优化后的性能")

print("\n=== 第29天学习示例结束 ===")
