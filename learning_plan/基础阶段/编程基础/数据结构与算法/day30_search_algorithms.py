#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第30天：搜索算法
数据结构与算法学习示例
内容：线性搜索、二分搜索、深度优先搜索、广度优先搜索
"""

print("=== 第30天：搜索算法 ===")

import time
import random

# 1. 搜索算法概述
print("\n1. 搜索算法概述")

print("搜索算法是用于在数据集合中查找特定元素的算法")
print("- 线性搜索：逐个检查元素")
print("- 二分搜索：在有序数组中使用分治策略")
print("- 深度优先搜索：用于图和树的搜索")
print("- 广度优先搜索：用于图和树的搜索")

# 2. 线性搜索
print("\n2. 线性搜索")

print("线性搜索是一种简单的搜索算法，逐个检查数组中的元素")


def linear_search(arr, target):
    """线性搜索"""
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1

# 测试线性搜索
print("测试线性搜索:")
test_arr = [64, 34, 25, 12, 22, 11, 90]
target = 25
result = linear_search(test_arr, target)
print(f"在数组 {test_arr} 中搜索 {target}: {'找到，索引为' + str(result) if result != -1 else '未找到'}")

target = 100
result = linear_search(test_arr, target)
print(f"在数组 {test_arr} 中搜索 {target}: {'找到，索引为' + str(result) if result != -1 else '未找到'}")

# 3. 二分搜索
print("\n3. 二分搜索")

print("二分搜索是一种高效的搜索算法，适用于有序数组")


def binary_search(arr, target):
    """二分搜索"""
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

def binary_search_recursive(arr, target, left, right):
    """递归二分搜索"""
    if left > right:
        return -1
    mid = left + (right - left) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)

# 测试二分搜索
print("测试二分搜索:")
sorted_arr = [11, 12, 22, 25, 34, 64, 90]
target = 25
result = binary_search(sorted_arr, target)
print(f"在数组 {sorted_arr} 中搜索 {target}: {'找到，索引为' + str(result) if result != -1 else '未找到'}")

result = binary_search_recursive(sorted_arr, target, 0, len(sorted_arr) - 1)
print(f"递归二分搜索结果: {'找到，索引为' + str(result) if result != -1 else '未找到'}")

target = 100
result = binary_search(sorted_arr, target)
print(f"在数组 {sorted_arr} 中搜索 {target}: {'找到，索引为' + str(result) if result != -1 else '未找到'}")

# 4. 深度优先搜索（DFS）
print("\n4. 深度优先搜索（DFS）")

print("深度优先搜索是一种用于图和树的搜索算法，优先探索深度")

# 图的表示
class Graph:
    def __init__(self, vertices):
        """初始化图"""
        self.V = vertices
        self.adj = [[] for _ in range(vertices)]
    
    def add_edge(self, u, v):
        """添加边"""
        self.adj[u].append(v)
        self.adj[v].append(u)  # 无向图
    
    def dfs(self, start, visited):
        """深度优先搜索"""
        visited[start] = True
        print(start, end=" ")
        for neighbor in self.adj[start]:
            if not visited[neighbor]:
                self.dfs(neighbor, visited)
    
    def dfs_iterative(self, start):
        """迭代式深度优先搜索"""
        visited = [False] * self.V
        stack = [start]
        while stack:
            vertex = stack.pop()
            if not visited[vertex]:
                print(vertex, end=" ")
                visited[vertex] = True
                # 逆序压栈，保证顺序与递归一致
                for neighbor in reversed(self.adj[vertex]):
                    if not visited[neighbor]:
                        stack.append(neighbor)

# 测试深度优先搜索
print("测试深度优先搜索:")
g = Graph(6)
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(1, 3)
g.add_edge(2, 4)
g.add_edge(2, 5)

print("递归式DFS (从0开始):")
visited = [False] * g.V
g.dfs(0, visited)
print()

print("迭代式DFS (从0开始):")
g.dfs_iterative(0)
print()

# 5. 广度优先搜索（BFS）
print("\n5. 广度优先搜索（BFS）")

print("广度优先搜索是一种用于图和树的搜索算法，优先探索广度")

class GraphBFS(Graph):
    def bfs(self, start):
        """广度优先搜索"""
        visited = [False] * self.V
        queue = [start]
        visited[start] = True
        while queue:
            vertex = queue.pop(0)
            print(vertex, end=" ")
            for neighbor in self.adj[vertex]:
                if not visited[neighbor]:
                    queue.append(neighbor)
                    visited[neighbor] = True

# 测试广度优先搜索
print("测试广度优先搜索:")
g_bfs = GraphBFS(6)
g_bfs.add_edge(0, 1)
g_bfs.add_edge(0, 2)
g_bfs.add_edge(1, 3)
g_bfs.add_edge(2, 4)
g_bfs.add_edge(2, 5)

print("BFS (从0开始):")
g_bfs.bfs(0)
print()

# 6. 时间复杂度分析
print("\n6. 时间复杂度分析")

print("各搜索算法的时间复杂度:")
print("- 线性搜索: O(n)")
print("- 二分搜索: O(log n)")
print("- 深度优先搜索: O(V + E)，其中V是顶点数，E是边数")
print("- 广度优先搜索: O(V + E)")

# 7. 空间复杂度分析
print("\n7. 空间复杂度分析")

print("各搜索算法的空间复杂度:")
print("- 线性搜索: O(1)")
print("- 二分搜索: O(1)（迭代），O(log n)（递归）")
print("- 深度优先搜索: O(V)（递归栈空间）")
print("- 广度优先搜索: O(V)（队列空间）")

# 8. 性能测试
print("\n8. 性能测试")

# 生成有序数组
def generate_sorted_array(size):
    return list(range(size))

# 测试搜索算法性能
def test_search_performance():
    sizes = [10000, 100000, 1000000]
    for size in sizes:
        print(f"\n测试数组大小: {size}")
        arr = generate_sorted_array(size)
        target = size - 1  # 搜索最后一个元素
        
        # 线性搜索
        start_time = time.time()
        linear_search(arr, target)
        end_time = time.time()
        print(f"线性搜索: {end_time - start_time:.6f} 秒")
        
        # 二分搜索
        start_time = time.time()
        binary_search(arr, target)
        end_time = time.time()
        print(f"二分搜索: {end_time - start_time:.6f} 秒")
        
        # 递归二分搜索
        start_time = time.time()
        binary_search_recursive(arr, target, 0, len(arr) - 1)
        end_time = time.time()
        print(f"递归二分搜索: {end_time - start_time:.6f} 秒")

# 运行性能测试
test_search_performance()

# 9. 搜索算法的应用场景
print("\n9. 搜索算法的应用场景")

print("不同搜索算法的适用场景:")
print("- 线性搜索: 小规模数据，或无序数据")
print("- 二分搜索: 有序数据，需要高效搜索")
print("- 深度优先搜索: 图的遍历，路径查找，拓扑排序")
print("- 广度优先搜索: 图的遍历，最短路径查找，层次遍历")

# 10. 高级搜索算法
print("\n10. 高级搜索算法")

print("- 插值搜索: 对均匀分布的有序数组进行优化")
print("- 斐波那契搜索: 黄金分割比例的搜索算法")
print("- 跳表搜索: 基于链表的高效搜索结构")
print("- 哈希表: O(1) 平均时间复杂度的搜索")

# 11. 练习
print("\n11. 练习")

# 练习1: 实现插值搜索
print("练习1: 实现插值搜索")
print("- 插值搜索是二分搜索的改进版，适用于均匀分布的有序数组")
print("- 实现插值搜索并测试性能")

# 练习2: 实现斐波那契搜索
print("\n练习2: 实现斐波那契搜索")
print("- 斐波那契搜索使用黄金分割比例来确定搜索位置")
print("- 实现斐波那契搜索并测试性能")

# 练习3: 实现跳表
print("\n练习3: 实现跳表")
print("- 跳表是一种基于链表的高效搜索结构")
print("- 实现跳表的插入、删除和搜索操作")

# 练习4: 实现哈希表
print("\n练习4: 实现哈希表")
print("- 哈希表是一种通过哈希函数将键映射到值的数据结构")
print("- 实现哈希表的插入、删除和搜索操作")

# 练习5: 实现图的最短路径算法
print("\n练习5: 实现图的最短路径算法")
print("- 使用BFS实现无权图的最短路径")
print("- 测试最短路径算法")

# 12. 代码优化建议
print("\n12. 代码优化建议")

print("- 对于有序数据，优先使用二分搜索")
print("- 对于图的搜索，根据具体问题选择DFS或BFS")
print("- 对于大规模数据，考虑使用哈希表或跳表等数据结构")
print("- 注意递归深度，避免栈溢出")
print("- 对于频繁搜索的场景，考虑预处理数据结构")

print("\n=== 第30天学习示例结束 ===")
