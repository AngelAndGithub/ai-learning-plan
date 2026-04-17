#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据结构与算法示例代码
涵盖线性数据结构、非线性数据结构、排序算法、搜索算法、动态规划等内容
"""

# 1. 线性数据结构
print("=== 1. 线性数据结构 ===")

# 1.1 数组
print("\n1.1 数组:")
array = [1, 2, 3, 4, 5]
print(f"原始数组: {array}")
print(f"访问元素: array[2] = {array[2]}")
array.append(6)
print(f"添加元素后: {array}")
array.pop()
print(f"删除元素后: {array}")

# 1.2 链表
print("\n1.2 链表:")

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class LinkedList:
    def __init__(self):
        self.head = None
    
    def append(self, val):
        if not self.head:
            self.head = ListNode(val)
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = ListNode(val)
    
    def print_list(self):
        current = self.head
        while current:
            print(current.val, end=" -> ")
            current = current.next
        print("None")
    
    def delete(self, val):
        if not self.head:
            return
        if self.head.val == val:
            self.head = self.head.next
            return
        current = self.head
        while current.next:
            if current.next.val == val:
                current.next = current.next.next
                return
            current = current.next

linked_list = LinkedList()
linked_list.append(1)
linked_list.append(2)
linked_list.append(3)
linked_list.append(4)
linked_list.append(5)
print("原始链表:")
linked_list.print_list()
linked_list.delete(3)
print("删除元素3后:")
linked_list.print_list()

# 1.3 栈
print("\n1.3 栈:")

class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        if not self.is_empty():
            return self.items.pop()
        return None
    
    def peek(self):
        if not self.is_empty():
            return self.items[-1]
        return None
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)

stack = Stack()
stack.push(1)
stack.push(2)
stack.push(3)
print(f"栈顶元素: {stack.peek()}")
print(f"栈大小: {stack.size()}")
print(f"弹出元素: {stack.pop()}")
print(f"栈大小: {stack.size()}")
print(f"栈是否为空: {stack.is_empty()}")

# 1.4 队列
print("\n1.4 队列:")

class Queue:
    def __init__(self):
        self.items = []
    
    def enqueue(self, item):
        self.items.append(item)
    
    def dequeue(self):
        if not self.is_empty():
            return self.items.pop(0)
        return None
    
    def front(self):
        if not self.is_empty():
            return self.items[0]
        return None
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)

queue = Queue()
queue.enqueue(1)
queue.enqueue(2)
queue.enqueue(3)
print(f"队首元素: {queue.front()}")
print(f"队列大小: {queue.size()}")
print(f"出队元素: {queue.dequeue()}")
print(f"队列大小: {queue.size()}")
print(f"队列是否为空: {queue.is_empty()}")

# 2. 非线性数据结构
print("\n=== 2. 非线性数据结构 ===")

# 2.1 二叉树
print("\n2.1 二叉树:")

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class BinaryTree:
    def __init__(self):
        self.root = None
    
    def insert(self, val):
        if not self.root:
            self.root = TreeNode(val)
            return
        queue = [self.root]
        while queue:
            current = queue.pop(0)
            if not current.left:
                current.left = TreeNode(val)
                return
            else:
                queue.append(current.left)
            if not current.right:
                current.right = TreeNode(val)
                return
            else:
                queue.append(current.right)
    
    def inorder_traversal(self, node):
        if node:
            self.inorder_traversal(node.left)
            print(node.val, end=" ")
            self.inorder_traversal(node.right)
    
    def preorder_traversal(self, node):
        if node:
            print(node.val, end=" ")
            self.preorder_traversal(node.left)
            self.preorder_traversal(node.right)
    
    def postorder_traversal(self, node):
        if node:
            self.postorder_traversal(node.left)
            self.postorder_traversal(node.right)
            print(node.val, end=" ")

binary_tree = BinaryTree()
binary_tree.insert(1)
binary_tree.insert(2)
binary_tree.insert(3)
binary_tree.insert(4)
binary_tree.insert(5)
print("中序遍历:")
binary_tree.inorder_traversal(binary_tree.root)
print()
print("前序遍历:")
binary_tree.preorder_traversal(binary_tree.root)
print()
print("后序遍历:")
binary_tree.postorder_traversal(binary_tree.root)
print()

# 2.2 图
print("\n2.2 图:")

class Graph:
    def __init__(self):
        self.adjacency_list = {}
    
    def add_vertex(self, vertex):
        if vertex not in self.adjacency_list:
            self.adjacency_list[vertex] = []
    
    def add_edge(self, vertex1, vertex2):
        if vertex1 in self.adjacency_list:
            self.adjacency_list[vertex1].append(vertex2)
        if vertex2 in self.adjacency_list:
            self.adjacency_list[vertex2].append(vertex1)
    
    def print_graph(self):
        for vertex in self.adjacency_list:
            print(f"{vertex}: {self.adjacency_list[vertex]}")
    
    def bfs(self, start_vertex):
        visited = set()
        queue = [start_vertex]
        visited.add(start_vertex)
        
        while queue:
            current = queue.pop(0)
            print(current, end=" ")
            
            for neighbor in self.adjacency_list[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        print()
    
    def dfs(self, start_vertex, visited=None):
        if visited is None:
            visited = set()
        
        visited.add(start_vertex)
        print(start_vertex, end=" ")
        
        for neighbor in self.adjacency_list[start_vertex]:
            if neighbor not in visited:
                self.dfs(neighbor, visited)

graph = Graph()
graph.add_vertex("A")
graph.add_vertex("B")
graph.add_vertex("C")
graph.add_vertex("D")
graph.add_vertex("E")
graph.add_edge("A", "B")
graph.add_edge("A", "C")
graph.add_edge("B", "D")
graph.add_edge("C", "E")
graph.add_edge("D", "E")
print("图的邻接表:")
graph.print_graph()
print("广度优先搜索:")
graph.bfs("A")
print("深度优先搜索:")
graph.dfs("A")
print()

# 3. 排序算法
print("\n=== 3. 排序算法 ===")

# 3.1 冒泡排序
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
                swapped = True
        if not swapped:
            break
    return arr

# 3.2 选择排序
def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i+1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr

# 3.3 插入排序
def insertion_sort(arr):
    n = len(arr)
    for i in range(1, n):
        key = arr[i]
        j = i-1
        while j >=0 and key < arr[j]:
            arr[j+1] = arr[j]
            j -= 1
        arr[j+1] = key
    return arr

# 3.4 快速排序
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr)//2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

# 3.5 归并排序
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr)//2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# 测试排序算法
test_arr = [64, 34, 25, 12, 22, 11, 90]
print(f"原始数组: {test_arr}")
print(f"冒泡排序: {bubble_sort(test_arr.copy())}")
print(f"选择排序: {selection_sort(test_arr.copy())}")
print(f"插入排序: {insertion_sort(test_arr.copy())}")
print(f"快速排序: {quick_sort(test_arr.copy())}")
print(f"归并排序: {merge_sort(test_arr.copy())}")

# 4. 搜索算法
print("\n=== 4. 搜索算法 ===")

# 4.1 线性搜索
def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1

# 4.2 二分搜索
def binary_search(arr, target):
    left, right = 0, len(arr)-1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# 测试搜索算法
sorted_arr = [11, 12, 22, 25, 34, 64, 90]
target = 25
print(f"线性搜索 {target} 在 {sorted_arr} 中的索引: {linear_search(sorted_arr, target)}")
print(f"二分搜索 {target} 在 {sorted_arr} 中的索引: {binary_search(sorted_arr, target)}")

# 5. 动态规划
print("\n=== 5. 动态规划 ===")

# 5.1 斐波那契数列
def fibonacci(n):
    if n <= 1:
        return n
    dp = [0] * (n+1)
    dp[1] = 1
    for i in range(2, n+1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]

# 5.2 爬楼梯
def climb_stairs(n):
    if n <= 2:
        return n
    dp = [0] * (n+1)
    dp[1] = 1
    dp[2] = 2
    for i in range(3, n+1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]

# 5.3 最长公共子序列
def longest_common_subsequence(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

# 5.4 背包问题
def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(1, capacity+1):
            if weights[i-1] <= j:
                dp[i][j] = max(dp[i-1][j], dp[i-1][j-weights[i-1]] + values[i-1])
            else:
                dp[i][j] = dp[i-1][j]
    return dp[n][capacity]

# 测试动态规划算法
print(f"斐波那契数列第10项: {fibonacci(10)}")
print(f"爬10阶楼梯的方法数: {climb_stairs(10)}")
print(f"最长公共子序列长度: {longest_common_subsequence('ABCBDAB', 'BDCAB')}")
weights = [1, 3, 4, 5]
values = [1, 4, 5, 7]
capacity = 7
print(f"背包问题最大价值: {knapsack(weights, values, capacity)}")

# 6. 贪心算法
print("\n=== 6. 贪心算法 ===")

# 6.1 活动选择问题
def activity_selection(activities):
    activities.sort(key=lambda x: x[1])
    selected = [activities[0]]
    last_end = activities[0][1]
    
    for i in range(1, len(activities)):
        if activities[i][0] >= last_end:
            selected.append(activities[i])
            last_end = activities[i][1]
    return selected

# 6.2 找零钱问题
def coin_change(coins, amount):
    coins.sort(reverse=True)
    count = 0
    for coin in coins:
        if amount >= coin:
            count += amount // coin
            amount = amount % coin
    return count if amount == 0 else -1

# 测试贪心算法
activities = [(1, 4), (3, 5), (0, 6), (5, 7), (3, 9), (5, 9), (6, 10), (8, 11), (8, 12), (2, 14), (12, 16)]
selected_activities = activity_selection(activities)
print(f"选择的活动: {selected_activities}")

coins = [1, 5, 10, 25]
amount = 37
print(f"找零钱 {amount} 需要的硬币数: {coin_change(coins, amount)}")

print("\n=== 数据结构与算法示例代码结束 ===")
