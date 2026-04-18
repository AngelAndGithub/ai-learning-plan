#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第25天：数据结构基础
数据结构与算法学习示例
内容：数据结构的基本概念、数组、链表、栈和队列
"""

print("=== 第25天：数据结构基础 ===")

# 1. 数据结构基本概念
print("\n1. 数据结构基本概念")

print("数据结构是组织和存储数据的方式")
print("- 线性数据结构：数组、链表、栈、队列")
print("- 非线性数据结构：树、图、堆")
print("- 数据结构的选择：根据操作的时间复杂度和空间复杂度")

# 2. 数组
print("\n2. 数组")

print("数组是一种线性数据结构，元素在内存中连续存储")

# 示例：数组操作
class Array:
    def __init__(self):
        self.items = []
    
    def insert(self, index, value):
        """在指定位置插入元素"""
        self.items.insert(index, value)
    
    def delete(self, index):
        """删除指定位置的元素"""
        if 0 <= index < len(self.items):
            return self.items.pop(index)
        return None
    
    def get(self, index):
        """获取指定位置的元素"""
        if 0 <= index < len(self.items):
            return self.items[index]
        return None
    
    def size(self):
        """返回数组大小"""
        return len(self.items)

# 测试数组
print("\n测试数组操作:")
array = Array()
array.insert(0, 10)
array.insert(1, 20)
array.insert(0, 5)
print(f"数组元素: {array.items}")
print(f"数组大小: {array.size()}")
print(f"获取索引1的元素: {array.get(1)}")
print(f"删除索引0的元素: {array.delete(0)}")
print(f"数组元素: {array.items}")

# 3. 链表
print("\n3. 链表")

print("链表是一种线性数据结构，元素通过指针链接")

# 示例：单链表
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
    
    def insert_at_beginning(self, data):
        """在链表开头插入元素"""
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node
    
    def insert_at_end(self, data):
        """在链表末尾插入元素"""
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
    
    def delete(self, data):
        """删除指定元素"""
        if not self.head:
            return
        if self.head.data == data:
            self.head = self.head.next
            return
        current = self.head
        while current.next:
            if current.next.data == data:
                current.next = current.next.next
                return
            current = current.next
    
    def print_list(self):
        """打印链表元素"""
        current = self.head
        elements = []
        while current:
            elements.append(current.data)
            current = current.next
        print(f"链表元素: {elements}")

# 测试链表
print("\n测试链表操作:")
linked_list = LinkedList()
linked_list.insert_at_beginning(10)
linked_list.insert_at_beginning(5)
linked_list.insert_at_end(20)
linked_list.print_list()
linked_list.delete(10)
linked_list.print_list()

# 4. 栈
print("\n4. 栈")

print("栈是一种后进先出（LIFO）的数据结构")

# 示例：栈
class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        """入栈"""
        self.items.append(item)
    
    def pop(self):
        """出栈"""
        if not self.is_empty():
            return self.items.pop()
        return None
    
    def peek(self):
        """查看栈顶元素"""
        if not self.is_empty():
            return self.items[-1]
        return None
    
    def is_empty(self):
        """检查栈是否为空"""
        return len(self.items) == 0
    
    def size(self):
        """返回栈的大小"""
        return len(self.items)

# 测试栈
print("\n测试栈操作:")
stack = Stack()
stack.push(10)
stack.push(20)
stack.push(30)
print(f"栈的大小: {stack.size()}")
print(f"栈顶元素: {stack.peek()}")
print(f"出栈元素: {stack.pop()}")
print(f"栈的大小: {stack.size()}")
print(f"栈是否为空: {stack.is_empty()}")

# 5. 队列
print("\n5. 队列")

print("队列是一种先进先出（FIFO）的数据结构")

# 示例：队列
class Queue:
    def __init__(self):
        self.items = []
    
    def enqueue(self, item):
        """入队"""
        self.items.append(item)
    
    def dequeue(self):
        """出队"""
        if not self.is_empty():
            return self.items.pop(0)
        return None
    
    def front(self):
        """查看队首元素"""
        if not self.is_empty():
            return self.items[0]
        return None
    
    def is_empty(self):
        """检查队列是否为空"""
        return len(self.items) == 0
    
    def size(self):
        """返回队列的大小"""
        return len(self.items)

# 测试队列
print("\n测试队列操作:")
queue = Queue()
queue.enqueue(10)
queue.enqueue(20)
queue.enqueue(30)
print(f"队列的大小: {queue.size()}")
print(f"队首元素: {queue.front()}")
print(f"出队元素: {queue.dequeue()}")
print(f"队列的大小: {queue.size()}")
print(f"队列是否为空: {queue.is_empty()}")

# 6. 时间复杂度分析
print("\n6. 时间复杂度分析")

print("不同数据结构的时间复杂度:")
print("- 数组:")
print("  - 访问: O(1)")
print("  - 插入/删除(中间): O(n)")
print("  - 插入/删除(末尾): O(1)")
print("- 链表:")
print("  - 访问: O(n)")
print("  - 插入/删除(开头): O(1)")
print("  - 插入/删除(中间/末尾): O(n)")
print("- 栈:")
print("  - 入栈: O(1)")
print("  - 出栈: O(1)")
print("  - 查看栈顶: O(1)")
print("- 队列:")
print("  - 入队: O(1)")
print("  - 出队: O(n) (使用列表实现)")
print("  - 查看队首: O(1)")

# 7. 空间复杂度分析
print("\n7. 空间复杂度分析")

print("不同数据结构的空间复杂度:")
print("- 数组: O(n)")
print("- 链表: O(n) (额外的指针空间)")
print("- 栈: O(n)")
print("- 队列: O(n)")

# 8. 应用场景
print("\n8. 应用场景")

print("不同数据结构的应用场景:")
print("- 数组: 随机访问频繁的场景，如存储固定大小的数据")
print("- 链表: 插入/删除频繁的场景，如动态数据结构")
print("- 栈: 回溯算法、表达式求值、函数调用栈")
print("- 队列: 任务调度、广度优先搜索、缓冲区")

# 9. 练习
print("\n9. 练习")

# 练习1: 实现双链表
print("练习1: 实现双链表")
print("- 实现双链表的插入、删除、遍历操作")
print("- 测试双链表的功能")

# 练习2: 实现循环队列
print("\n练习2: 实现循环队列")
print("- 使用固定大小的数组实现循环队列")
print("- 实现入队、出队、查看队首元素操作")

# 练习3: 实现栈的应用
print("\n练习3: 实现栈的应用")
print("- 使用栈实现括号匹配")
print("- 使用栈实现表达式求值")

# 练习4: 实现队列的应用
print("\n练习4: 实现队列的应用")
print("- 使用队列实现广度优先搜索")
print("- 使用队列实现任务调度")

print("\n=== 第25天学习示例结束 ===")
