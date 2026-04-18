#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第26天：链表
数据结构与算法学习示例
内容：单链表、双链表、循环链表
"""

print("=== 第26天：链表 ===")

# 1. 链表概述
print("\n1. 链表概述")

print("链表是一种线性数据结构，元素通过指针链接")
print("- 链表的优点：插入和删除操作效率高")
print("- 链表的缺点：访问元素需要遍历")

# 2. 单链表
print("\n2. 单链表")

print("单链表是链表的基本形式，每个节点只有一个指向下一个节点的指针")

class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class SinglyLinkedList:
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
    
    def insert_after(self, prev_node, data):
        """在指定节点后插入元素"""
        if not prev_node:
            print("前一个节点不能为空")
            return
        new_node = Node(data)
        new_node.next = prev_node.next
        prev_node.next = new_node
    
    def delete(self, key):
        """删除指定元素"""
        if not self.head:
            return
        if self.head.data == key:
            self.head = self.head.next
            return
        current = self.head
        while current.next:
            if current.next.data == key:
                current.next = current.next.next
                return
            current = current.next
    
    def delete_at_position(self, position):
        """删除指定位置的元素"""
        if not self.head:
            return
        if position == 0:
            self.head = self.head.next
            return
        current = self.head
        for _ in range(position - 1):
            if current.next:
                current = current.next
            else:
                return
        if current.next:
            current.next = current.next.next
    
    def search(self, key):
        """搜索指定元素"""
        current = self.head
        while current:
            if current.data == key:
                return current
            current = current.next
        return None
    
    def get_length(self):
        """获取链表长度"""
        length = 0
        current = self.head
        while current:
            length += 1
            current = current.next
        return length
    
    def print_list(self):
        """打印链表元素"""
        current = self.head
        elements = []
        while current:
            elements.append(current.data)
            current = current.next
        print(f"单链表元素: {elements}")

# 测试单链表
print("\n测试单链表操作:")
sll = SinglyLinkedList()
sll.insert_at_end(10)
sll.insert_at_end(20)
sll.insert_at_beginning(5)
sll.insert_after(sll.head.next, 15)
sll.print_list()
print(f"链表长度: {sll.get_length()}")
sll.delete(15)
sll.print_list()
sll.delete_at_position(1)
sll.print_list()
print(f"搜索元素20: {sll.search(20) is not None}")

# 3. 双链表
print("\n3. 双链表")

print("双链表每个节点有两个指针，分别指向前一个和后一个节点")

class DoublyNode:
    def __init__(self, data):
        self.data = data
        self.prev = None
        self.next = None

class DoublyLinkedList:
    def __init__(self):
        self.head = None
    
    def insert_at_beginning(self, data):
        """在链表开头插入元素"""
        new_node = DoublyNode(data)
        new_node.next = self.head
        if self.head:
            self.head.prev = new_node
        self.head = new_node
    
    def insert_at_end(self, data):
        """在链表末尾插入元素"""
        new_node = DoublyNode(data)
        if not self.head:
            self.head = new_node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
        new_node.prev = current
    
    def insert_after(self, prev_node, data):
        """在指定节点后插入元素"""
        if not prev_node:
            print("前一个节点不能为空")
            return
        new_node = DoublyNode(data)
        new_node.next = prev_node.next
        new_node.prev = prev_node
        if prev_node.next:
            prev_node.next.prev = new_node
        prev_node.next = new_node
    
    def delete(self, key):
        """删除指定元素"""
        if not self.head:
            return
        current = self.head
        while current:
            if current.data == key:
                if current.prev:
                    current.prev.next = current.next
                else:
                    self.head = current.next
                if current.next:
                    current.next.prev = current.prev
                return
            current = current.next
    
    def print_list(self):
        """打印链表元素"""
        current = self.head
        elements = []
        while current:
            elements.append(current.data)
            current = current.next
        print(f"双链表元素: {elements}")
    
    def print_list_reverse(self):
        """反向打印链表元素"""
        if not self.head:
            return
        current = self.head
        while current.next:
            current = current.next
        elements = []
        while current:
            elements.append(current.data)
            current = current.prev
        print(f"反向双链表元素: {elements}")

# 测试双链表
print("\n测试双链表操作:")
dll = DoublyLinkedList()
dll.insert_at_end(10)
dll.insert_at_end(20)
dll.insert_at_beginning(5)
dll.insert_after(dll.head.next, 15)
dll.print_list()
dll.print_list_reverse()
dll.delete(15)
dll.print_list()

# 4. 循环链表
print("\n4. 循环链表")

print("循环链表的尾节点指向头节点，形成一个环")

class CircularLinkedList:
    def __init__(self):
        self.head = None
    
    def insert_at_beginning(self, data):
        """在链表开头插入元素"""
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            new_node.next = self.head
            return
        current = self.head
        while current.next != self.head:
            current = current.next
        new_node.next = self.head
        current.next = new_node
        self.head = new_node
    
    def insert_at_end(self, data):
        """在链表末尾插入元素"""
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            new_node.next = self.head
            return
        current = self.head
        while current.next != self.head:
            current = current.next
        current.next = new_node
        new_node.next = self.head
    
    def delete(self, key):
        """删除指定元素"""
        if not self.head:
            return
        if self.head.data == key:
            current = self.head
            while current.next != self.head:
                current = current.next
            current.next = self.head.next
            self.head = self.head.next
            return
        current = self.head
        while current.next != self.head:
            if current.next.data == key:
                current.next = current.next.next
                return
            current = current.next
    
    def print_list(self):
        """打印链表元素"""
        if not self.head:
            return
        elements = []
        current = self.head
        while True:
            elements.append(current.data)
            current = current.next
            if current == self.head:
                break
        print(f"循环链表元素: {elements}")

# 测试循环链表
print("\n测试循环链表操作:")
cll = CircularLinkedList()
cll.insert_at_end(10)
cll.insert_at_end(20)
cll.insert_at_beginning(5)
cll.insert_at_end(30)
cll.print_list()
cll.delete(20)
cll.print_list()

# 5. 链表的时间复杂度
print("\n5. 链表的时间复杂度")

print("单链表的时间复杂度:")
print("- 访问: O(n)")
print("- 插入/删除(开头): O(1)")
print("- 插入/删除(中间/末尾): O(n)")

print("\n双链表的时间复杂度:")
print("- 访问: O(n)")
print("- 插入/删除(开头): O(1)")
print("- 插入/删除(中间/末尾): O(n)")
print("- 反向遍历: O(n)")

print("\n循环链表的时间复杂度:")
print("- 访问: O(n)")
print("- 插入/删除(开头): O(1)")
print("- 插入/删除(中间/末尾): O(n)")

# 6. 链表的应用场景
print("\n6. 链表的应用场景")

print("- 单链表: 简单的队列实现、内存管理")
print("- 双链表: 双向队列、浏览器历史记录、撤销/重做操作")
print("- 循环链表: 环形缓冲区、约瑟夫环问题")

# 7. 链表的优缺点
print("\n7. 链表的优缺点")

print("优点:")
print("- 动态大小，不需要预先分配内存")
print("- 插入和删除操作效率高")
print("- 不需要连续的内存空间")

print("\n缺点:")
print("- 访问元素需要遍历，时间复杂度为O(n)")
print("- 每个节点需要额外的指针空间")
print("- 缓存局部性差，不利于CPU缓存")

# 8. 练习
print("\n8. 练习")

# 练习1: 链表反转
print("练习1: 链表反转")
print("- 实现单链表的反转")
print("- 测试反转后的链表")

# 练习2: 检测链表中的环
print("\n练习2: 检测链表中的环")
print("- 使用快慢指针法检测链表中的环")
print("- 测试环的检测")

# 练习3: 合并两个有序链表
print("\n练习3: 合并两个有序链表")
print("- 实现合并两个有序单链表的函数")
print("- 测试合并后的链表")

# 练习4: 查找链表的中间节点
print("\n练习4: 查找链表的中间节点")
print("- 使用快慢指针法查找链表的中间节点")
print("- 测试中间节点的查找")

# 练习5: 删除链表的倒数第n个节点
print("\n练习5: 删除链表的倒数第n个节点")
print("- 实现删除链表倒数第n个节点的函数")
print("- 测试删除后的链表")

# 9. 代码优化建议
print("\n9. 代码优化建议")

print("- 使用哨兵节点简化边界条件处理")
print("- 注意内存管理，避免内存泄漏")
print("- 对于频繁的插入和删除操作，考虑使用双向链表")
print("- 对于需要频繁访问的场景，考虑使用数组或其他数据结构")

print("\n=== 第26天学习示例结束 ===")
