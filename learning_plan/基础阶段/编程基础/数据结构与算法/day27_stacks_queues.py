#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第27天：栈和队列
数据结构与算法学习示例
内容：栈的实现和应用、队列的实现和应用
"""

print("=== 第27天：栈和队列 ===")

# 1. 栈的概念
print("\n1. 栈的概念")

print("栈是一种后进先出（LIFO）的数据结构")
print("- 栈的基本操作：入栈（push）、出栈（pop）、查看栈顶（peek）")
print("- 栈的应用场景：表达式求值、括号匹配、函数调用栈")

# 2. 栈的实现
print("\n2. 栈的实现")

print("使用列表实现栈")

class Stack:
    def __init__(self):
        """初始化栈"""
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
    
    def print_stack(self):
        """打印栈的元素"""
        print(f"栈元素: {self.items}")

# 测试栈
print("\n测试栈操作:")
stack = Stack()
stack.push(10)
stack.push(20)
stack.push(30)
stack.print_stack()
print(f"栈的大小: {stack.size()}")
print(f"栈顶元素: {stack.peek()}")
print(f"出栈元素: {stack.pop()}")
stack.print_stack()
print(f"栈是否为空: {stack.is_empty()}")

# 3. 栈的应用：括号匹配
print("\n3. 栈的应用：括号匹配")

def is_valid_parentheses(s):
    """检查括号是否匹配"""
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    for char in s:
        if char in mapping:
            top_element = stack.pop() if stack else '#'
            if mapping[char] != top_element:
                return False
        else:
            stack.append(char)
    return not stack

# 测试括号匹配
print("测试括号匹配:")
test_cases = ["()", "()[]{}", "(]", "([)]", "{[]}"]
for test in test_cases:
    print(f"{test}: {is_valid_parentheses(test)}")

# 4. 栈的应用：表达式求值
print("\n4. 栈的应用：表达式求值")

def evaluate_expression(expression):
    """计算后缀表达式的值"""
    stack = []
    operators = {'+': lambda x, y: x + y, '-': lambda x, y: x - y, '*': lambda x, y: x * y, '/': lambda x, y: x / y}
    for token in expression.split():
        if token.isdigit():
            stack.append(int(token))
        elif token in operators:
            if len(stack) < 2:
                return "Error: 表达式无效"
            y = stack.pop()
            x = stack.pop()
            result = operators[token](x, y)
            stack.append(result)
    if len(stack) == 1:
        return stack[0]
    else:
        return "Error: 表达式无效"

# 测试表达式求值
print("测试后缀表达式求值:")
test_expressions = ["3 4 +", "5 2 * 3 +", "10 5 / 2 +"]
for expr in test_expressions:
    print(f"{expr} = {evaluate_expression(expr)}")

# 5. 队列的概念
print("\n5. 队列的概念")

print("队列是一种先进先出（FIFO）的数据结构")
print("- 队列的基本操作：入队（enqueue）、出队（dequeue）、查看队首（front）")
print("- 队列的应用场景：任务调度、广度优先搜索、缓冲区")

# 6. 队列的实现
print("\n6. 队列的实现")

print("使用列表实现队列")

class Queue:
    def __init__(self):
        """初始化队列"""
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
    
    def print_queue(self):
        """打印队列的元素"""
        print(f"队列元素: {self.items}")

# 测试队列
print("\n测试队列操作:")
queue = Queue()
queue.enqueue(10)
queue.enqueue(20)
queue.enqueue(30)
queue.print_queue()
print(f"队列的大小: {queue.size()}")
print(f"队首元素: {queue.front()}")
print(f"出队元素: {queue.dequeue()}")
queue.print_queue()
print(f"队列是否为空: {queue.is_empty()}")

# 7. 循环队列
print("\n7. 循环队列")

print("循环队列使用固定大小的数组实现，避免队列移动元素的开销")

class CircularQueue:
    def __init__(self, capacity):
        """初始化循环队列"""
        self.capacity = capacity
        self.items = [None] * capacity
        self.front = 0
        self.rear = 0
        self.size = 0
    
    def enqueue(self, item):
        """入队"""
        if self.is_full():
            return False
        self.items[self.rear] = item
        self.rear = (self.rear + 1) % self.capacity
        self.size += 1
        return True
    
    def dequeue(self):
        """出队"""
        if self.is_empty():
            return None
        item = self.items[self.front]
        self.front = (self.front + 1) % self.capacity
        self.size -= 1
        return item
    
    def front_element(self):
        """查看队首元素"""
        if self.is_empty():
            return None
        return self.items[self.front]
    
    def is_empty(self):
        """检查队列是否为空"""
        return self.size == 0
    
    def is_full(self):
        """检查队列是否已满"""
        return self.size == self.capacity
    
    def get_size(self):
        """返回队列的大小"""
        return self.size
    
    def print_queue(self):
        """打印队列的元素"""
        if self.is_empty():
            print("队列为空")
            return
        elements = []
        for i in range(self.size):
            index = (self.front + i) % self.capacity
            elements.append(self.items[index])
        print(f"循环队列元素: {elements}")

# 测试循环队列
print("\n测试循环队列操作:")
circular_queue = CircularQueue(5)
circular_queue.enqueue(10)
circular_queue.enqueue(20)
circular_queue.enqueue(30)
circular_queue.print_queue()
print(f"队列的大小: {circular_queue.get_size()}")
print(f"队首元素: {circular_queue.front_element()}")
print(f"出队元素: {circular_queue.dequeue()}")
circular_queue.print_queue()
circular_queue.enqueue(40)
circular_queue.enqueue(50)
circular_queue.enqueue(60)
circular_queue.print_queue()
print(f"队列是否已满: {circular_queue.is_full()}")

# 8. 双端队列
print("\n8. 双端队列")

print("双端队列允许在两端进行插入和删除操作")

class Deque:
    def __init__(self):
        """初始化双端队列"""
        self.items = []
    
    def add_front(self, item):
        """在队首添加元素"""
        self.items.insert(0, item)
    
    def add_rear(self, item):
        """在队尾添加元素"""
        self.items.append(item)
    
    def remove_front(self):
        """从队首删除元素"""
        if not self.is_empty():
            return self.items.pop(0)
        return None
    
    def remove_rear(self):
        """从队尾删除元素"""
        if not self.is_empty():
            return self.items.pop()
        return None
    
    def front(self):
        """查看队首元素"""
        if not self.is_empty():
            return self.items[0]
        return None
    
    def rear(self):
        """查看队尾元素"""
        if not self.is_empty():
            return self.items[-1]
        return None
    
    def is_empty(self):
        """检查队列是否为空"""
        return len(self.items) == 0
    
    def size(self):
        """返回队列的大小"""
        return len(self.items)
    
    def print_deque(self):
        """打印双端队列的元素"""
        print(f"双端队列元素: {self.items}")

# 测试双端队列
print("\n测试双端队列操作:")
deque = Deque()
deque.add_rear(10)
deque.add_rear(20)
deque.add_front(5)
deque.add_rear(30)
deque.print_deque()
print(f"双端队列的大小: {deque.size()}")
print(f"队首元素: {deque.front()}")
print(f"队尾元素: {deque.rear()}")
print(f"删除队首元素: {deque.remove_front()}")
print(f"删除队尾元素: {deque.remove_rear()}")
deque.print_deque()

# 9. 时间复杂度分析
print("\n9. 时间复杂度分析")

print("栈的时间复杂度:")
print("- 入栈: O(1)")
print("- 出栈: O(1)")
print("- 查看栈顶: O(1)")

print("\n队列的时间复杂度:")
print("- 入队: O(1)")
print("- 出队: O(n) (使用列表实现)")
print("- 查看队首: O(1)")

print("\n循环队列的时间复杂度:")
print("- 入队: O(1)")
print("- 出队: O(1)")
print("- 查看队首: O(1)")

print("\n双端队列的时间复杂度:")
print("- 队首添加/删除: O(n) (使用列表实现)")
print("- 队尾添加/删除: O(1)")
print("- 查看队首/队尾: O(1)")

# 10. 应用场景
print("\n10. 应用场景")

print("栈的应用场景:")
print("- 表达式求值和转换")
print("- 括号匹配")
print("- 函数调用栈")
print("- 回溯算法")
print("- 浏览器的前进/后退功能")

print("\n队列的应用场景:")
print("- 任务调度")
print("- 广度优先搜索")
print("- 缓冲区管理")
print("- 消息队列")
print("- 打印队列")

print("\n双端队列的应用场景:")
print("- 滑动窗口问题")
print("- 回文检查")
print("- 缓存实现")

# 11. 练习
print("\n11. 练习")

# 练习1: 用栈实现队列
print("练习1: 用栈实现队列")
print("- 使用两个栈实现队列的基本操作")
print("- 测试队列的入队和出队操作")

# 练习2: 用队列实现栈
print("\n练习2: 用队列实现栈")
print("- 使用两个队列实现栈的基本操作")
print("- 测试栈的入栈和出栈操作")

# 练习3: 最小栈
print("\n练习3: 最小栈")
print("- 实现一个支持获取最小值的栈")
print("- 测试获取最小值的操作")

# 练习4: 滑动窗口最大值
print("\n练习4: 滑动窗口最大值")
print("- 使用双端队列解决滑动窗口最大值问题")
print("- 测试不同滑动窗口大小的情况")

# 练习5: 队列的广度优先搜索应用
print("\n练习5: 队列的广度优先搜索应用")
print("- 使用队列实现二叉树的层序遍历")
print("- 测试层序遍历的结果")

print("\n=== 第27天学习示例结束 ===")
