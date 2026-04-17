#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第21天：Python面向对象编程
Python编程基础学习示例
内容：类、对象、继承、多态等面向对象编程概念
"""

print("=== 第21天：Python面向对象编程 ===")

# 1. 类的定义和使用
print("\n1. 类的定义和使用")

# 定义一个简单的类
class Person:
    """人"""
    
    # 类变量
    species = "Homo sapiens"
    
    # 初始化方法
    def __init__(self, name, age):
        # 实例变量
        self.name = name
        self.age = age
    
    # 实例方法
    def greet(self):
        """问候"""
        print(f"Hello, my name is {self.name}. I'm {self.age} years old.")
    
    # 类方法
    @classmethod
    def get_species(cls):
        """获取物种"""
        return cls.species
    
    # 静态方法
    @staticmethod
    def is_adult(age):
        """判断是否成年"""
        return age >= 18

# 创建对象
person1 = Person("Alice", 30)
person2 = Person("Bob", 25)

# 访问实例变量
print(f"person1.name = {person1.name}")
print(f"person1.age = {person1.age}")

# 调用实例方法
person1.greet()
person2.greet()

# 访问类变量
print(f"Person.species = {Person.species}")
print(f"person1.species = {person1.species}")

# 调用类方法
print(f"Person.get_species() = {Person.get_species()}")

# 调用静态方法
print(f"Person.is_adult(18) = {Person.is_adult(18)}")
print(f"Person.is_adult(17) = {Person.is_adult(17)}")

# 2. 继承
print("\n2. 继承")

# 定义父类
class Animal:
    """动物"""
    
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        """发出声音"""
        pass

# 定义子类
class Dog(Animal):
    """狗"""
    
    def speak(self):
        """发出声音"""
        return "Woof!"

class Cat(Animal):
    """猫"""
    
    def speak(self):
        """发出声音"""
        return "Meow!"

# 创建子类对象
dog = Dog("Rex")
cat = Cat("Whiskers")

print(f"Dog name: {dog.name}")
print(f"Dog speaks: {dog.speak()}")
print(f"Cat name: {cat.name}")
print(f"Cat speaks: {cat.speak()}")

# 3. 多态
print("\n3. 多态")

def make_animal_speak(animal):
    """让动物发出声音"""
    print(f"{animal.name} says: {animal.speak()}")

# 多态：不同类型的对象调用相同的方法
make_animal_speak(dog)
make_animal_speak(cat)

# 4. 方法重写
print("\n4. 方法重写")

class Vehicle:
    """交通工具"""
    
    def start(self):
        """启动"""
        print("Vehicle starting...")
    
    def stop(self):
        """停止"""
        print("Vehicle stopping...")

class Car(Vehicle):
    """汽车"""
    
    def start(self):
        """启动"""
        print("Car starting...")
    
    def stop(self):
        """停止"""
        print("Car stopping...")

class Bike(Vehicle):
    """自行车"""
    
    def start(self):
        """启动"""
        print("Bike starting...")

# 创建对象
vehicle = Vehicle()
car = Car()
bike = Bike()

print("Vehicle:")
vehicle.start()
vehicle.stop()

print("\nCar:")
car.start()
car.stop()

print("\nBike:")
bike.start()
bike.stop()  # 调用父类的stop方法

# 5. 封装
print("\n5. 封装")

class BankAccount:
    """银行账户"""
    
    def __init__(self, balance=0):
        # 私有变量（用双下划线开头）
        self.__balance = balance
    
    def deposit(self, amount):
        """存款"""
        if amount > 0:
            self.__balance += amount
            print(f"Deposited ${amount}. New balance: ${self.__balance}")
        else:
            print("Amount must be positive.")
    
    def withdraw(self, amount):
        """取款"""
        if 0 < amount <= self.__balance:
            self.__balance -= amount
            print(f"Withdrew ${amount}. New balance: ${self.__balance}")
        else:
            print("Invalid amount.")
    
    def get_balance(self):
        """获取余额"""
        return self.__balance

# 创建账户
account = BankAccount(1000)
print(f"Initial balance: ${account.get_balance()}")

# 存款
account.deposit(500)

# 取款
account.withdraw(200)

# 尝试直接访问私有变量（会失败）
# print(account.__balance)  # 这会引发错误

# 6. 特殊方法
print("\n6. 特殊方法")

class Vector:
    """向量"""
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    # 加法
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)
    
    # 减法
    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)
    
    # 乘法（标量）
    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)
    
    # 字符串表示
    def __str__(self):
        return f"Vector({self.x}, {self.y})"
    
    # 长度
    def __abs__(self):
        return (self.x**2 + self.y**2)**0.5
    
    # 相等性
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

# 创建向量
v1 = Vector(3, 4)
v2 = Vector(1, 2)

print(f"v1 = {v1}")
print(f"v2 = {v2}")
print(f"v1 + v2 = {v1 + v2}")
print(f"v1 - v2 = {v1 - v2}")
print(f"v1 * 2 = {v1 * 2}")
print(f"|v1| = {abs(v1)}")
print(f"v1 == v2: {v1 == v2}")

# 7. 抽象类
print("\n7. 抽象类")

from abc import ABC, abstractmethod

class Shape(ABC):
    """形状（抽象类）"""
    
    @abstractmethod
    def area(self):
        """计算面积"""
        pass
    
    @abstractmethod
    def perimeter(self):
        """计算周长"""
        pass

class Rectangle(Shape):
    """矩形"""
    
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        """计算面积"""
        return self.width * self.height
    
    def perimeter(self):
        """计算周长"""
        return 2 * (self.width + self.height)

class Circle(Shape):
    """圆形"""
    
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        """计算面积"""
        import math
        return math.pi * self.radius**2
    
    def perimeter(self):
        """计算周长"""
        import math
        return 2 * math.pi * self.radius

# 创建形状
rectangle = Rectangle(5, 3)
circle = Circle(4)

print(f"Rectangle area: {rectangle.area()}")
print(f"Rectangle perimeter: {rectangle.perimeter()}")
print(f"Circle area: {circle.area()}")
print(f"Circle perimeter: {circle.perimeter()}")

# 8. 组合
print("\n8. 组合")

class Engine:
    """发动机"""
    
    def start(self):
        """启动"""
        print("Engine starting...")
    
    def stop(self):
        """停止"""
        print("Engine stopping...")

class Car:
    """汽车"""
    
    def __init__(self):
        # 组合：Car 包含一个 Engine
        self.engine = Engine()
    
    def start(self):
        """启动"""
        print("Car starting...")
        self.engine.start()
    
    def stop(self):
        """停止"""
        print("Car stopping...")
        self.engine.stop()

# 创建汽车
car = Car()
car.start()
car.stop()

# 9. 练习
print("\n9. 练习")

# 练习1: 定义一个学生类
class Student:
    """学生"""
    
    def __init__(self, name, student_id, grades=None):
        self.name = name
        self.student_id = student_id
        self.grades = grades or []
    
    def add_grade(self, grade):
        """添加成绩"""
        self.grades.append(grade)
    
    def get_average(self):
        """计算平均成绩"""
        if not self.grades:
            return 0
        return sum(self.grades) / len(self.grades)
    
    def __str__(self):
        """字符串表示"""
        return f"Student(name='{self.name}', id={self.student_id}, average={self.get_average():.2f})"

# 创建学生
student = Student("Alice", 123)
student.add_grade(85)
student.add_grade(90)
student.add_grade(95)
print(student)

# 练习2: 定义一个图书类和图书馆类
class Book:
    """图书"""
    
    def __init__(self, title, author, isbn):
        self.title = title
        self.author = author
        self.isbn = isbn
        self.is_available = True
    
    def __str__(self):
        """字符串表示"""
        status = "Available" if self.is_available else "Borrowed"
        return f"Book('{self.title}', by {self.author}, ISBN: {self.isbn}, Status: {status})"

class Library:
    """图书馆"""
    
    def __init__(self):
        self.books = []
    
    def add_book(self, book):
        """添加图书"""
        self.books.append(book)
    
    def borrow_book(self, isbn):
        """借阅图书"""
        for book in self.books:
            if book.isbn == isbn and book.is_available:
                book.is_available = False
                return f"You borrowed '{book.title}'"
        return "Book not available"
    
    def return_book(self, isbn):
        """归还图书"""
        for book in self.books:
            if book.isbn == isbn and not book.is_available:
                book.is_available = True
                return f"You returned '{book.title}'"
        return "Book not found or already returned"
    
    def list_books(self):
        """列出所有图书"""
        for book in self.books:
            print(book)

# 创建图书馆和图书
library = Library()
book1 = Book("Python Programming", "John Doe", "1234567890")
book2 = Book("Data Science", "Jane Smith", "0987654321")
library.add_book(book1)
library.add_book(book2)

print("\nLibrary books:")
library.list_books()

print("\nBorrowing book:")
print(library.borrow_book("1234567890"))

print("\nLibrary books after borrowing:")
library.list_books()

print("\nReturning book:")
print(library.return_book("1234567890"))

print("\nLibrary books after returning:")
library.list_books()

print("\n=== 第21天学习示例结束 ===")
