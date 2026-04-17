#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第10天：求导法则
微积分学习示例
内容：各种求导法则的应用
"""

import numpy as np

print("=== 第10天：求导法则 ===")

# 1. 常数法则
print("\n1. 常数法则")

# 常数函数的导数为0
def constant_function(x):
    return 5

def derivative_constant(x):
    return 0

x = 2
print(f"常数函数 f(x) = 5 在 x={x} 处的导数: {derivative_constant(x)}")

# 2. 幂法则
print("\n2. 幂法则")

# 幂法则：d/dx [x^n] = n*x^(n-1)
def power_function(x, n):
    return x**n

def derivative_power(x, n):
    return n * x**(n-1)

x = 3
n = 4
print(f"幂函数 f(x) = x^{n} 在 x={x} 处的导数: {derivative_power(x, n)}")
print(f"验证: {n}*{x}^{n-1} = {n*x**(n-1)}")

# 3. 加减法则
print("\n3. 加减法则")

# 加减法则：d/dx [f(x) ± g(x)] = f'(x) ± g'(x)
def f(x):
    return x**2
def g(x):
    return 3*x
def h(x):
    return f(x) + g(x)

def derivative_h(x):
    return 2*x + 3

x = 2
print(f"函数 h(x) = x² + 3x 在 x={x} 处的导数: {derivative_h(x)}")
print(f"验证: f'(x) + g'(x) = 2*{x} + 3 = {2*x + 3}")

# 4. 乘积法则
print("\n4. 乘积法则")

# 乘积法则：d/dx [f(x)*g(x)] = f'(x)*g(x) + f(x)*g'(x)
def f(x):
    return x**2
def g(x):
    return np.sin(x)
def product_function(x):
    return f(x) * g(x)

def derivative_product(x):
    return 2*x*np.sin(x) + x**2*np.cos(x)

x = np.pi/2
print(f"函数 f(x)*g(x) = x²*sin(x) 在 x=π/2 处的导数: {derivative_product(x)}")
print(f"验证: f'(x)*g(x) + f(x)*g'(x) = 2*{x}*sin({x}) + {x}²*cos({x}) = {2*x*np.sin(x) + x**2*np.cos(x)}")

# 5. 商法则
print("\n5. 商法则")

# 商法则：d/dx [f(x)/g(x)] = [f'(x)*g(x) - f(x)*g'(x)] / g(x)²
def f(x):
    return x + 1
def g(x):
    return x - 1
def quotient_function(x):
    return f(x) / g(x)

def derivative_quotient(x):
    return (1*(x-1) - (x+1)*1) / (x-1)**2

x = 3
print(f"函数 f(x)/g(x) = (x+1)/(x-1) 在 x={x} 处的导数: {derivative_quotient(x)}")
print(f"验证: [f'(x)*g(x) - f(x)*g'(x)]/g(x)² = [1*({x}-1) - ({x}+1)*1]/({x}-1)² = {(1*(x-1) - (x+1)*1)/(x-1)**2}")

# 6. 链式法则
print("\n6. 链式法则")

# 链式法则：d/dx [f(g(x))] = f'(g(x)) * g'(x)
def g(x):
    return x**2
def f(u):
    return np.sin(u)
def composite_function(x):
    return f(g(x))

def derivative_composite(x):
    return np.cos(x**2) * 2*x

x = 1
print(f"复合函数 f(g(x)) = sin(x²) 在 x={x} 处的导数: {derivative_composite(x)}")
print(f"验证: f'(g(x)) * g'(x) = cos({x}²) * 2*{x} = {np.cos(x**2) * 2*x}")

# 7. 反函数求导法则
print("\n7. 反函数求导法则")

# 反函数求导法则：如果 y = f(x)，则 dx/dy = 1/(dy/dx)
def f(x):
    return x**3  # 反函数为 g(y) = y^(1/3)

def derivative_f(x):
    return 3*x**2

def derivative_inverse(y):
    x = y**(1/3)
    return 1/(3*x**2)

y = 8
x = 2  # 因为 f(2) = 8
print(f"函数 f(x) = x³ 的反函数在 y={y} 处的导数: {derivative_inverse(y)}")
print(f"验证: 1/f'(x) = 1/(3*{x}²) = 1/{3*x**2} = {1/(3*x**2)}")

# 8. 综合应用
print("\n8. 综合应用")

# 复杂函数的求导
def complex_function(x):
    return np.sin(x**2 + 2*x + 1) * np.exp(x)

def derivative_complex(x):
    # 使用乘积法则和链式法则
    # d/dx [sin(x²+2x+1) * e^x] = cos(x²+2x+1)*(2x+2)*e^x + sin(x²+2x+1)*e^x
    return np.cos(x**2 + 2*x + 1)*(2*x + 2)*np.exp(x) + np.sin(x**2 + 2*x + 1)*np.exp(x)

x = 0
print(f"复杂函数 sin(x²+2x+1)*e^x 在 x={x} 处的导数: {derivative_complex(x)}")

# 数值导数验证
def numerical_derivative(func, x, h=1e-6):
    return (func(x+h) - func(x-h))/(2*h)

numerical_result = numerical_derivative(complex_function, x)
print(f"数值导数验证: {numerical_result}")
print(f"误差: {abs(derivative_complex(x) - numerical_result)}")

# 9. 应用示例：优化问题
print("\n9. 应用示例：优化问题")

# 寻找函数的极值点
def objective_function(x):
    return x**3 - 6*x**2 + 9*x + 2

def derivative_objective(x):
    return 3*x**2 - 12*x + 9

# 解方程 3x² - 12x + 9 = 0
# 因式分解: 3(x-1)(x-3) = 0
critical_points = [1, 3]

for point in critical_points:
    print(f"临界点 x={point}, f(x)={objective_function(point)}")

# 二阶导数判断极值类型
def second_derivative_objective(x):
    return 6*x - 12

for point in critical_points:
    second_deriv = second_derivative_objective(point)
    if second_deriv > 0:
        print(f"x={point} 是极小值点")
    elif second_deriv < 0:
        print(f"x={point} 是极大值点")
    else:
        print(f"x={point} 需要进一步分析")

print("\n=== 第10天学习示例结束 ===")
