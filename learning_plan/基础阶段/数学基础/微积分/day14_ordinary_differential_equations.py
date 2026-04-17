#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第14天：常微分方程
微积分学习示例
内容：常微分方程的求解和应用
"""

import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

print("=== 第14天：常微分方程 ===")

# 1. 一阶常微分方程
print("\n1. 一阶常微分方程")

# 示例：dy/dt = -y
# 解析解：y(t) = y0 * e^(-t)
def dydt(y, t):
    return -y

# 初始条件
y0 = 1.0
# 时间点
t = np.linspace(0, 5, 100)

# 数值求解
sol = integrate.odeint(dydt, y0, t)

# 解析解
analytical_solution = y0 * np.exp(-t)

print(f"一阶常微分方程 dy/dt = -y")
print(f"初始条件 y(0) = {y0}")
print(f"解析解: y(t) = {y0}e^(-t)")
print(f"数值解在 t=1 处: {sol[20][0]}")
print(f"解析解在 t=1 处: {analytical_solution[20]}")
print(f"误差: {abs(sol[20][0] - analytical_solution[20])}")

# 2. 可分离变量的微分方程
print("\n2. 可分离变量的微分方程")

# 示例：dy/dt = y(1 - y)
# 这是逻辑斯谛方程，解析解：y(t) = 1/(1 + (1/y0 - 1)e^(-t))
def logistic_equation(y, t):
    return y * (1 - y)

# 初始条件
y0 = 0.1
# 时间点
t = np.linspace(0, 10, 100)

# 数值求解
sol_logistic = integrate.odeint(logistic_equation, y0, t)

# 解析解
analytical_logistic = 1 / (1 + (1/y0 - 1)*np.exp(-t))

print(f"逻辑斯谛方程 dy/dt = y(1 - y)")
print(f"初始条件 y(0) = {y0}")
print(f"数值解在 t=5 处: {sol_logistic[50][0]}")
print(f"解析解在 t=5 处: {analytical_logistic[50]}")
print(f"误差: {abs(sol_logistic[50][0] - analytical_logistic[50])}")

# 3. 一阶线性微分方程
print("\n3. 一阶线性微分方程")

# 示例：dy/dt + y = t
# 解析解：y(t) = t - 1 + (y0 + 1)e^(-t)
def linear_equation(y, t):
    return t - y

# 初始条件
y0 = 0
# 时间点
t = np.linspace(0, 5, 100)

# 数值求解
sol_linear = integrate.odeint(linear_equation, y0, t)

# 解析解
analytical_linear = t - 1 + (y0 + 1)*np.exp(-t)

print(f"一阶线性微分方程 dy/dt + y = t")
print(f"初始条件 y(0) = {y0}")
print(f"数值解在 t=2 处: {sol_linear[40][0]}")
print(f"解析解在 t=2 处: {analytical_linear[40]}")
print(f"误差: {abs(sol_linear[40][0] - analytical_linear[40])}")

# 4. 二阶微分方程
print("\n4. 二阶微分方程")

# 示例：d²y/dt² + y = 0
# 这是简谐运动方程，解析解：y(t) = y0*cos(t) + v0*sin(t)
def harmonic_oscillator(y, t):
    return [y[1], -y[0]]  # y[0] = y, y[1] = dy/dt

# 初始条件 [y0, v0]
y0 = 1.0
v0 = 0.0
initial_conditions = [y0, v0]
# 时间点
t = np.linspace(0, 2*np.pi, 100)

# 数值求解
sol_harmonic = integrate.odeint(harmonic_oscillator, initial_conditions, t)

# 解析解
analytical_harmonic = y0*np.cos(t) + v0*np.sin(t)

print(f"二阶微分方程 d²y/dt² + y = 0")
print(f"初始条件 y(0) = {y0}, y'(0) = {v0}")
print(f"数值解在 t=π/2 处: {sol_harmonic[25][0]}")
print(f"解析解在 t=π/2 处: {analytical_harmonic[25]}")
print(f"误差: {abs(sol_harmonic[25][0] - analytical_harmonic[25])}")

# 5. 应用示例：人口增长模型
print("\n5. 应用示例：人口增长模型")

# 指数增长模型：dP/dt = rP
# 其中 r 是增长率
def exponential_growth(P, t, r):
    return r * P

# 参数
r = 0.1  # 10% 增长率
P0 = 100  # 初始人口

# 时间点
t = np.linspace(0, 20, 100)

# 数值求解
sol_population = integrate.odeint(exponential_growth, P0, t, args=(r,))

# 解析解
analytical_population = P0 * np.exp(r * t)

print(f"人口增长模型 dP/dt = {r}P")
print(f"初始人口 P(0) = {P0}")
print(f"20年后人口（数值解）: {sol_population[-1][0]}")
print(f"20年后人口（解析解）: {analytical_population[-1]}")

# 6. 应用示例：冷却定律
print("\n6. 应用示例：冷却定律")

# 牛顿冷却定律：dT/dt = -k(T - T_env)
# 其中 T 是物体温度，T_env 是环境温度，k 是常数
def cooling_law(T, t, k, T_env):
    return -k * (T - T_env)

# 参数
k = 0.1  # 冷却常数
T_env = 25  # 环境温度（摄氏度）
T0 = 100  # 初始温度（摄氏度）

# 时间点
t = np.linspace(0, 30, 100)

# 数值求解
sol_cooling = integrate.odeint(cooling_law, T0, t, args=(k, T_env))

# 解析解
analytical_cooling = T_env + (T0 - T_env) * np.exp(-k * t)

print(f"牛顿冷却定律 dT/dt = -{k}(T - {T_env})")
print(f"初始温度 T(0) = {T0}°C")
print(f"30分钟后温度（数值解）: {sol_cooling[-1][0]:.2f}°C")
print(f"30分钟后温度（解析解）: {analytical_cooling[-1]:.2f}°C")

# 7. 应用示例：RL电路
print("\n7. 应用示例：RL电路")

# RL电路方程：L di/dt + Ri = V
# 其中 L 是电感，R 是电阻，V 是电压
def RL_circuit(i, t, L, R, V):
    return (V - R * i) / L

# 参数
L = 1.0  # 电感（亨利）
R = 10.0  # 电阻（欧姆）
V = 5.0  # 电压（伏特）
i0 = 0  # 初始电流（安培）

# 时间点
t = np.linspace(0, 1, 100)

# 数值求解
sol_RL = integrate.odeint(RL_circuit, i0, t, args=(L, R, V))

# 解析解
analytical_RL = (V/R) * (1 - np.exp(-R*t/L))

print(f"RL电路方程 L di/dt + Ri = V")
print(f"参数: L={L}H, R={R}Ω, V={V}V")
print(f"初始电流 i(0) = {i0}A")
print(f"1秒后电流（数值解）: {sol_RL[-1][0]:.4f}A")
print(f"1秒后电流（解析解）: {analytical_RL[-1]:.4f}A")

# 8. 常微分方程的数值方法
print("\n8. 常微分方程的数值方法")

print("常用的数值方法:")
print("1. 欧拉法（简单但精度低）")
print("2. 龙格-库塔法（精度高，如RK4）")
print("3. 亚当斯法（多步方法，适合高精度需求）")

# 实现简单的欧拉法
def euler_method(dydt, y0, t):
    """欧拉法求解常微分方程"""
    y = np.zeros(len(t))
    y[0] = y0
    h = t[1] - t[0]
    
    for i in range(1, len(t)):
        y[i] = y[i-1] + h * dydt(y[i-1], t[i-1])
    
    return y

# 用欧拉法求解 dy/dt = -y
t_euler = np.linspace(0, 5, 100)
y_euler = euler_method(dydt, 1.0, t_euler)

print(f"\n欧拉法求解 dy/dt = -y 的结果在 t=1 处: {y_euler[20]}")
print(f"解析解: {np.exp(-t_euler[20])}")

print("\n=== 第14天学习示例结束 ===")
