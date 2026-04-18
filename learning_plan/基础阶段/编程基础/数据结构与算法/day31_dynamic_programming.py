#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第31天：动态规划
数据结构与算法学习示例
内容：动态规划的基本概念、背包问题、最长公共子序列
"""

print("=== 第31天：动态规划 ===")

# 1. 动态规划概述
print("\n1. 动态规划概述")

print("动态规划是一种解决具有重叠子问题和最优子结构的问题的算法")
print("- 重叠子问题：问题可以分解为多个子问题，且子问题会重复出现")
print("- 最优子结构：问题的最优解可以由子问题的最优解构造")
print("- 状态转移方程：描述问题状态之间的关系")
print("- 记忆化搜索：存储子问题的解以避免重复计算")

# 2. 斐波那契数列
print("\n2. 斐波那契数列")

print("斐波那契数列是一个经典的动态规划问题")

# 递归实现
def fibonacci_recursive(n):
    """递归实现斐波那契数列"""
    if n <= 1:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)

# 动态规划实现
def fibonacci_dp(n):
    """动态规划实现斐波那契数列"""
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[0] = 0
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]

# 空间优化的动态规划实现
def fibonacci_dp_optimized(n):
    """空间优化的动态规划实现斐波那契数列"""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

# 测试斐波那契数列
print("测试斐波那契数列:")
for n in range(10):
    print(f"fib({n}) = {fibonacci_dp(n)}")

# 3. 爬楼梯问题
print("\n3. 爬楼梯问题")

print("爬楼梯问题：每次可以爬1或2级台阶，有多少种不同的方法爬到第n级台阶")


def climb_stairs(n):
    """爬楼梯问题"""
    if n <= 2:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    dp[2] = 2
    for i in range(3, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]

# 空间优化的爬楼梯问题
def climb_stairs_optimized(n):
    """空间优化的爬楼梯问题"""
    if n <= 2:
        return n
    a, b = 1, 2
    for _ in range(3, n + 1):
        a, b = b, a + b
    return b

# 测试爬楼梯问题
print("测试爬楼梯问题:")
for n in range(1, 10):
    print(f"爬到第{n}级台阶的方法数: {climb_stairs(n)}")

# 4. 背包问题
print("\n4. 背包问题")

print("背包问题是动态规划中的经典问题")

# 4.1 0-1背包问题
print("\n4.1 0-1背包问题")
print("0-1背包问题：每个物品只能选择一次，求在背包容量有限的情况下，能装下的最大价值")


def knapsack_01(weights, values, capacity):
    """0-1背包问题"""
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, capacity + 1):
            if weights[i - 1] <= j:
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weights[i - 1]] + values[i - 1])
            else:
                dp[i][j] = dp[i - 1][j]
    return dp[n][capacity]

# 测试0-1背包问题
print("测试0-1背包问题:")
weights = [2, 3, 4, 5]
values = [3, 4, 5, 6]
capacity = 8
print(f"物品重量: {weights}")
print(f"物品价值: {values}")
print(f"背包容量: {capacity}")
print(f"最大价值: {knapsack_01(weights, values, capacity)}")

# 4.2 完全背包问题
print("\n4.2 完全背包问题")
print("完全背包问题：每个物品可以选择多次，求在背包容量有限的情况下，能装下的最大价值")


def knapsack_complete(weights, values, capacity):
    """完全背包问题"""
    n = len(weights)
    dp = [0] * (capacity + 1)
    for i in range(n):
        for j in range(weights[i], capacity + 1):
            dp[j] = max(dp[j], dp[j - weights[i]] + values[i])
    return dp[capacity]

# 测试完全背包问题
print("测试完全背包问题:")
print(f"物品重量: {weights}")
print(f"物品价值: {values}")
print(f"背包容量: {capacity}")
print(f"最大价值: {knapsack_complete(weights, values, capacity)}")

# 5. 最长公共子序列
print("\n5. 最长公共子序列")

print("最长公共子序列（LCS）：找出两个字符串中最长的公共子序列")


def longest_common_subsequence(s1, s2):
    """最长公共子序列"""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]

# 测试最长公共子序列
print("测试最长公共子序列:")
s1 = "ABCBDAB"
s2 = "BDCABA"
print(f"字符串1: {s1}")
print(f"字符串2: {s2}")
print(f"最长公共子序列长度: {longest_common_subsequence(s1, s2)}")

# 6. 最长递增子序列
print("\n6. 最长递增子序列")

print("最长递增子序列（LIS）：找出数组中最长的递增子序列")


def longest_increasing_subsequence(arr):
    """最长递增子序列"""
    n = len(arr)
    dp = [1] * n
    for i in range(1, n):
        for j in range(i):
            if arr[i] > arr[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)

# 测试最长递增子序列
print("测试最长递增子序列:")
arr = [10, 9, 2, 5, 3, 7, 101, 18]
print(f"数组: {arr}")
print(f"最长递增子序列长度: {longest_increasing_subsequence(arr)}")

# 7. 编辑距离
print("\n7. 编辑距离")

print("编辑距离：将一个字符串转换为另一个字符串所需的最少操作数")


def edit_distance(s1, s2):
    """编辑距离"""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]

# 测试编辑距离
print("测试编辑距离:")
s1 = "horse"
s2 = "ros"
print(f"字符串1: {s1}")
print(f"字符串2: {s2}")
print(f"编辑距离: {edit_distance(s1, s2)}")

# 8. 最大子数组和
print("\n8. 最大子数组和")

print("最大子数组和：找出数组中连续子数组的最大和")


def max_subarray_sum(arr):
    """最大子数组和"""
    if not arr:
        return 0
    max_current = max_global = arr[0]
    for i in range(1, len(arr)):
        max_current = max(arr[i], max_current + arr[i])
        if max_current > max_global:
            max_global = max_current
    return max_global

# 测试最大子数组和
print("测试最大子数组和:")
arr = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
print(f"数组: {arr}")
print(f"最大子数组和: {max_subarray_sum(arr)}")

# 9. 动态规划的时间复杂度分析
print("\n9. 动态规划的时间复杂度分析")

print("各动态规划问题的时间复杂度:")
print("- 斐波那契数列: O(n)")
print("- 爬楼梯问题: O(n)")
print("- 0-1背包问题: O(n * capacity)")
print("- 完全背包问题: O(n * capacity)")
print("- 最长公共子序列: O(m * n)")
print("- 最长递增子序列: O(n²)")
print("- 编辑距离: O(m * n)")
print("- 最大子数组和: O(n)")

# 10. 动态规划的空间复杂度分析
print("\n10. 动态规划的空间复杂度分析")

print("各动态规划问题的空间复杂度:")
print("- 斐波那契数列: O(n) 或 O(1)（优化后）")
print("- 爬楼梯问题: O(n) 或 O(1)（优化后）")
print("- 0-1背包问题: O(n * capacity) 或 O(capacity)（优化后）")
print("- 完全背包问题: O(capacity)")
print("- 最长公共子序列: O(m * n)")
print("- 最长递增子序列: O(n)")
print("- 编辑距离: O(m * n) 或 O(min(m, n))（优化后）")
print("- 最大子数组和: O(1)")

# 11. 动态规划的应用场景
print("\n11. 动态规划的应用场景")

print("动态规划的应用场景:")
print("- 优化问题：如背包问题、最短路径问题")
print("- 字符串问题：如最长公共子序列、编辑距离")
print("- 序列问题：如最长递增子序列、最大子数组和")
print("- 计数问题：如爬楼梯问题、组合问题")
print("- 游戏问题：如博弈论中的最优策略")

# 12. 动态规划的解题步骤
print("\n12. 动态规划的解题步骤")

print("动态规划的解题步骤:")
print("1. 定义状态：确定问题的状态表示")
print("2. 确定状态转移方程：描述状态之间的关系")
print("3. 初始化：设置初始状态的值")
print("4. 计算顺序：确定计算状态的顺序")
print("5. 求解目标：根据状态计算最终结果")

# 13. 练习
print("\n13. 练习")

# 练习1: 实现分割等和子集
print("练习1: 实现分割等和子集")
print("- 给定一个非负整数数组，判断是否可以将数组分割成两个和相等的子集")
print("- 提示：使用0-1背包思想")

# 练习2: 实现零钱兑换
print("\n练习2: 实现零钱兑换")
print("- 给定不同面额的硬币和一个总金额，计算凑成总金额所需的最少硬币数")
print("- 提示：使用动态规划，状态表示为凑成金额i所需的最少硬币数")

# 练习3: 实现打家劫舍
print("\n练习3: 实现打家劫舍")
print("- 给定一个数组，代表每个房屋的金额，不能抢劫相邻的房屋，求最大抢劫金额")
print("- 提示：使用动态规划，状态表示为前i个房屋的最大抢劫金额")

# 练习4: 实现买卖股票的最佳时机
print("\n练习4: 实现买卖股票的最佳时机")
print("- 给定一个数组，代表每天的股票价格，求买卖一次的最大利润")
print("- 提示：使用动态规划，状态表示为前i天的最低价格和最大利润")

# 练习5: 实现最长回文子串
print("\n练习5: 实现最长回文子串")
print("- 给定一个字符串，找出最长的回文子串")
print("- 提示：使用动态规划，状态表示为子串是否是回文")

print("\n=== 第31天学习示例结束 ===")
