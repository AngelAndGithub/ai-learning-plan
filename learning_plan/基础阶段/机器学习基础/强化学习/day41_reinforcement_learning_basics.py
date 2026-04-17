#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第41天：强化学习基础
机器学习基础学习示例
内容：强化学习的基本概念、马尔可夫决策过程和Q-learning
"""

print("=== 第41天：强化学习基础 ===")

# 1. 强化学习基本概念
print("\n1. 强化学习基本概念")

import numpy as np
import matplotlib.pyplot as plt

print("强化学习是一种机器学习方法，智能体通过与环境交互来学习最优策略")
print("- 智能体 (Agent): 学习和执行动作的实体")
print("- 环境 (Environment): 智能体交互的外部世界")
print("- 状态 (State): 环境的当前情况")
print("- 动作 (Action): 智能体可以执行的操作")
print("- 奖励 (Reward): 智能体执行动作后获得的反馈")
print("- 策略 (Policy): 从状态到动作的映射")
print("- 值函数 (Value Function): 状态或状态-动作对的长期价值")

# 2. 马尔可夫决策过程 (MDP)
print("\n2. 马尔可夫决策过程 (MDP)")

print("马尔可夫决策过程是强化学习的数学框架")
print("- 状态空间 S")
print("- 动作空间 A")
print("- 转移概率 P(s'|s,a)")
print("- 奖励函数 R(s,a,s')")
print("- 折扣因子 γ")

# 3. Q-learning算法
print("\n3. Q-learning算法")

print("Q-learning是一种基于值函数的强化学习算法")
print("- Q函数: Q(s,a) 表示在状态s下执行动作a的预期累积奖励")
print("- 更新规则: Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]")
print("  其中 α 是学习率，γ 是折扣因子")

# 4. 网格世界环境
print("\n4. 网格世界环境")

# 定义网格世界环境
class GridWorld:
    """简单的网格世界环境"""
    
    def __init__(self, size=5):
        self.size = size
        self.state = (0, 0)  # 起始位置
        self.goal = (size-1, size-1)  # 目标位置
        self.obstacles = [(1, 1), (2, 2), (3, 3)]  # 障碍物位置
    
    def reset(self):
        """重置环境"""
        self.state = (0, 0)
        return self.state
    
    def step(self, action):
        """执行动作"""
        x, y = self.state
        
        # 动作定义: 0=上, 1=右, 2=下, 3=左
        if action == 0:  # 上
            new_state = (max(0, x-1), y)
        elif action == 1:  # 右
            new_state = (x, min(self.size-1, y+1))
        elif action == 2:  # 下
            new_state = (min(self.size-1, x+1), y)
        elif action == 3:  # 左
            new_state = (x, max(0, y-1))
        else:
            new_state = (x, y)
        
        # 检查是否撞到障碍物
        if new_state in self.obstacles:
            return self.state, -1, False
        
        # 检查是否到达目标
        if new_state == self.goal:
            return new_state, 10, True
        
        # 普通移动
        reward = -0.1
        done = False
        
        self.state = new_state
        return new_state, reward, done
    
    def render(self):
        """渲染环境"""
        grid = [['.' for _ in range(self.size)] for _ in range(self.size)]
        grid[self.state[0]][self.state[1]] = 'A'  # 智能体位置
        grid[self.goal[0]][self.goal[1]] = 'G'  # 目标位置
        for obs in self.obstacles:
            grid[obs[0]][obs[1]] = 'X'  # 障碍物位置
        
        for row in grid:
            print(' '.join(row))
        print()

# 创建环境
env = GridWorld()
print("网格世界环境:")
env.render()

# 5. Q-learning实现
print("\n5. Q-learning实现")

class QLearningAgent:
    """Q-learning智能体"""
    
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.999, exploration_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.exploration_min = exploration_min
        
        # 初始化Q表
        self.q_table = np.zeros((state_size, state_size, action_size))
    
    def choose_action(self, state):
        """选择动作"""
        if np.random.rand() < self.exploration_rate:
            # 探索：随机选择动作
            return np.random.randint(self.action_size)
        else:
            # 利用：选择Q值最大的动作
            x, y = state
            return np.argmax(self.q_table[x, y, :])
    
    def learn(self, state, action, reward, next_state, done):
        """更新Q表"""
        x, y = state
        next_x, next_y = next_state
        
        # Q-learning更新规则
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * np.max(self.q_table[next_x, next_y, :])
        
        # 更新Q值
        self.q_table[x, y, action] = self.q_table[x, y, action] + self.learning_rate * (target - self.q_table[x, y, action])
        
        # 衰减探索率
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay

# 创建智能体
agent = QLearningAgent(state_size=5, action_size=4)

# 训练参数
n_episodes = 10000
max_steps = 100

# 训练智能体
total_rewards = []

for episode in range(n_episodes):
    state = env.reset()
    done = False
    episode_reward = 0
    
    for step in range(max_steps):
        # 选择动作
        action = agent.choose_action(state)
        
        # 执行动作
        next_state, reward, done = env.step(action)
        
        # 学习
        agent.learn(state, action, reward, next_state, done)
        
        # 更新状态和奖励
        state = next_state
        episode_reward += reward
        
        if done:
            break
    
    total_rewards.append(episode_reward)
    
    # 每1000个 episode 打印一次
    if (episode + 1) % 1000 == 0:
        avg_reward = np.mean(total_rewards[-1000:])
        print(f"Episode {episode+1}, Average Reward: {avg_reward:.2f}, Exploration Rate: {agent.exploration_rate:.4f}")

# 6. 训练结果分析
print("\n6. 训练结果分析")

# 绘制奖励曲线
plt.figure(figsize=(10, 6))
plt.plot(total_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Q-learning Training Progress')
plt.show()

# 绘制移动平均奖励曲线
window_size = 100
moving_average = np.convolve(total_rewards, np.ones(window_size)/window_size, mode='valid')

plt.figure(figsize=(10, 6))
plt.plot(moving_average)
plt.xlabel('Episode')
plt.ylabel('Moving Average Reward')
plt.title('Q-learning Training Progress (Moving Average)')
plt.show()

# 7. 测试智能体
print("\n7. 测试智能体")

# 测试智能体
state = env.reset()
done = False
steps = 0

print("测试智能体的路径:")
env.render()

while not done and steps < max_steps:
    # 选择动作（纯利用）
    action = np.argmax(agent.q_table[state[0], state[1], :])
    
    # 执行动作
    next_state, reward, done = env.step(action)
    
    # 更新状态
    state = next_state
    steps += 1
    
    # 渲染环境
    env.render()

print(f"到达目标所需步数: {steps}")

# 8. Q表可视化
print("\n8. Q表可视化")

# 打印Q表
print("Q表:")
action_names = ['上', '右', '下', '左']

for i in range(5):
    for j in range(5):
        print(f"状态 ({i},{j}):")
        for a in range(4):
            print(f"  {action_names[a]}: {agent.q_table[i, j, a]:.2f}")
        print()

# 9. 策略可视化
print("\n9. 策略可视化")

# 可视化策略
policy = np.argmax(agent.q_table, axis=2)
action_symbols = ['↑', '→', '↓', '←']

print("学习到的策略:")
for i in range(5):
    row = []
    for j in range(5):
        if (i, j) == env.goal:
            row.append('G')
        elif (i, j) in env.obstacles:
            row.append('X')
        else:
            row.append(action_symbols[policy[i, j]])
    print(' '.join(row))

# 10. 练习
print("\n10. 练习")

# 练习1: 修改环境参数
print("练习1: 修改环境参数")
print("尝试修改以下参数并观察结果:")
print("1. 学习率 (learning_rate)")
print("2. 折扣因子 (discount_factor)")
print("3. 探索率 (exploration_rate)")
print("4. 探索衰减率 (exploration_decay)")

# 练习2: 不同的奖励函数
print("\n练习2: 不同的奖励函数")
print("尝试修改奖励函数:")
print("1. 增加到达目标的奖励")
print("2. 增加撞到障碍物的惩罚")
print("3. 增加每步的惩罚")

# 练习3: 更大的网格世界
print("\n练习3: 更大的网格世界")
print("尝试创建一个更大的网格世界，并训练智能体")

# 示例：创建一个10x10的网格世界
def create_large_gridworld():
    env = GridWorld(size=10)
    # 添加更多障碍物
    env.obstacles = [(1, 1), (1, 2), (1, 3), (2, 1), (3, 1), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)]
    return env

print("创建了一个10x10的网格世界，包含更多障碍物")

print("\n=== 第41天学习示例结束 ===")
