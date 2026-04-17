#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第42天：强化学习深入
机器学习基础学习示例
内容：强化学习的高级算法、策略梯度和深度强化学习
"""

print("=== 第42天：强化学习深入 ===")

# 1. 强化学习算法分类
print("\n1. 强化学习算法分类")

import numpy as np
import matplotlib.pyplot as plt

print("强化学习算法可以分为以下几类:")
print("1. 值函数方法:")
print("   - Q-learning")
print("   - SARSA")
print("   - DQN (Deep Q-Network)")
print("2. 策略梯度方法:")
print("   - REINFORCE")
print("   - Actor-Critic")
print("   - PPO (Proximal Policy Optimization)")
print("3. 基于模型的方法:")
print("   - Dyna-Q")
print("   - Model Predictive Control")

# 2. SARSA算法
print("\n2. SARSA算法")

print("SARSA (State-Action-Reward-State-Action) 是一种在线时序差分学习算法")
print("- 与Q-learning不同，SARSA是一种on-policy算法")
print("- 更新规则: Q(s,a) = Q(s,a) + α * [r + γ * Q(s',a') - Q(s,a)]")
print("  其中a'是根据当前策略选择的下一个动作")

# 实现SARSA算法
class SARSAAgent:
    """SARSA智能体"""
    
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
    
    def learn(self, state, action, reward, next_state, next_action, done):
        """更新Q表"""
        x, y = state
        next_x, next_y = next_state
        
        # SARSA更新规则
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * self.q_table[next_x, next_y, next_action]
        
        # 更新Q值
        self.q_table[x, y, action] = self.q_table[x, y, action] + self.learning_rate * (target - self.q_table[x, y, action])
        
        # 衰减探索率
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay

# 3. 策略梯度方法
print("\n3. 策略梯度方法")

print("策略梯度方法直接优化策略函数，而不是值函数")
print("- 策略函数π(a|s)：给定状态s，选择动作a的概率分布")
print("- 目标：最大化期望累积奖励")
print("- 常用算法：REINFORCE, Actor-Critic, PPO")

# 4. 深度Q网络 (DQN)
print("\n4. 深度Q网络 (DQN)")

print("DQN是将深度神经网络与Q-learning结合的算法")
print("- 使用神经网络近似Q函数")
print("- 经验回放：存储和重放经验，减少样本相关性")
print("- 目标网络：使用两个网络，一个用于选择动作，一个用于计算目标Q值")

# 5. OpenAI Gym环境
print("\n5. OpenAI Gym环境")

print("OpenAI Gym是一个用于开发和比较强化学习算法的工具包")
print("- 提供了各种环境，从简单的网格世界到复杂的Atari游戏")
print("- 标准化的接口：reset(), step(), render()")

# 示例：使用CartPole环境
"""
import gym

# 创建环境
env = gym.make('CartPole-v1')

# 重置环境
state = env.reset()
print(f"初始状态: {state}")

# 执行动作
done = False
while not done:
    # 渲染环境
    env.render()
    
    # 随机选择动作
    action = env.action_space.sample()
    
    # 执行动作
    next_state, reward, done, info = env.step(action)
    print(f"动作: {action}, 奖励: {reward}, 完成: {done}")
    
    # 更新状态
    state = next_state

# 关闭环境
env.close()
"""
print("示例代码：使用CartPole环境")
print("import gym")
print("env = gym.make('CartPole-v1')")
print("state = env.reset()")
print("done = False")
print("while not done:")
print("    env.render()")
print("    action = env.action_space.sample()")
print("    next_state, reward, done, info = env.step(action)")
print("    state = next_state")
print("env.close()")

# 6. 经验回放
print("\n6. 经验回放")

print("经验回放是DQN的重要组成部分")
print("- 存储智能体的经验 (s, a, r, s', done) 到回放缓冲区")
print("- 从缓冲区中随机采样小批量经验进行学习")
print("- 减少样本之间的相关性，提高训练稳定性")

# 实现经验回放缓冲区
class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        """存储经验"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """随机采样经验"""
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in batch])
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def __len__(self):
        """返回缓冲区大小"""
        return len(self.buffer)

# 7. 深度Q网络实现
print("\n7. 深度Q网络实现")

print("使用PyTorch实现DQN")
print("- 神经网络结构：输入状态，输出每个动作的Q值")
print("- 训练过程：使用经验回放和目标网络")

# 示例：DQN实现
"""
import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    """深度Q网络"""
    
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    """DQN智能体"""
    
    def __init__(self, state_size, action_size, learning_rate=0.001, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.995, exploration_min=0.01, batch_size=64, replay_buffer_capacity=10000):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.exploration_min = exploration_min
        self.batch_size = batch_size
        
        # 创建Q网络和目标网络
        self.q_network = DQN(state_size, action_size)
        self.target_network = DQN(state_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(replay_buffer_capacity)
    
    def choose_action(self, state):
        """选择动作"""
        if np.random.rand() < self.exploration_rate:
            return np.random.randint(self.action_size)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state)
            return np.argmax(q_values.cpu().numpy())
    
    def learn(self):
        """学习"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # 采样经验
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # 转换为张量
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # 计算当前Q值
        current_q = self.q_network(states).gather(1, actions)
        
        # 计算目标Q值
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.discount_factor * next_q
        
        # 计算损失
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 衰减探索率
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay
    
    def update_target_network(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())
"""
print("示例代码：DQN实现")
print("class DQN(nn.Module):")
print("    def __init__(self, state_size, action_size):")
print("        super(DQN, self).__init__()")
print("        self.fc1 = nn.Linear(state_size, 64)")
print("        self.fc2 = nn.Linear(64, 64)")
print("        self.fc3 = nn.Linear(64, action_size)")
print("    ")
print("    def forward(self, x):")
print("        x = torch.relu(self.fc1(x))")
print("        x = torch.relu(self.fc2(x))")
print("        x = self.fc3(x)")
print("        return x")

# 8. Actor-Critic算法
print("\n8. Actor-Critic算法")

print("Actor-Critic是一种结合值函数和策略梯度的算法")
print("- Actor：学习策略函数，选择动作")
print("- Critic：学习值函数，评估动作的价值")
print("- 优势函数：A(s,a) = Q(s,a) - V(s)")
print("- 策略梯度：∇θ J(θ) ∝ E[∇θ log π(a|s) * A(s,a)]")

# 9. 强化学习的挑战
print("\n9. 强化学习的挑战")

print("强化学习面临的主要挑战:")
print("1. 信用分配问题：延迟奖励")
print("2. 探索与利用的平衡")
print("3. 高维状态空间")
print("4. 样本效率低")
print("5. 稳定性和收敛性")

# 10. 练习
print("\n10. 练习")

# 练习1: 比较Q-learning和SARSA
print("练习1: 比较Q-learning和SARSA")
print("- 在同一个环境中实现Q-learning和SARSA")
print("- 比较两种算法的学习曲线")
print("- 分析两种算法的优缺点")

# 练习2: 实现DQN
print("\n练习2: 实现DQN")
print("- 使用PyTorch实现DQN")
print("- 在CartPole环境中训练智能体")
print("- 调整超参数，观察性能变化")

# 练习3: 尝试不同的环境
print("\n练习3: 尝试不同的环境")
print("- 尝试使用MountainCar-v0环境")
print("- 尝试使用LunarLander-v2环境")
print("- 比较不同环境的训练难度")

print("\n=== 第42天学习示例结束 ===")
