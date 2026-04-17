import numpy as np
import matplotlib.pyplot as plt
import gym
from collections import deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 1. Q-learning算法实现
def q_learning_example():
    """Q-learning算法示例"""
    print("=== Q-learning算法示例 ===")
    
    # 创建CartPole环境
    env = gym.make('CartPole-v1')
    
    # 超参数
    episodes = 1000
    learning_rate = 0.1
    discount_factor = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    
    # 离散化状态空间
    def discretize_state(state, bins=(6, 12, 6, 12)):
        cart_position_bins = np.linspace(-2.4, 2.4, bins[0]-1)
        cart_velocity_bins = np.linspace(-3.0, 3.0, bins[1]-1)
        pole_angle_bins = np.linspace(-0.2094, 0.2094, bins[2]-1)
        pole_velocity_bins = np.linspace(-3.0, 3.0, bins[3]-1)
        
        state_discrete = (
            np.digitize(state[0], cart_position_bins),
            np.digitize(state[1], cart_velocity_bins),
            np.digitize(state[2], pole_angle_bins),
            np.digitize(state[3], pole_velocity_bins)
        )
        return state_discrete
    
    # 初始化Q表
    state_space_size = (6, 12, 6, 12)
    action_space_size = env.action_space.n
    Q = np.zeros(state_space_size + (action_space_size,))
    
    # 存储每个回合的奖励
    rewards = []
    
    # 训练
    for episode in range(episodes):
        state = env.reset()
        state = discretize_state(state[0]) if isinstance(state, tuple) else discretize_state(state)
        total_reward = 0
        done = False
        
        while not done:
            # ε-greedy策略
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])
            
            # 执行行动
            next_state, reward, done, _, _ = env.step(action)
            next_state = discretize_state(next_state)
            
            # Q-learning更新
            best_next_action = np.argmax(Q[next_state])
            Q[state][action] = Q[state][action] + learning_rate * (reward + discount_factor * Q[next_state][best_next_action] - Q[state][action])
            
            state = next_state
            total_reward += reward
        
        # 衰减ε
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards.append(total_reward)
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}: Average reward = {np.mean(rewards[-100:]):.2f}")
    
    # 可视化学习曲线
    plt.plot(rewards)
    plt.title('Q-learning学习曲线')
    plt.xlabel('回合')
    plt.ylabel('奖励')
    plt.savefig('q_learning_rewards.png')
    plt.close()
    
    # 测试训练好的策略
    print("\n测试训练好的策略:")
    state = env.reset()
    state = discretize_state(state[0]) if isinstance(state, tuple) else discretize_state(state)
    total_reward = 0
    done = False
    
    while not done:
        action = np.argmax(Q[state])
        next_state, reward, done, _, _ = env.step(action)
        state = discretize_state(next_state)
        total_reward += reward
    
    print(f"测试回合奖励: {total_reward}")
    env.close()
    
    return Q

# 2. SARSA算法实现
def sarsa_example():
    """SARSA算法示例"""
    print("\n=== SARSA算法示例 ===")
    
    # 创建MountainCar环境
    env = gym.make('MountainCar-v0')
    
    # 超参数
    episodes = 10000
    learning_rate = 0.1
    discount_factor = 0.99
    epsilon = 1.0
    epsilon_decay = 0.999
    epsilon_min = 0.01
    
    # 离散化状态空间
    def discretize_state(state, bins=(20, 20)):
        position_bins = np.linspace(-1.2, 0.6, bins[0]-1)
        velocity_bins = np.linspace(-0.07, 0.07, bins[1]-1)
        
        state_discrete = (
            np.digitize(state[0], position_bins),
            np.digitize(state[1], velocity_bins)
        )
        return state_discrete
    
    # 初始化Q表
    state_space_size = (20, 20)
    action_space_size = env.action_space.n
    Q = np.zeros(state_space_size + (action_space_size,))
    
    # 存储每个回合的奖励
    rewards = []
    
    # 训练
    for episode in range(episodes):
        state = env.reset()
        state = discretize_state(state[0]) if isinstance(state, tuple) else discretize_state(state)
        
        # ε-greedy策略选择初始行动
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])
        
        total_reward = 0
        done = False
        
        while not done:
            # 执行行动
            next_state, reward, done, _, _ = env.step(action)
            next_state = discretize_state(next_state)
            
            # ε-greedy策略选择下一个行动
            if np.random.random() < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(Q[next_state])
            
            # SARSA更新
            Q[state][action] = Q[state][action] + learning_rate * (reward + discount_factor * Q[next_state][next_action] - Q[state][action])
            
            state = next_state
            action = next_action
            total_reward += reward
        
        # 衰减ε
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards.append(total_reward)
        
        if (episode + 1) % 1000 == 0:
            print(f"Episode {episode+1}: Average reward = {np.mean(rewards[-1000:]):.2f}")
    
    # 可视化学习曲线
    plt.plot(rewards)
    plt.title('SARSA学习曲线')
    plt.xlabel('回合')
    plt.ylabel('奖励')
    plt.savefig('sarsa_rewards.png')
    plt.close()
    
    # 测试训练好的策略
    print("\n测试训练好的策略:")
    state = env.reset()
    state = discretize_state(state[0]) if isinstance(state, tuple) else discretize_state(state)
    action = np.argmax(Q[state])
    total_reward = 0
    done = False
    
    while not done:
        next_state, reward, done, _, _ = env.step(action)
        state = discretize_state(next_state)
        action = np.argmax(Q[state])
        total_reward += reward
    
    print(f"测试回合奖励: {total_reward}")
    env.close()
    
    return Q

# 3. DQN算法实现
class DQN(nn.Module):
    """DQN网络"""
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

def dqn_example():
    """DQN算法示例"""
    print("\n=== DQN算法示例 ===")
    
    # 创建CartPole环境
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # 超参数
    episodes = 1000
    batch_size = 64
    learning_rate = 0.001
    discount_factor = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    replay_buffer_capacity = 10000
    target_update_freq = 10
    
    # 初始化网络
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DQN(state_size, action_size).to(device)
    target_net = DQN(state_size, action_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(replay_buffer_capacity)
    
    # 存储每个回合的奖励
    rewards = []
    
    # 训练
    for episode in range(episodes):
        state = env.reset()
        state = state[0] if isinstance(state, tuple) else state
        state = torch.FloatTensor(state).to(device)
        total_reward = 0
        done = False
        
        while not done:
            # ε-greedy策略
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = policy_net(state).argmax().item()
            
            # 执行行动
            next_state, reward, done, _, _ = env.step(action)
            next_state = torch.FloatTensor(next_state).to(device)
            
            # 存储经验
            replay_buffer.add(state.cpu().numpy(), action, reward, next_state.cpu().numpy(), done)
            
            # 经验回放
            if len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                states, actions, rewards_batch, next_states, dones = zip(*batch)
                
                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).to(device)
                rewards_batch = torch.FloatTensor(rewards_batch).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.FloatTensor(dones).to(device)
                
                # 计算目标值
                with torch.no_grad():
                    target_q = rewards_batch + (1 - dones) * discount_factor * target_net(next_states).max(1)[0]
                
                # 计算当前Q值
                current_q = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
                
                # 计算损失
                loss = F.mse_loss(current_q, target_q)
                
                # 优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            state = next_state
            total_reward += reward
        
        # 衰减ε
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards.append(total_reward)
        
        # 更新目标网络
        if (episode + 1) % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}: Average reward = {np.mean(rewards[-100:]):.2f}")
    
    # 可视化学习曲线
    plt.plot(rewards)
    plt.title('DQN学习曲线')
    plt.xlabel('回合')
    plt.ylabel('奖励')
    plt.savefig('dqn_rewards.png')
    plt.close()
    
    # 测试训练好的策略
    print("\n测试训练好的策略:")
    state = env.reset()
    state = state[0] if isinstance(state, tuple) else state
    state = torch.FloatTensor(state).to(device)
    total_reward = 0
    done = False
    
    while not done:
        with torch.no_grad():
            action = policy_net(state).argmax().item()
        next_state, reward, done, _, _ = env.step(action)
        state = torch.FloatTensor(next_state).to(device)
        total_reward += reward
    
    print(f"测试回合奖励: {total_reward}")
    env.close()
    
    return policy_net

# 4. 策略梯度算法实现
class PolicyNetwork(nn.Module):
    """策略网络"""
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=1)

def policy_gradient_example():
    """策略梯度算法示例"""
    print("\n=== 策略梯度算法示例 ===")
    
    # 创建CartPole环境
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # 超参数
    episodes = 1000
    learning_rate = 0.001
    discount_factor = 0.99
    
    # 初始化网络
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = PolicyNetwork(state_size, action_size).to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    
    # 存储每个回合的奖励
    rewards = []
    
    # 训练
    for episode in range(episodes):
        state = env.reset()
        state = state[0] if isinstance(state, tuple) else state
        state = torch.FloatTensor(state).to(device)
        
        log_probs = []
        episode_rewards = []
        done = False
        
        while not done:
            # 选择行动
            action_probs = policy_net(state.unsqueeze(0))
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            # 执行行动
            next_state, reward, done, _, _ = env.step(action.item())
            next_state = torch.FloatTensor(next_state).to(device)
            
            log_probs.append(log_prob)
            episode_rewards.append(reward)
            state = next_state
        
        # 计算回报
        returns = []
        G = 0
        for r in reversed(episode_rewards):
            G = r + discount_factor * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(device)
        
        # 归一化回报
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # 计算损失
        loss = 0
        for log_prob, G in zip(log_probs, returns):
            loss -= log_prob * G
        
        # 优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_reward = sum(episode_rewards)
        rewards.append(total_reward)
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}: Average reward = {np.mean(rewards[-100:]):.2f}")
    
    # 可视化学习曲线
    plt.plot(rewards)
    plt.title('策略梯度学习曲线')
    plt.xlabel('回合')
    plt.ylabel('奖励')
    plt.savefig('policy_gradient_rewards.png')
    plt.close()
    
    # 测试训练好的策略
    print("\n测试训练好的策略:")
    state = env.reset()
    state = state[0] if isinstance(state, tuple) else state
    state = torch.FloatTensor(state).to(device)
    total_reward = 0
    done = False
    
    while not done:
        with torch.no_grad():
            action_probs = policy_net(state.unsqueeze(0))
            action = torch.argmax(action_probs).item()
        next_state, reward, done, _, _ = env.step(action)
        state = torch.FloatTensor(next_state).to(device)
        total_reward += reward
    
    print(f"测试回合奖励: {total_reward}")
    env.close()
    
    return policy_net

# 5. 演员-评论家方法实现
class ValueNetwork(nn.Module):
    """价值网络"""
    def __init__(self, state_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def actor_critic_example():
    """演员-评论家方法示例"""
    print("\n=== 演员-评论家方法示例 ===")
    
    # 创建LunarLander环境
    env = gym.make('LunarLander-v2')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # 超参数
    episodes = 1000
    learning_rate = 0.001
    discount_factor = 0.99
    
    # 初始化网络
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor_net = PolicyNetwork(state_size, action_size).to(device)
    critic_net = ValueNetwork(state_size).to(device)
    optimizer_actor = optim.Adam(actor_net.parameters(), lr=learning_rate)
    optimizer_critic = optim.Adam(critic_net.parameters(), lr=learning_rate)
    
    # 存储每个回合的奖励
    rewards = []
    
    # 训练
    for episode in range(episodes):
        state = env.reset()
        state = state[0] if isinstance(state, tuple) else state
        state = torch.FloatTensor(state).to(device)
        
        episode_rewards = []
        done = False
        
        while not done:
            # 选择行动
            action_probs = actor_net(state.unsqueeze(0))
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            # 执行行动
            next_state, reward, done, _, _ = env.step(action.item())
            next_state = torch.FloatTensor(next_state).to(device)
            
            # 计算价值
            value = critic_net(state.unsqueeze(0))
            next_value = critic_net(next_state.unsqueeze(0)) if not done else torch.tensor([[0.0]]).to(device)
            
            # 计算优势
            advantage = reward + discount_factor * next_value.item() - value.item()
            
            # 计算损失
            actor_loss = -log_prob * advantage
            critic_loss = F.mse_loss(value, torch.tensor([[reward + discount_factor * next_value.item()]]).to(device))
            
            # 优化
            optimizer_actor.zero_grad()
            optimizer_critic.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            optimizer_actor.step()
            optimizer_critic.step()
            
            episode_rewards.append(reward)
            state = next_state
        
        total_reward = sum(episode_rewards)
        rewards.append(total_reward)
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}: Average reward = {np.mean(rewards[-100:]):.2f}")
    
    # 可视化学习曲线
    plt.plot(rewards)
    plt.title('演员-评论家学习曲线')
    plt.xlabel('回合')
    plt.ylabel('奖励')
    plt.savefig('actor_critic_rewards.png')
    plt.close()
    
    # 测试训练好的策略
    print("\n测试训练好的策略:")
    state = env.reset()
    state = state[0] if isinstance(state, tuple) else state
    state = torch.FloatTensor(state).to(device)
    total_reward = 0
    done = False
    
    while not done:
        with torch.no_grad():
            action_probs = actor_net(state.unsqueeze(0))
            action = torch.argmax(action_probs).item()
        next_state, reward, done, _, _ = env.step(action)
        state = torch.FloatTensor(next_state).to(device)
        total_reward += reward
    
    print(f"测试回合奖励: {total_reward}")
    env.close()
    
    return actor_net, critic_net

if __name__ == "__main__":
    # 运行所有示例
    q_learning_example()
    sarsa_example()
    dqn_example()
    policy_gradient_example()
    actor_critic_example()
    
    print("\n所有示例运行完成！")