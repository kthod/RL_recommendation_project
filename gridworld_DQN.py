import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import namedtuple, deque
import random
import matplotlib.pyplot as plt

# Define the neural network architecture
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.layers(x)

# Experience replay buffer
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Experience(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# Custom GridWorld environment
class GridWorld:
    def __init__(self, size):
        self.size = size
        self.agent_pos = [0, 0]

    def step(self, action):
        if action == 0:   # Move right
            self.agent_pos[1] = min(self.size - 1, self.agent_pos[1] + 1)
        elif action == 1: # Move left
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 2: # Move down
            self.agent_pos[0] = min(self.size - 1, self.agent_pos[0] + 1)
        elif action == 3: # Move up
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)

        reward = -1
        done = False
        if self.agent_pos == [self.size - 1, self.size - 1]:
            reward = 10
            done = True
        return self.agent_pos, reward, done

    def reset(self):
        self.agent_pos = [0, 0]
        return self.agent_pos

# Hyperparameters
lr = 0.001
gamma = 0.99
batch_size = 64
buffer_capacity = 10000
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 200
target_update_freq = 10

# Environment
env = GridWorld(3)
input_dim = 2
output_dim = 4

# Networks
policy_net = DQN(input_dim, output_dim)
target_net = DQN(input_dim, output_dim)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=lr)

# Replay buffer
buffer = ReplayBuffer(buffer_capacity)

def select_action(state, epsilon):
    print(state)
    print("++++++++++++")
    if random.random() < epsilon:
        return random.randint(0, output_dim - 1)
    return torch.argmax(policy_net(torch.FloatTensor(state))).item()

def train():
    if len(buffer) < batch_size:
        return

    experiences = buffer.sample(batch_size)
    batch = Experience(*zip(*experiences))

    state_batch = torch.FloatTensor(batch.state)
    action_batch = torch.LongTensor(batch.action).unsqueeze(1)
    reward_batch = torch.FloatTensor(batch.reward)
    next_state_batch = torch.FloatTensor(batch.next_state)
    done_batch = torch.BoolTensor(batch.done)

    print("state batch")
    print(state_batch)
    q_values = policy_net(state_batch).gather(1, action_batch)
    print("q values")
    print(q_values)
    next_q_values = target_net(next_state_batch).max(1)[0].detach()
    target_values = reward_batch + (gamma * next_q_values * ~done_batch)

    loss = nn.functional.smooth_l1_loss(q_values, target_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def main():
    num_episodes = 5
    rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-episode / epsilon_decay)
            action = select_action(state, epsilon)
            print(action)
            next_state, reward, done = env.step(action)
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            train()

        if episode % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())
        rewards.append(total_reward)
        print(f'Episode: {episode}, Total Reward: {total_reward}')
    
    plt.plot(range(num_episodes), rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    #plt.show()
if __name__ == '__main__':
    main()
