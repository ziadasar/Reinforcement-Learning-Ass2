"""
Double DQN (DDQN) implementation with GPU support.
Usage:
    python ddqn.py

This script mirrors the DQN notebook code but uses a target network and Double DQN update rule.
It detects CUDA and places model and tensors on the selected device.
Optional: set USE_WANDB to True and put your WandB API key in key.txt (same folder) to log runs.
"""

import random
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import os
import argparse

# Optional: wandb integration
USE_WANDB = False
try:
    if USE_WANDB:
        import wandb
except Exception:
    USE_WANDB = False

# Device config: will prefer CUDA when available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed_all(42)


class DQNNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.memory)


class DDQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, tau=1.0, update_target_every=1000, enable_wandb=True):
        # hyperparams
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.tau = tau  # for soft update if used
        self.update_target_every = update_target_every

        # device from module-level configuration
        self.device = device

        # networks
        self.online = DQNNet(state_dim, action_dim).to(self.device)
        self.target = DQNNet(state_dim, action_dim).to(self.device)
        # initialize target same as online
        self.target.load_state_dict(self.online.state_dict())

        self.optimizer = optim.Adam(self.online.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # tracking steps for periodic target updates
        self.learn_step_counter = 0

        # wandb
        self.enable_wandb = enable_wandb and USE_WANDB
        if self.enable_wandb:
            wandb.watch(self.online, log="all", log_freq=100)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            qvals = self.online(state_t)
            return int(torch.argmax(qvals, dim=1).item())

    def soft_update(self, tau=None):
        # soft update target params: target = tau*online + (1-tau)*target
        if tau is None:
            tau = self.tau
        for target_param, online_param in zip(self.target.parameters(), self.online.parameters()):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)

    def hard_update(self):
        self.target.load_state_dict(self.online.state_dict())

    def train_step(self, memory, batch_size):
        if len(memory) < batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = memory.sample(batch_size)

        # tensors directly on device
        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Current Q estimates
        q_eval = self.online(states_t).gather(1, actions_t)  # Q(s,a)

        # Double DQN target calculation:
        # actions from online network (next state); values from target network
        with torch.no_grad():
            next_actions = torch.argmax(self.online(next_states_t), dim=1, keepdim=True)  # indices
            q_next = self.target(next_states_t).gather(1, next_actions)  # Q_target(next_state, argmax_online)
            q_target = rewards_t + (1 - dones_t) * self.gamma * q_next

        loss = self.loss_fn(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # update target periodically (hard update) or you can use soft_update
        self.learn_step_counter += 1
        if self.learn_step_counter % self.update_target_every == 0:
            self.hard_update()

        return loss.item()

    def save(self, path):
        torch.save(self.online.state_dict(), path)

    def load(self, path):
        # load to device
        self.online.load_state_dict(torch.load(path, map_location=self.device))
        self.target.load_state_dict(self.online.state_dict())
        self.online.to(self.device)
        self.target.to(self.device)


def train(env_name='CartPole-v1', episodes=500, batch_size=64, lr=5e-4, buffer_size=10000, seed=42, save_path='ddqn_model.pth'):
    env = gym.make(env_name)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DDQNAgent(state_dim, action_dim, lr=lr, update_target_every=1000)
    memory = ReplayBuffer(capacity=buffer_size)

    if USE_WANDB and agent.enable_wandb:
        # login from key.txt if present
        try:
            with open('key.txt', 'r') as f:
                api_key = f.read().strip()
            wandb.login(key=api_key)
            wandb.init(project='ddqn-project', name='ddqn_run')
        except Exception as e:
            print('WandB login failed, continuing without logging:', e)

    print('Starting DDQN training on', env_name)

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0.0
        ep_loss = 0.0
        train_steps = 0

        for t in range(1000):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            memory.push(state, action, reward, next_state, done)
            loss = agent.train_step(memory, batch_size)

            if loss > 0:
                ep_loss += loss
                train_steps += 1

            state = next_state
            total_reward += reward

            if done:
                break

        avg_loss = ep_loss / train_steps if train_steps > 0 else 0.0
        print(f"Episode {ep+1}/{episodes}  Reward: {total_reward:.2f}  Epsilon: {agent.epsilon:.4f}  AvgLoss: {avg_loss:.4f}")

        if USE_WANDB and agent.enable_wandb:
            wandb.log({
                'episode': ep+1,
                'reward': total_reward,
                'epsilon': agent.epsilon,
                'avg_loss': avg_loss,
                'buffer_size': len(memory)
            })

    # save final
    agent.save(save_path)
    print('Saved model to', save_path)
    if USE_WANDB and agent.enable_wandb:
        wandb.save(save_path)
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--episodes', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--save', type=str, default='ddqn_model.pth')
    args = parser.parse_args()

    train(env_name=args.env, episodes=args.episodes, batch_size=args.batch_size, lr=args.lr, save_path=args.save)
