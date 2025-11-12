# ddqn.py
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import wandb


class DDQNNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super(DDQNNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class DDQNAgent:
    """Double DQN agent with the same public API as `DQNAgent` in `models/dqn.py`.

    Methods:
    - select_action(state) -> int
    - train_step(memory, batch_size) -> loss (float)
    - save(path)
    - load(path)
    """

    def __init__(self, state_dim, action_dim, gamma=0.99, lr=1e-3,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 update_target_every=1000, enable_wandb=True):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.update_target_every = update_target_every

        # device-aware
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Informative print so user knows whether CUDA will be used
        print(f"DDQNAgent using device: {self.device} (CUDA available: {torch.cuda.is_available()})")

        # networks
        self.online = DDQNNet(state_dim, action_dim).to(self.device)
        self.target = DDQNNet(state_dim, action_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())

        self.optimizer = optim.Adam(self.online.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.learn_step_counter = 0

        self.enable_wandb = enable_wandb
        if self.enable_wandb:
            try:
                wandb.watch(self.online, log="all", log_freq=100)
            except Exception:
                pass

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            qvals = self.online(state_t)
            return int(torch.argmax(qvals, dim=1).item())

    def hard_update(self):
        self.target.load_state_dict(self.online.state_dict())

    def train_step(self, memory, batch_size):
        if len(memory) < batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = memory.sample(batch_size)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Q(s,a)
        q_eval = self.online(states_t).gather(1, actions_t)

        # Double DQN target: actions from online network, values from target network
        with torch.no_grad():
            next_actions = torch.argmax(self.online(next_states_t), dim=1, keepdim=True)
            q_next = self.target(next_states_t).gather(1, next_actions)
            q_target = rewards_t + (1 - dones_t) * self.gamma * q_next

        loss = self.loss_fn(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.learn_step_counter += 1
        if self.learn_step_counter % self.update_target_every == 0:
            self.hard_update()

        return loss.item()

    def save(self, path):
        torch.save(self.online.state_dict(), path)

    def load(self, path):
        self.online.load_state_dict(torch.load(path, map_location=self.device))
        self.target.load_state_dict(self.online.state_dict())
        self.online.to(self.device)
        self.target.to(self.device)


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
