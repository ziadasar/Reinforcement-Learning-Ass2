# train_all_dqn_single_file.py
"""
Complete DQN training for all 4 environments in one file - Perfect for Google Colab
"""

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import numpy as np
import random
from collections import deque
import time
import os
import sys
# Reset WandB login - add this at the TOP of your Colab cell
import wandb
with open("key.txt", "r") as f:
    api_key = f.read().strip()
wandb.login(key=api_key)
# ==================== DQN MODEL DEFINITIONS ====================

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=1e-3, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, enable_wandb=True):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
        if enable_wandb:
            wandb.watch(self.model, log="all", log_freq=10)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def train_step(self, memory, batch_size):
        if len(memory) < batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = memory.sample(batch_size)

        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_values = self.model(states).gather(1, actions)
        next_q_values = self.model(next_states).max(1)[0].unsqueeze(1)
        target_q = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.loss_fn(q_values, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss.item()

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
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self):
        return len(self.memory)

# ==================== TRAINING FUNCTIONS ====================

def train_dqn_cartpole():
    """Train DQN on CartPole-v1"""
    print("🚀 Training DQN on CartPole-v1...")
    
    wandb.init(project="dqn-all-environments", name="cartpole-dqn", reinit=False)
    
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"CartPole-v1 - State dim: {state_dim}, Action dim: {action_dim}")

    agent = DQNAgent(state_dim, action_dim, lr=5e-4, epsilon_decay=0.99)
    memory = ReplayBuffer(10000)
    episodes = 500
    batch_size = 64

    start_time = time.time()
    best_reward = 0

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        episode_loss = 0
        train_steps = 0

        for t in range(500):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            memory.push(state, action, reward, next_state, done)
            loss = agent.train_step(memory, batch_size)
            
            if loss > 0:
                episode_loss += loss
                train_steps += 1

            state = next_state
            total_reward += reward

            if done:
                break

        # Update best reward
        if total_reward > best_reward:
            best_reward = total_reward

        avg_loss = episode_loss / train_steps if train_steps > 0 else 0
        
        wandb.log({
            "environment": "cartpole",
            "episode": ep + 1,
            "total_reward": total_reward,
            "best_reward": best_reward,
            "epsilon": agent.epsilon,
            "avg_loss": avg_loss,
            "buffer_size": len(memory)
        })

        if (ep + 1) % 50 == 0:
            print(f"  Episode {ep+1}, Reward: {total_reward}, Best: {best_reward}, Epsilon: {agent.epsilon:.3f}")

    # Save model
    torch.save(agent.model.state_dict(), "dqn_cartpole_model.pth")
    
    training_time = time.time() - start_time
    print(f"✅ CartPole training completed in {training_time:.1f} seconds")
    print(f"   Best Reward: {best_reward}")
    
    wandb.finish()
    env.close()
    return agent

def train_dqn_acrobot():
    """Train DQN on Acrobot-v1"""
    print("🚀 Training DQN on Acrobot-v1...")
    
    wandb.init(project="dqn-all-environments", name="acrobot-dqn", reinit=True)
    
    env = gym.make("Acrobot-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"Acrobot-v1 - State dim: {state_dim}, Action dim: {action_dim}")

    agent = DQNAgent(state_dim, action_dim, lr=1e-3, epsilon_decay=0.995)
    memory = ReplayBuffer(20000)
    episodes = 1000
    batch_size = 64

    start_time = time.time()
    best_reward = -float('inf')

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        episode_loss = 0
        train_steps = 0

        for t in range(500):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            memory.push(state, action, reward, next_state, done)
            loss = agent.train_step(memory, batch_size)
            
            if loss > 0:
                episode_loss += loss
                train_steps += 1

            state = next_state
            total_reward += reward

            if done:
                break

        # Update best reward (Acrobot has negative rewards)
        if total_reward > best_reward:
            best_reward = total_reward

        avg_loss = episode_loss / train_steps if train_steps > 0 else 0
        
        wandb.log({
            "environment": "acrobot",
            "episode": ep + 1,
            "total_reward": total_reward,
            "best_reward": best_reward,
            "epsilon": agent.epsilon,
            "avg_loss": avg_loss,
            "buffer_size": len(memory)
        })

        if (ep + 1) % 100 == 0:
            print(f"  Episode {ep+1}, Reward: {total_reward}, Best: {best_reward}, Epsilon: {agent.epsilon:.3f}")

    torch.save(agent.model.state_dict(), "dqn_acrobot_model.pth")
    
    training_time = time.time() - start_time
    print(f"✅ Acrobot training completed in {training_time:.1f} seconds")
    print(f"   Best Reward: {best_reward}")
    
    wandb.finish()
    env.close()
    return agent

def train_dqn_mountaincar():
    """Train DQN on MountainCar-v0 with reward shaping"""
    print("🚀 Training DQN on MountainCar-v0...")
    
    wandb.init(project="dqn-all-environments", name="mountaincar-dqn", reinit=True)
    
    env = gym.make("MountainCar-v0")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"MountainCar-v0 - State dim: {state_dim}, Action dim: {action_dim}")

    agent = DQNAgent(state_dim, action_dim, lr=1e-3, epsilon_decay=0.999)
    memory = ReplayBuffer(50000)
    episodes = 2000
    batch_size = 64

    start_time = time.time()
    success_count = 0
    best_reward = -float('inf')

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        episode_loss = 0
        train_steps = 0

        for t in range(200):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Enhanced reward shaping for MountainCar
            position = next_state[0]
            velocity = next_state[1]
            
            # Reward shaping to encourage reaching the top
            shaped_reward = position + 0.1 * abs(velocity)
            
            if position >= 0.5:  # Success condition
                shaped_reward += 100
                success_count += 1
                
            memory.push(state, action, shaped_reward, next_state, done)
            loss = agent.train_step(memory, batch_size)
            
            if loss > 0:
                episode_loss += loss
                train_steps += 1

            state = next_state
            total_reward += shaped_reward

            if done:
                break

        # Update best reward
        if total_reward > best_reward:
            best_reward = total_reward

        avg_loss = episode_loss / train_steps if train_steps > 0 else 0
        success_rate = (success_count / (ep + 1)) * 100
        
        wandb.log({
            "environment": "mountaincar",
            "episode": ep + 1,
            "total_reward": total_reward,
            "best_reward": best_reward,
            "epsilon": agent.epsilon,
            "avg_loss": avg_loss,
            "buffer_size": len(memory),
            "success_rate": success_rate
        })

        if (ep + 1) % 200 == 0:
            print(f"  Episode {ep+1}, Reward: {total_reward:.1f}, Best: {best_reward:.1f}, Success Rate: {success_rate:.1f}%")

    torch.save(agent.model.state_dict(), "dqn_mountaincar_model.pth")
    
    training_time = time.time() - start_time
    final_success_rate = (success_count / episodes) * 100
    print(f"✅ MountainCar training completed in {training_time:.1f} seconds")
    print(f"   Best Reward: {best_reward:.1f}")
    print(f"   Final Success Rate: {final_success_rate:.1f}%")
    
    wandb.finish()
    env.close()
    return agent

def train_dqn_pendulum():
    """Train DQN on Pendulum-v1 with discretized actions"""
    print("🚀 Training DQN on Pendulum-v1...")
    
    wandb.init(project="dqn-all-environments", name="pendulum-dqn", reinit=True)
    
    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]
    
    # Discretize the continuous action space
    action_dim = 5
    action_values = np.linspace(-2.0, 2.0, action_dim)

    print(f"Pendulum-v1 - State dim: {state_dim}, Discrete actions: {action_dim}")

    agent = DQNAgent(state_dim, action_dim, lr=1e-3, epsilon_decay=0.995)
    memory = ReplayBuffer(20000)
    episodes = 1000
    batch_size = 64

    start_time = time.time()
    best_reward = -float('inf')

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        episode_loss = 0
        train_steps = 0

        for t in range(200):
            # Select discrete action
            discrete_action = agent.select_action(state)
            # Convert to continuous action
            action = [action_values[discrete_action]]
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Pendulum rewards are negative, normalize them
            normalized_reward = (reward + 16.273) / 16.273

            memory.push(state, discrete_action, normalized_reward, next_state, done)
            loss = agent.train_step(memory, batch_size)
            
            if loss > 0:
                episode_loss += loss
                train_steps += 1

            state = next_state
            total_reward += reward  # Use original reward for logging

            if done:
                break

        # Update best reward
        if total_reward > best_reward:
            best_reward = total_reward

        avg_loss = episode_loss / train_steps if train_steps > 0 else 0
        
        wandb.log({
            "environment": "pendulum",
            "episode": ep + 1,
            "total_reward": total_reward,
            "best_reward": best_reward,
            "epsilon": agent.epsilon,
            "avg_loss": avg_loss,
            "buffer_size": len(memory)
        })

        if (ep + 1) % 100 == 0:
            print(f"  Episode {ep+1}, Reward: {total_reward:.1f}, Best: {best_reward:.1f}, Epsilon: {agent.epsilon:.3f}")

    torch.save(agent.model.state_dict(), "dqn_pendulum_model.pth")
    
    training_time = time.time() - start_time
    print(f"✅ Pendulum training completed in {training_time:.1f} seconds")
    print(f"   Best Reward: {best_reward:.1f}")
    
    wandb.finish()
    env.close()
    return agent

# ==================== MAIN EXECUTION ====================

def train_all_environments():
    """Train DQN on all 4 environments sequentially"""
    print("🎯 Starting DQN Training on All Environments")
    print("=" * 60)
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"🎮 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚡ Using CPU")
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Train on all environments
    print("\n" + "="*50)
    train_dqn_cartpole()
    
    print("\n" + "="*50)
    train_dqn_acrobot()
    
    print("\n" + "="*50)
    train_dqn_mountaincar()
    
    print("\n" + "="*50)
    train_dqn_pendulum()
    
    print("\n" + "="*60)
    print("🎉 All DQN Training Completed!")
    print("📁 Model files saved:")
    print("   - dqn_cartpole_model.pth")
    print("   - dqn_acrobot_model.pth") 
    print("   - dqn_mountaincar_model.pth")
    print("   - dqn_pendulum_model.pth")

def show_usage():
    """Show how to use the script"""
    print("\n📖 USAGE:")
    print("  python train_all_dqn_single_file.py                    # Train ALL environments")
    print("  python train_all_dqn_single_file.py cartpole          # Train only CartPole")
    print("  python train_all_dqn_single_file.py acrobot           # Train only Acrobot")
    print("  python train_all_dqn_single_file.py mountaincar       # Train only MountainCar")
    print("  python train_all_dqn_single_file.py pendulum          # Train only Pendulum")
    print("\n🎯 Available environments: cartpole, acrobot, mountaincar, pendulum")

if __name__ == "__main__":
    print("🤖 DQN Training Script - All in One!")
    print("=" * 50)
    
   
        # Train all environments
    print("🎯 Training ALL environments sequentially...")
    train_all_environments()