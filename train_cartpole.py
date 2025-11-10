import random
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from dqn_agent import DQNAgent
from replay_buffer import ReplayBuffer


def train_dqn():
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Corrected print statement
    print(f"state dimension: {state_dim}, action dimension: {action_dim}")

    agent = DQNAgent(state_dim, action_dim, lr=5e-4, epsilon_decay=0.99)
    memory = ReplayBuffer(50000)
    episodes = 1500
    batch_size = 128

    print("Starting training...")	

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        for t in range(500):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            memory.push(state, action, reward, next_state, done)
            agent.train_step(memory, batch_size)

            state = next_state
            total_reward += reward

            if done:
                break

        print(f"Episode {ep+1}, Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")

    torch.save(agent.model.state_dict(), "dqn_model.pth")


if __name__ == "__main__":
    train_dqn()
