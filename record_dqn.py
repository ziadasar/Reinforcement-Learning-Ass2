import random
import numpy as np
import gymnasium as gym
import torch
import os
from datetime import datetime
from dqn_agent import DQNAgent

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Create a unique folder for videos
video_folder = f"videos/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(video_folder, exist_ok=True)

env = gym.make("CartPole-v1", render_mode="rgb_array")
env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)  # Increase max steps
env = gym.wrappers.RecordVideo(env, video_folder, episode_trigger=lambda e: True)

# Load the trained model
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = DQNAgent(state_dim, action_dim)
agent.model.load_state_dict(torch.load("dqn_model.pth"))
agent.epsilon = 0  # Disable exploration

# Record multiple episodes
for episode in range(3):  # Record 3 episodes
    state, _ = env.reset()
    done = False
    while not done:
        action = agent.select_action(state)
        state, _, terminated, truncated, _ = env.step(action)
        print("terminated", terminated, "truncated", truncated)
        done = terminated or truncated

# Close the environment to avoid cleanup errors
env.close()


