# mountaincar_ddqn_train.py
import gymnasium as gym
import torch
import wandb
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.ddqn import DDQNAgent, ReplayBuffer
import numpy as np

def train_ddqn_mountaincar():
    wandb.init(project="ddqn-mountaincar", name="ddqn_mountaincar_experiment")

    env = gym.make("MountainCar-v0")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"MountainCar-v0 - State dim: {state_dim}, Action dim: {action_dim}")

    agent = DDQNAgent(state_dim, action_dim, lr=1e-3, epsilon_decay=0.999)
    memory = ReplayBuffer(50000)
    episodes = 2000
    batch_size = 64

    print("Starting DDQN MountainCar training...")

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        episode_loss = 0
        train_steps = 0

        for t in range(500):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            position = next_state[0]
            velocity = next_state[1]
            reward = position + 0.1 * abs(velocity)
            if position >= 0.5:
                reward += 100

            memory.push(state, action, reward, next_state, done)
            loss = agent.train_step(memory, batch_size)
            
            if loss > 0:
                episode_loss += loss
                train_steps += 1

            state = next_state
            total_reward += reward

            if done:
                break

        if (ep + 1) % 100 == 0:
            print(f"Episode {ep+1}, Reward: {total_reward:.1f}, Epsilon: {agent.epsilon:.3f}")

    model_path = "ddqn_mountaincar_model.pth"
    try:
        if hasattr(agent, 'save') and callable(agent.save):
            agent.save(model_path)
        else:
            torch.save(agent.online.state_dict(), model_path)
        try:
            artifact = wandb.Artifact(os.path.basename(model_path), type='model')
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)
        except Exception:
            try:
                wandb.save(model_path)
            except Exception:
                print("Warning: wandb failed to save the model file")
    except Exception as e:
        print('Failed saving model:', e)

    wandb.config.update({
        "environment": "MountainCar-v0",
        "state_dim": state_dim,
        "action_dim": action_dim,
        "learning_rate": 1e-3,
        "batch_size": batch_size,
        "episodes": episodes
    })

    wandb.finish()
    env.close()


if __name__ == "__main__":
    train_ddqn_mountaincar()
