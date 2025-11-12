# pendulum_ddqn_train.py
import gymnasium as gym
import torch
import wandb
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from models.ddqn import DDQNAgent, ReplayBuffer

def train_ddqn_pendulum_discrete():
    wandb.init(project="ddqn-pendulum", name="ddqn_pendulum_experiment")

    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = 5
    action_values = np.linspace(-2.0, 2.0, action_dim)

    print(f"Pendulum-v1 - State dim: {state_dim}, Discrete actions: {action_dim}")

    agent = DDQNAgent(state_dim, action_dim, lr=1e-3, epsilon_decay=0.995)
    memory = ReplayBuffer(20000)
    episodes = 1000
    batch_size = 64

    print("Starting DDQN Pendulum training (discretized actions)...")

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        episode_loss = 0
        train_steps = 0

        for t in range(500):
            discrete_action = agent.select_action(state)
            action = [action_values[discrete_action]]

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            normalized_reward = (reward + 16.273) / 16.273

            memory.push(state, discrete_action, normalized_reward, next_state, done)
            loss = agent.train_step(memory, batch_size)
            
            if loss > 0:
                episode_loss += loss
                train_steps += 1

            state = next_state
            total_reward += reward

            if done:
                break

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}, Reward: {total_reward:.1f}, Epsilon: {agent.epsilon:.3f}")

    model_path = "ddqn_pendulum_model.pth"
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
        "environment": "Pendulum-v1",
        "state_dim": state_dim,
        "action_dim": action_dim,
        "learning_rate": 1e-3,
        "batch_size": batch_size,
        "episodes": episodes,
        "note": "Continuous actions discretized to 5 values"
    })

    wandb.finish()
    env.close()


if __name__ == "__main__":
    train_ddqn_pendulum_discrete()
