# environments/pendulum_train.py
import gymnasium as gym
import torch
import wandb
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dqn import DQNAgent, ReplayBuffer
def train_dqn_pendulum_discrete():
    wandb.init(project="dqn-pendulum", name="pendulum_experiment")
    
    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]
    
    # Better action discretization
    action_dim = 9
    action_values = np.linspace(-2.0, 2.0, action_dim)

    print(f"Pendulum-v1 - State dim: {state_dim}, Discrete actions: {action_dim}")

    
    # PENDULUM-SPECIFIC HYPERPARAMETERS
    agent = DQNAgent(
        state_dim=state_dim, 
        action_dim=action_dim, 
        lr=1e-4,
        gamma=0.95,         # Slightly lower discount - focus on immediate rewards
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.997
    )
    
    memory = ReplayBuffer(50000)
    episodes = 500
    batch_size = 128

    print("Starting Pendulum training...")

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        episode_loss = 0
        train_steps = 0

        for t in range(200):
            discrete_action = agent.select_action(state)
            action = [action_values[discrete_action]]
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # FIXED REWARD PROCESSING
            # Pendulum rewards are ALREADY good for DQN - just use them as-is
            # But we can scale them to be more reasonable
            processed_reward = reward * 0.1  # Scale to [-1.6, 0] range
            
            memory.push(state, discrete_action, processed_reward, next_state, done)
            loss = agent.train_step(memory, batch_size)
            
            if loss > 0:
                episode_loss += loss
                train_steps += 1

            state = next_state
            total_reward += reward  # Keep original for logging

            if done:
                break

        avg_loss = episode_loss / train_steps if train_steps > 0 else 0
        
        # Calculate normalized score for better interpretation
        # Worst case: -16.27 per step * 200 steps = -3254
        # Best case: 0 per step * 200 steps = 0
        normalized_score = (total_reward + 3254) / 3254  # Convert to [0, 1] scale
        
        wandb.log({
            "episode": ep + 1,
            "total_reward": total_reward,           # Original: -1800 to -600
            "normalized_score": normalized_score,   # Scaled: 0.45 to 0.82
            "epsilon": agent.epsilon,
            "avg_loss": avg_loss,
            "buffer_size": len(memory),
            "episode_length": t + 1
        })

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1:3d}, Reward: {total_reward:7.1f}, "
                  f"Norm: {normalized_score:.3f}, Epsilon: {agent.epsilon:.3f}")

    torch.save(agent.model.state_dict(), "dqn_pendulum_model.pth")
    wandb.save("dqn_pendulum_model.pth")
    
    wandb.config.update({
        "environment": "Pendulum-v1",
        "state_dim": state_dim,
        "action_dim": action_dim,
        "learning_rate": 1e-4,
        "gamma": 0.95,
        "batch_size": batch_size,
        "episodes": episodes,
        "reward_scale": 0.1,
        "note": "Using scaled rewards, 9 discrete actions"
    })
    
    wandb.finish()
    env.close()
    print("Pendulum training completed!")

if __name__ == "__main__":
    train_dqn_pendulum_discrete()