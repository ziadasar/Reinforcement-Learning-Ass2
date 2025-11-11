# environments/mountaincar_train.py
import gymnasium as gym
import torch
import wandb
from models.dqn import DQNAgent, ReplayBuffer
import numpy as np

def train_dqn_mountaincar():
    wandb.init(project="dqn-mountaincar", name="mountaincar_experiment")
    
    env = gym.make("MountainCar-v0")
    state_dim = env.observation_space.shape[0]  # 2 dimensions
    action_dim = env.action_space.n             # 3 actions

    print(f"MountainCar-v0 - State dim: {state_dim}, Action dim: {action_dim}")

    # MountainCar needs different approach - sparse rewards
    agent = DQNAgent(state_dim, action_dim, lr=1e-3, epsilon_decay=0.999)
    memory = ReplayBuffer(50000)  # Much larger memory
    episodes = 2000  # Many episodes needed due to sparse rewards
    batch_size = 64

    print("Starting MountainCar training...")

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        episode_loss = 0
        train_steps = 0

        for t in range(200):  # Shorter episodes
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Custom reward for MountainCar - encourage reaching top
            position = next_state[0]
            velocity = next_state[1]
            
            # Enhanced reward shaping
            reward = position + 0.1 * abs(velocity)  # Encourage moving right and having velocity
            
            if position >= 0.5:  # Success condition
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

        avg_loss = episode_loss / train_steps if train_steps > 0 else 0
        
        wandb.log({
            "episode": ep + 1,
            "total_reward": total_reward,
            "epsilon": agent.epsilon,
            "avg_loss": avg_loss,
            "buffer_size": len(memory),
            "success": 1 if total_reward > 90 else 0  # Track successes
        })

        if (ep + 1) % 100 == 0:
            print(f"Episode {ep+1}, Reward: {total_reward:.1f}, Epsilon: {agent.epsilon:.3f}")

    torch.save(agent.model.state_dict(), "dqn_mountaincar_model.pth")
    wandb.save("dqn_mountaincar_model.pth")
    
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
    print("MountainCar training completed!")

if __name__ == "__main__":
    train_dqn_mountaincar()