# environments/pendulum_train.py
import gymnasium as gym
import torch
import wandb
import numpy as np

# For Pendulum we need a different approach - it has continuous actions
# We'll use a discrete version or need a different algorithm

def train_dqn_pendulum_discrete():
    """
    Pendulum has continuous actions, so we discretize them
    """
    wandb.init(project="dqn-pendulum", name="pendulum_experiment")
    
    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]  # 3 dimensions (cosθ, sinθ, θdot)
    
    # Discretize the continuous action space [-2, 2] into 5 actions
    action_dim = 5
    action_values = np.linspace(-2.0, 2.0, action_dim)  # [-2, -1, 0, 1, 2]

    print(f"Pendulum-v1 - State dim: {state_dim}, Discrete actions: {action_dim}")

    from models.dqn import DQNAgent, ReplayBuffer
    
    agent = DQNAgent(state_dim, action_dim, lr=1e-3, epsilon_decay=0.995)
    memory = ReplayBuffer(20000)
    episodes = 1000
    batch_size = 64

    print("Starting Pendulum training (discretized actions)...")

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
            normalized_reward = (reward + 16.273) / 16.273  # Normalize to ~[0, 1]
            
            memory.push(state, discrete_action, normalized_reward, next_state, done)
            loss = agent.train_step(memory, batch_size)
            
            if loss > 0:
                episode_loss += loss
                train_steps += 1

            state = next_state
            total_reward += reward  # Use original reward for logging

            if done:
                break

        avg_loss = episode_loss / train_steps if train_steps > 0 else 0
        
        wandb.log({
            "episode": ep + 1,
            "total_reward": total_reward,
            "epsilon": agent.epsilon,
            "avg_loss": avg_loss,
            "buffer_size": len(memory)
        })

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}, Reward: {total_reward:.1f}, Epsilon: {agent.epsilon:.3f}")

    torch.save(agent.model.state_dict(), "dqn_pendulum_model.pth")
    wandb.save("dqn_pendulum_model.pth")
    
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
    print("Pendulum training completed!")

if __name__ == "__main__":
    train_dqn_pendulum_discrete()