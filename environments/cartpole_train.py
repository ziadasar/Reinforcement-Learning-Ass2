# cartpole_train.py
import gymnasium as gym
import torch
import wandb
from models.dqn import DQNAgent, ReplayBuffer
# To run
#  python -m environments.cartpole_train                               

def train_dqn_cartpole():
    # Initialize WandB
    wandb.init(project="dqn-cartpole", name="cartpole_experiment")
    
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"state dimension: {state_dim}, action dimension: {action_dim}")

    agent = DQNAgent(state_dim, action_dim, lr=5e-4, epsilon_decay=0.99)
    memory = ReplayBuffer(10000)
    episodes = 500
    batch_size = 64

    print("Starting training...")

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

        avg_loss = episode_loss / train_steps if train_steps > 0 else 0
        
        # Log metrics to WandB
        wandb.log({
            "episode": ep + 1,
            "total_reward": total_reward,
            "epsilon": agent.epsilon,
            "avg_loss": avg_loss,
            "buffer_size": len(memory)
        })

        print(f"Episode {ep+1}, Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}, Avg Loss: {avg_loss:.4f}")

    # Save model and cleanup
    torch.save(agent.model.state_dict(), "dqn_cartpole_model.pth")
    wandb.save("dqn_cartpole_model.pth")
    
    # Log hyperparameters
    wandb.config.update({
        "state_dim": state_dim,
        "action_dim": action_dim,
        "gamma": agent.gamma,
        "learning_rate": 5e-4,
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "epsilon_decay": 0.99,
        "batch_size": batch_size,
        "buffer_size": 10000,
        "episodes": episodes
    })
    
    wandb.finish()
    env.close()


if __name__ == "__main__":
    train_dqn_cartpole()