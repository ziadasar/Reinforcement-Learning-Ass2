# environments/acrobot_train.py
import gymnasium as gym
import torch
import wandb
from models.dqn import DQNAgent, ReplayBuffer

def train_dqn_acrobot():
    wandb.init(project="dqn-acrobot", name="acrobot_experiment")
    
    env = gym.make("Acrobot-v1")
    state_dim = env.observation_space.shape[0]  # 6 dimensions
    action_dim = env.action_space.n             # 3 actions

    print(f"Acrobot-v1 - State dim: {state_dim}, Action dim: {action_dim}")

    # Different hyperparameters for Acrobot
    agent = DQNAgent(state_dim, action_dim, lr=1e-3, epsilon_decay=0.995)
    memory = ReplayBuffer(20000)  # Larger memory for complex environment
    episodes = 1000  # More episodes needed
    batch_size = 64

    print("Starting Acrobot training...")

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        episode_loss = 0
        train_steps = 0

        for t in range(500):  # Max steps
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Acrobot gives negative rewards until success
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
            "buffer_size": len(memory)
        })

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}, Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")

    torch.save(agent.model.state_dict(), "dqn_acrobot_model.pth")
    wandb.save("dqn_acrobot_model.pth")
    
    wandb.config.update({
        "environment": "Acrobot-v1",
        "state_dim": state_dim,
        "action_dim": action_dim,
        "learning_rate": 1e-3,
        "batch_size": batch_size,
        "episodes": episodes
    })
    
    wandb.finish()
    env.close()
    print("Acrobot training completed!")

if __name__ == "__main__":
    train_dqn_acrobot()