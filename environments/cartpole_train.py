import gymnasium as gym
import torch
import wandb
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dqn import DQNAgent, ReplayBuffer

def train_dqn_cartpole():
    wandb.init(project="dqn-cartpole", name="cartpole_stable")

    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    print(f"state_dim={state_dim}, action_dim={action_dim}")

    agent = DQNAgent(
        state_dim,
        action_dim,
        lr=1e-4,           # smaller learning rate
        gamma=0.99,
        epsilon_decay=0.995
    )

    memory = ReplayBuffer(50000)
    episodes = 500
    batch_size = 64
    min_buffer_size = 1000  # don't train before this many experiences

    print("Starting stable training...")

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        episode_loss = 0
        train_steps = 0

        for t in range(500):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # normalize reward (optional but stabilizes)
            reward = reward / 100.0

            memory.push(state, action, reward, next_state, done)

            # Train only if enough samples
            if len(memory) > min_buffer_size:
                loss = agent.train_step(memory, batch_size)
                if loss > 0:
                    episode_loss += loss
                    train_steps += 1

            state = next_state
            total_reward += reward * 100.0  # log original reward

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

        print(f"Ep {ep+1:3d} | Reward: {total_reward:6.1f} | "
              f"Eps: {agent.epsilon:.3f} | Loss: {avg_loss:.5f}")

    torch.save(agent.model.state_dict(), "dqn_cartpole_model.pth")
    wandb.save("dqn_cartpole_model.pth")
    wandb.finish()
    env.close()

if __name__ == "__main__":
    train_dqn_cartpole()
