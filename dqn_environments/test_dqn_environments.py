import os
import torch
import gymnasium as gym
import numpy as np
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.dqn import DQNAgent

# ✅ Same environment config as your video recorder
ENV_CONFIGS = {
    "cartpole": {
        "env_name": "CartPole-v1",
        "model_path": "dqn_cartpole_model.pth",
        "state_dim": 4,
        "action_dim": 2,
        "discrete": True
    },
    "acrobot": {
        "env_name": "Acrobot-v1",
        "model_path": "dqn_acrobot_model.pth",
        "state_dim": 6,
        "action_dim": 3,
        "discrete": True
    },
    "mountaincar": {
        "env_name": "MountainCar-v0",
        "model_path": "dqn_mountaincar_model.pth",
        "state_dim": 2,
        "action_dim": 3,
        "discrete": True
    },
    "pendulum": {
        "env_name": "Pendulum-v1",
        "model_path": "dqn_pendulum_model.pth",
        "state_dim": 3,
        "action_dim": 9,
        "discrete": False,
        "action_values": np.linspace(-2.0, 2.0, 9)
    }
}


def test_agent(environment_name, num_episodes=10, max_steps=1000):
    """
    Test a trained DQN agent and save episode results to a text file.
    """
    if environment_name not in ENV_CONFIGS:
        print(f"❌ Environment '{environment_name}' not found.")
        print(f"Available: {list(ENV_CONFIGS.keys())}")
        return

    config = ENV_CONFIGS[environment_name]

    # ✅ Create a results folder with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_folder = f"results/{environment_name}_tests"
    os.makedirs(results_folder, exist_ok=True)
    results_file = os.path.join(results_folder, f"test_results_{timestamp}.txt")

    print(f"🧪 Testing {environment_name} ...")
    print(f"Results will be saved to: {results_file}")

    # ✅ Load environment
    env = gym.make(config["env_name"])
    agent = DQNAgent(config["state_dim"], config["action_dim"], enable_wandb=False)

    try:
        agent.model.load_state_dict(torch.load(config["model_path"]))
        agent.epsilon = 0  # No exploration during evaluation
        print(f"✅ Loaded model: {config['model_path']}")
    except FileNotFoundError:
        print(f"❌ Model file '{config['model_path']}' not found. Please train it first.")
        return

    episode_rewards = []
    episode_durations = []

    # ✅ Run testing episodes
    for ep in range(1, num_episodes + 1):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done and steps < max_steps:
            if config["discrete"]:
                action = agent.select_action(state)
            else:
                discrete_action = agent.select_action(state)
                action = [config["action_values"][discrete_action]]

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            state = next_state

        episode_rewards.append(total_reward)
        episode_durations.append(steps)
        print(f"Episode {ep:02d}: Reward = {total_reward:.2f}, Duration = {steps}")

    env.close()

    # ✅ Compute averages
    avg_reward = np.mean(episode_rewards)
    avg_duration = np.mean(episode_durations)

    # ✅ Save all results to .txt
    with open(results_file, "w") as f:
        f.write(f"Test Results - {environment_name}\n")
        f.write("=" * 40 + "\n\n")
        for i, (r, d) in enumerate(zip(episode_rewards, episode_durations), start=1):
            f.write(f"Episode {i:02d}: Reward = {r:.2f}, Duration = {d}\n")
        f.write("\n" + "=" * 40 + "\n")
        f.write(f"Average Reward: {avg_reward:.2f}\n")
        f.write(f"Average Duration: {avg_duration:.2f}\n")

    print("\n✅ Testing complete.")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Duration: {avg_duration:.2f}")
    print(f"📄 Results saved to: {results_file}")


def interactive_test():
    """Interactive CLI for testing any trained agent"""
    print("🧠 DQN Agent Tester")
    print("=" * 40)
    print("Available environments:")
    for i, env_name in enumerate(ENV_CONFIGS.keys(), 1):
        print(f"  {i}. {env_name}")

    choice = input("\nSelect environment (number or name): ").strip().lower()
    if choice.isdigit():
        idx = int(choice) - 1
        env_names = list(ENV_CONFIGS.keys())
        if 0 <= idx < len(env_names):
            environment_name = env_names[idx]
        else:
            print("Invalid number.")
            return
    else:
        environment_name = choice

    episodes_input = input("Number of test episodes [10]: ").strip()
    num_episodes = int(episodes_input) if episodes_input else 10

    steps_input = input("Max steps per episode [1000]: ").strip()
    max_steps = int(steps_input) if steps_input else 1000

    test_agent(environment_name, num_episodes, max_steps)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # Command line usage: python test_agent.py <env> [num_episodes] [max_steps]
        env_name = sys.argv[1]
        num_eps = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        max_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
        test_agent(env_name, num_eps, max_steps)
    else:
        interactive_test()
