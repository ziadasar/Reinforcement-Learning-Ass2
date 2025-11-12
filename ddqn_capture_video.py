"""ddqn_capture_video.py
Record videos using DDQN models. Mirrors capture_video.py but explicitly loads DDQNAgent
and prefers ddqn_*-named model files. Run:
    python ddqn_capture_video.py cartpole 3 1000
or interactive:
    python ddqn_capture_video.py
"""

import os
from datetime import datetime
import warnings
import gymnasium as gym
import torch
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

# Environment configurations (reuse names from capture_video but prefer ddqn model filenames)
ENV_CONFIGS = {
    "cartpole": {
        "env_name": "CartPole-v1",
        "model_path": "ddqn_cartpole_model.pth",
        "state_dim": 4,
        "action_dim": 2,
        "discrete": True
    },
    "acrobot": {
        "env_name": "Acrobot-v1",
        "model_path": "ddqn_acrobot_model.pth",
        "state_dim": 6,
        "action_dim": 3,
        "discrete": True
    },
    "mountaincar": {
        "env_name": "MountainCar-v0",
        "model_path": "ddqn_mountaincar_model.pth",
        "state_dim": 2,
        "action_dim": 3,
        "discrete": True
    },
    "pendulum": {
        "env_name": "Pendulum-v1",
        "model_path": "ddqn_pendulum_model.pth",
        "state_dim": 3,
        "action_dim": 5,
        "discrete": False,
        "action_values": np.linspace(-2.0, 2.0, 5)
    }
}


def record_ddqn_video(environment_name, num_episodes=3, max_steps=1000, video_root="videos"):
    if environment_name not in ENV_CONFIGS:
        print(f"Unknown environment: {environment_name}. Available: {list(ENV_CONFIGS.keys())}")
        return

    config = dict(ENV_CONFIGS[environment_name])

    # prefer ddqn model path (already set) but if missing, try replacing dqn_ prefix
    model_path = config.get("model_path")
    if not os.path.exists(model_path):
        alt = model_path.replace("ddqn_", "dqn_")
        if os.path.exists(alt):
            print(f"DDQN model not found, falling back to DQN model: {alt}")
            model_path = alt

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    video_folder = os.path.join(video_root, f"ddqn_{environment_name}_{timestamp}")
    os.makedirs(video_folder, exist_ok=True)

    print(f"Recording {environment_name} to {video_folder}/ using model: {model_path}")

    env = gym.make(config["env_name"], render_mode="rgb_array")
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
    env = gym.wrappers.RecordVideo(env, video_folder, episode_trigger=lambda e: True)

    try:
        # import DDQNAgent
        from models.ddqn import DDQNAgent

        agent = DDQNAgent(config["state_dim"], config["action_dim"], enable_wandb=False)
        # DDQNAgent provides load(path) which handles device placement
        try:
            agent.load(model_path)
        except FileNotFoundError:
            raise
        except Exception:
            # fallback: load state dict onto agent.device
            state_dict = torch.load(model_path, map_location=agent.device)
            if hasattr(agent, 'online'):
                agent.online.load_state_dict(state_dict)
                agent.target.load_state_dict(agent.online.state_dict())

        agent.epsilon = 0

        # print device for confirmation
        try:
            print(f"Agent using device: {agent.device} (CUDA available: {torch.cuda.is_available()})")
        except Exception:
            pass

        print(f"Loaded model: {model_path}")

        for episode in range(num_episodes):
            state, _ = env.reset()
            done = False
            steps = 0
            total_reward = 0

            while not done and steps < max_steps:
                if config.get("discrete", True):
                    action = agent.select_action(state)
                else:
                    discrete_action = agent.select_action(state)
                    action = [config["action_values"][discrete_action]]

                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward
                steps += 1

            print(f"Episode {episode+1}: steps={steps}, reward={total_reward:.2f}")

    except FileNotFoundError:
        print(f"Model file not found: {model_path}")
    except Exception as e:
        print(f"Error during recording: {e}")
    finally:
        env.close()
        print(f"Videos saved to: {video_folder}")


def interactive():
    print("DDQN Video Recorder")
    print("Available envs:", list(ENV_CONFIGS.keys()))
    env_name = input("Environment name: ").strip().lower()
    episodes = input("Episodes [3]: ").strip() or "3"
    steps = input("Max steps [1000]: ").strip() or "1000"
    record_ddqn_video(env_name, int(episodes), int(steps))


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        env_name = sys.argv[1]
        episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        steps = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
        record_ddqn_video(env_name, episodes, steps)
    else:
        interactive()
