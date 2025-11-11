# environments/record_video.py
import os
from datetime import datetime
import warnings
import gymnasium as gym
import torch
import numpy as np
from models.dqn import DQNAgent

warnings.filterwarnings("ignore", category=UserWarning)

# Environment configurations
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
        "action_dim": 5,  # Discretized actions
        "discrete": False,
        "action_values": np.linspace(-2.0, 2.0, 5)
    }
}

def record_video(environment_name, num_episodes=3, max_steps=1000, agent_type="DQN"):
    """
    Record video of a trained agent playing the specified environment
    
    Args:
        environment_name: One of ['cartpole', 'acrobot', 'mountaincar', 'pendulum']
        num_episodes: Number of episodes to record
        max_steps: Maximum steps per episode
        agent_type: Type of agent (DQN or DDQN) for folder naming
    """
    
    if environment_name not in ENV_CONFIGS:
        available_envs = list(ENV_CONFIGS.keys())
        print(f"Error: Environment '{environment_name}' not found.")
        print(f"Available environments: {available_envs}")
        return
    
    config = ENV_CONFIGS[environment_name]
    
    # Create video folder with DQN_environment_date format
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    video_folder = f"videos/{agent_type}_{environment_name}_{timestamp}"  # CHANGED THIS LINE
    os.makedirs(video_folder, exist_ok=True)
    
    print(f"Recording {environment_name} to {video_folder}/")
    
    # Create environment with video recording
    env = gym.make(config["env_name"], render_mode="rgb_array")
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
    env = gym.wrappers.RecordVideo(env, video_folder, episode_trigger=lambda e: True)
    
    try:
        # Load the trained model
        agent = DQNAgent(config["state_dim"], config["action_dim"], enable_wandb=False)
        agent.model.load_state_dict(torch.load(config["model_path"]))
        agent.epsilon = 0  # Disable exploration (pure exploitation)
        
        print(f"Loaded model: {config['model_path']}")
        print(f"Recording {num_episodes} episodes...")
        
        # Record episodes
        for episode in range(num_episodes):
            state, _ = env.reset()
            done = False
            steps = 0
            total_reward = 0
            
            while not done and steps < max_steps:
                if config["discrete"]:
                    action = agent.select_action(state)
                else:
                    # For continuous environments like Pendulum
                    discrete_action = agent.select_action(state)
                    action = [config["action_values"][discrete_action]]
                
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                steps += 1
                total_reward += reward
            
            print(f"Episode {episode + 1}: {steps} steps, Reward: {total_reward:.1f}")
            
    except FileNotFoundError:
        print(f"Error: Model file {config['model_path']} not found.")
        print("Please train the model first using the appropriate training script.")
    except Exception as e:
        print(f"Error during recording: {e}")
    finally:
        env.close()
        print(f"Videos saved to: {video_folder}")

def interactive_record():
    """Interactive version that asks user for input"""
    print("🎥 DQN Agent Video Recorder")
    print("=" * 40)
    print("Available environments:")
    for i, env_name in enumerate(ENV_CONFIGS.keys(), 1):
        print(f"  {i}. {env_name}")
    
    try:
        choice = input("\nSelect environment (number or name): ").strip().lower()
        
        # Handle numeric input
        if choice.isdigit():
            choice_num = int(choice) - 1
            env_names = list(ENV_CONFIGS.keys())
            if 0 <= choice_num < len(env_names):
                environment_name = env_names[choice_num]
            else:
                print("Invalid number selection.")
                return
        else:
            # Handle text input
            environment_name = choice
        
        # Get number of episodes
        episodes_input = input("Number of episodes to record [3]: ").strip()
        num_episodes = int(episodes_input) if episodes_input else 3
        
        # Get max steps
        steps_input = input("Max steps per episode [1000]: ").strip()
        max_steps = int(steps_input) if steps_input else 1000
        
        # Get agent type
        agent_input = input("Agent type (DQN/DDQN) [DQN]: ").strip().upper()
        agent_type = agent_input if agent_input in ["DQN", "DDQN"] else "DQN"
        
        print(f"\nRecording {num_episodes} episodes of {environment_name} with {agent_type}...")
        record_video(environment_name, num_episodes, max_steps, agent_type)
        
    except ValueError:
        print("Invalid input. Please enter a valid number.")
    except KeyboardInterrupt:
        print("\nRecording cancelled.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Command line mode
        environment_name = sys.argv[1]
        num_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        max_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
        agent_type = sys.argv[4] if len(sys.argv) > 4 else "DQN"
        
        record_video(environment_name, num_episodes, max_steps, agent_type)
    else:
        # Interactive mode
        interactive_record()