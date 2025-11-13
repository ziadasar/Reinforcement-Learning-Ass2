import matplotlib.pyplot as plt
import re

# Read and parse data
episodes = []
rewards = []

with open('results//acrobot_tests//test_results_20251112_181316.txt', 'r') as file:
    for line in file:
        if 'Episode' in line and 'Reward' in line:
            numbers = re.findall(r'-?\d+\.\d+', line)
            if numbers:
                episodes.append(len(episodes) + 1)
                rewards.append(float(numbers[0]))

# Calculate average reward
avg_reward = sum(rewards) / len(rewards)

# Create plot
plt.figure(figsize=(10, 6))
plt.plot(episodes, rewards, 'bo-', alpha=0.7, markersize=3)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.grid(True, alpha=0.3)

# Add title and average reward below it
plt.suptitle('Acrobot Performance - Rewards per Episode', fontsize=14, y=0.98)
plt.title(f'Average Reward: {avg_reward:.2f}', fontsize=11, style='italic', pad=10)

plt.tight_layout()
plt.show()

print(f"Average Reward: {avg_reward:.2f}")