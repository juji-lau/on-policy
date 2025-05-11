import re
import argparse
import matplotlib.pyplot as plt

# --- Argument parser ---
parser = argparse.ArgumentParser(description="Plot training metrics from log file.")
parser.add_argument("--log_file", type=str, help="Path to the log file.")
parser.add_argument("--reward_type", type=str, help="Reward Type")
args = parser.parse_args()

# --- Load log data from file ---
with open(args.log_file, "r") as f:
    log_data = f.read()
    
# --- Regular expressions (robust against leading spaces or prefixes) ---
episode_pattern = re.compile(r".*updates (\d+)/\d+ episodes, total num timesteps (\d+)/\d+")
reward_pattern = re.compile(r"average episode rewards is ([\-\d\.eE]+)")
entropy_pattern = re.compile(r"policy entropy is ([\-\d\.eE]+)")
eval_pattern = re.compile(r"eval average episode rewards of agent: ([\-\d\.eE]+)")

# --- Containers ---
episodes = []
timesteps = []
avg_rewards = []
entropies = []
eval_points = []
eval_rewards = []

# --- Parse ---
lines = log_data.strip().splitlines()
i = 0
while i < len(lines):
    line = lines[i].strip()  # Strip leading/trailing whitespace
    episode_match = episode_pattern.search(line)

    if episode_match:
        try:
            ep = int(episode_match.group(1))
            ts = int(episode_match.group(2))

            reward_line = lines[i + 2].strip() if i + 1 < len(lines) else ""
            entropy_line = lines[i + 3].strip() if i + 2 < len(lines) else ""

            reward_match = reward_pattern.search(reward_line)
            entropy_match = entropy_pattern.search(entropy_line)

            if reward_match and entropy_match:
                reward = float(reward_match.group(1))
                entropy = float(entropy_match.group(1))

                episodes.append(ep)
                timesteps.append(ts)
                avg_rewards.append(reward)
                entropies.append(entropy)
            else:
                print(f"[Warning] Skipping line {i}: reward or entropy not found")
                i += 4
                continue

            if i + 4 < len(lines):
                eval_line = lines[i + 4].strip()
                eval_match = eval_pattern.search(eval_line)
                if eval_match:
                    eval_reward = float(eval_match.group(1))
                    eval_points.append(ep)
                    eval_rewards.append(eval_reward)
                    i += 5
                    continue

            i += 4
        except Exception as e:
            print(f"[Error] Failed parsing block at line {i}: {e}")
            i += 1
    else:
        i += 1

# --- Plot ---
plt.figure(figsize=(12, 6))

# Reward Plot
plt.subplot(2, 1, 1)
plt.plot(episodes, avg_rewards, label="Avg Reward", marker='o')
plt.plot(eval_points, eval_rewards, 'ro', label="Eval Reward")
plt.ylabel("Reward")
plt.title(f"{args.reward_type} Average and Eval Rewards Over Training")
plt.legend()

# Entropy Plot
plt.subplot(2, 1, 2)
plt.plot(episodes, entropies, label="Policy Entropy", color="orange", marker='o')
plt.xlabel("Episode (x5)")
plt.ylabel("Entropy")
plt.title(f"{args.reward_type} Policy Entropy Over Training")
plt.legend()

plt.tight_layout()
img_file = f"on-policy/logging/{args.reward_type}.png"
plt.savefig(f"{img_file}", dpi=300)  # Save the figure as a high-res PNG