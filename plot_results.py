"""Plot training metrics and save a training dashboard image.

Usage:
    python3 plot_results.py [RUNNERS]

Example:
    python3 plot_results.py 96
"""

import json
import matplotlib.pyplot as plt
import os
import argparse

parser = argparse.ArgumentParser(description="Plot training results for given number of env runners")
parser.add_argument("runners", nargs="?", type=int, default=1, help="Number of env runners (default: 1)")
args = parser.parse_args()
runners = args.runners

# Load metrics
metrics_file = f"out/metrics_{runners}envrunners.jsonl"

if not os.path.exists(metrics_file):
    raise FileNotFoundError(f"{metrics_file} does not exist. Make sure training for {runners} env runners completed successfully.")

metrics = []
with open(metrics_file) as f:
    for line in f:
        metrics.append(json.loads(line))

# Extract data
iterations = [m["iteration"] for m in metrics]
returns = [m["episode_return_mean"] for m in metrics]
lengths = [m["episode_len_mean"] for m in metrics]
steps = [m["env_steps_lifetime"] for m in metrics]
policy_loss = [m["policy_loss"] for m in metrics]
vf_loss = [m["vf_loss"] for m in metrics]
entropy = [m["entropy"] for m in metrics]
env_steps_per_sec = [m["env_steps_per_second"] for m in metrics]    

# Plot Dashboard
plt.figure(figsize=(12, 10))

# 1. Sample efficiency (reward vs env steps)
plt.subplot(2, 2, 1)
plt.plot(steps, returns, marker='o', color='tab:green')
plt.xlabel("Total Environment Steps")
plt.ylabel("Mean Episode Return")
plt.title("Sample Efficiency (Reward vs Env steps)")
plt.grid(True)

# 2. Episode length
plt.subplot(2, 2, 2)
plt.plot(iterations, lengths, marker='o', color='tab:orange')
plt.xlabel("Training Iteration")
plt.ylabel("Mean Episode Length")
plt.title("Episode Length Over Time")
plt.grid(True)

# 3. Losses and entropy
plt.subplot(2, 2, 3)
plt.plot(iterations, policy_loss, label="Policy Loss", color='tab:red')
plt.plot(iterations, vf_loss, label="Value Loss", color='tab:purple')
plt.plot(iterations, entropy, label="Entropy", color='tab:brown')
plt.xlabel("Training Iteration")
plt.ylabel("Loss / Entropy")
plt.title("Losses & Policy Entropy")
plt.legend()
plt.grid(True)

# 4. Environment steps per second
plt.subplot(2, 2, 4)
plt.plot(iterations[1:], env_steps_per_sec[1:], marker='o', color='tab:blue')
plt.xlabel("Training Iteration")
plt.ylabel("Env Steps / Second")
plt.title("Sampling Throughput")
plt.grid(True)

plt.tight_layout()

out_file = f"out/training_dashboard_{runners}envrunners.png"
plt.savefig(out_file, dpi=300)
plt.show()

print(f"Dashboard saved to {out_file}")
