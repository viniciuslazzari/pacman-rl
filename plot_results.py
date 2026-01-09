import json
import matplotlib.pyplot as plt
import os

# Load metrics
metrics_file = "out/metrics.jsonl"

if not os.path.exists(metrics_file):
    raise FileNotFoundError(f"{metrics_file} does not exist. Make sure training completed successfully.")

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

# Plot Dashboard
plt.figure(figsize=(15, 12))

# 1. Reward curve
plt.subplot(2, 2, 1)
plt.plot(iterations, returns, marker='o', color='tab:blue')
plt.xlabel("Training Iteration")
plt.ylabel("Mean Episode Return")
plt.title("Learning Curve (Reward)")
plt.grid(True)

# 2. Episode length
plt.subplot(2, 2, 2)
plt.plot(iterations, lengths, marker='o', color='tab:orange')
plt.xlabel("Training Iteration")
plt.ylabel("Mean Episode Length")
plt.title("Episode Length Over Time")
plt.grid(True)

# 3. Sample efficiency (reward vs env steps)
plt.subplot(2, 2, 3)
plt.plot(steps, returns, marker='o', color='tab:green')
plt.xlabel("Total Environment Steps")
plt.ylabel("Mean Episode Return")
plt.title("Sample Efficiency")
plt.grid(True)

# 4. Losses and entropy
plt.subplot(2, 2, 4)
plt.plot(iterations, policy_loss, label="Policy Loss", color='tab:red')
plt.plot(iterations, vf_loss, label="Value Loss", color='tab:purple')
plt.plot(iterations, entropy, label="Entropy", color='tab:brown')
plt.xlabel("Training Iteration")
plt.ylabel("Loss / Entropy")
plt.title("Losses & Policy Entropy")
plt.legend()
plt.grid(True)

plt.tight_layout(rect=[0, 0, 0.72, 1])

plt.savefig("out/training_dashboard.png", dpi=300)
plt.show()

print("Dashboard saved to out/training_dashboard.png")
