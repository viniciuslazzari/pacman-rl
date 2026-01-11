import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import os

jsonl_folder = "experiments" 
output_folder = "comparison_results"
os.makedirs(output_folder, exist_ok=True)

# Load data from all experiments
experiments = {}
for file in glob.glob(os.path.join(jsonl_folder, "*.jsonl")):
    exp_name = os.path.splitext(os.path.basename(file))[0]
    metrics = []
    with open(file, "r") as f:
        for line in f:
            metrics.append(json.loads(line))
    experiments[exp_name] = pd.DataFrame(metrics)

# Extract metrics and create summary table
summary_rows = []
for exp_name, df in experiments.items():
    final_reward = df['episode_return_mean'].iloc[-1]
    max_reward = df['episode_return_mean'].max()
    mean_reward = df['episode_return_mean'].mean()
    reward_std = df['episode_return_mean'].std()
    mean_episode_len = df['episode_len_mean'].mean()
    # Exclude iteration 0 when computing mean env steps/sec
    if 'env_steps_per_second' in df.columns:
        if len(df) > 1:
            mean_env_steps_per_sec = df['env_steps_per_second'].iloc[1:].mean()
        else:
            mean_env_steps_per_sec = float('nan')
    else:
        mean_env_steps_per_sec = float('nan')
    
    summary_rows.append({
        "Experiment": exp_name,
        "Final Reward": final_reward,
        "Max Reward": max_reward,
        "Mean Reward": mean_reward,
        "Reward Std": reward_std,
        "Mean Episode Length": mean_episode_len,
        "Mean Steps/sec": mean_env_steps_per_sec
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(output_folder, "experiments_summary.csv"), index=False)

# Plot comparison graphs
# Reward vs iteration
plt.figure(figsize=(12, 8))
for exp_name, df in experiments.items():
    plt.plot(df.index, df['episode_return_mean'], label=exp_name)
plt.xlabel("Iteration")
plt.ylabel("Mean Episode Reward")
plt.title("Reward vs Iteration")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "reward_vs_iteration.png"))
plt.close()

# Reward vs environment steps
plt.figure(figsize=(12, 8))
for exp_name, df in experiments.items():
    plt.plot(df['env_steps_lifetime'], df['episode_return_mean'], label=exp_name)
plt.xlabel("Environment Steps")
plt.ylabel("Mean Episode Reward")
plt.title("Reward vs Environment Steps")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "reward_vs_env_steps.png"))
plt.close()

# Other metrics vs iteration
metrics_to_plot = ['policy_loss', 'vf_loss', 'entropy', 'env_steps_per_second']
for metric in metrics_to_plot:
    plt.figure(figsize=(12, 8))
    for exp_name, df in experiments.items():
        if metric in df.columns:
            # For env_steps_per_second, exclude iteration 0
            if metric == 'env_steps_per_second':
                if len(df) > 1:
                    plt.plot(df.index[1:], df[metric].iloc[1:], label=exp_name)
                else:
                    plt.plot(df.index, df[metric], label=exp_name)
            else:
                plt.plot(df.index, df[metric], label=exp_name)
    plt.xlabel("Iteration")
    plt.ylabel(metric)
    plt.title(f"{metric} vs Iteration")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{metric}_vs_iteration.png"))
    plt.close()

print(f"Comparison plots and summary table saved in {output_folder}")
