import torch
import os
import gymnasium as gym
import numpy as np
import logging
import json
import time
from datetime import datetime


NUM_ITERATIONS = 100
NUM_ENV_RUNNERS = 60
NUM_ENVS_PER_ENV_RUNNER = 2

# ==============================
# Helper to sanitize metrics
# ==============================
def sanitize(value):
    """Convert NumPy scalars to native Python types for JSON"""
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    elif isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    return value

# ==============================
#  Atari → Float32 Wrapper
# ==============================
class FloatObsEnv(gym.Env):
    def __init__(self, config=None):
        self.env = gym.make("ale_py:ALE/Pacman-v5")
        original_space = self.env.observation_space
        self.observation_space = gym.spaces.Box(
            low=original_space.low.astype(np.float32) / 255.0,
            high=original_space.high.astype(np.float32) / 255.0,
            shape=original_space.shape,
            dtype=np.float32,
        )
        self.action_space = self.env.action_space

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs.astype(np.float32) / 255.0, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs.astype(np.float32) / 255.0, reward, terminated, truncated, info

# ==============================
#  Register environment
# ==============================
from ray.tune.registry import register_env
def env_creator(config):
    return FloatObsEnv(config)
register_env("PacmanFloat", env_creator)

# ==============================
#  RLlib PPO Config
# ==============================
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig

config = (
    PPOConfig()
    .environment("PacmanFloat")
    .env_runners(num_env_runners=NUM_ENV_RUNNERS, num_envs_per_env_runner=NUM_ENVS_PER_ENV_RUNNER)
    .rl_module(
        model_config=DefaultModelConfig(
            conv_activation="relu",
            conv_filters=[
                [32, [8, 8], 4],
                [64, [4, 4], 2],
                [64, [3, 3], 1],
            ],
            head_fcnet_hiddens=[256, 256],
            vf_share_layers=False
        )
    )
    .training(
        lr=0.0002,
        train_batch_size=8000,
        num_epochs=10,
    )
    .evaluation(
        evaluation_interval=5,
        evaluation_num_env_runners=4
    )
)

# Build the algorithm
algo = config.build_algo()

# ==============================
# Output directory and logging
# ==============================
project_dir = os.path.dirname(os.path.abspath(__file__))
default_out = os.path.join(project_dir, "out")
save_dir = os.environ.get("SAVE_DIR", default_out)

# Ensure save_dir is inside the project folder
try:
    common = os.path.commonpath([project_dir, os.path.abspath(save_dir)])
except Exception:
    common = None
if common != project_dir:
    save_dir = default_out
    print(f"WARNING: SAVE_DIR was outside project tree. Forcing save_dir to {save_dir}")

os.makedirs(save_dir, exist_ok=True)

# Configure logger
logger = logging.getLogger("training")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
fh = logging.FileHandler(os.path.join(save_dir, "train.log"))
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
sh.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(fh)
    logger.addHandler(sh)

# Metrics file
metrics_jsonl_path = os.path.join(save_dir, "metrics.jsonl")

# ==============================
# Training Loop
# ==============================
training_start = time.perf_counter()
for i in range(NUM_ITERATIONS):
    result = algo.train()
    logger.info("=== Training iteration %d ===", i)

    env_stats = result.get("env_runners", {})
    learner_stats = result.get("learners", {}).get("default_policy", {})

    raw_metrics = {
        "timestamp": datetime.utcnow().isoformat(),
        "iteration": i,
        "episode_return_mean": env_stats.get("episode_return_mean"),
        "episode_len_mean": env_stats.get("episode_len_mean"),
        "num_episodes": env_stats.get("num_episodes"),
        "num_env_steps_sampled_lifetime": env_stats.get("num_env_steps_sampled_lifetime"),
        "time_this_iter_s": result.get("time_this_iter_s"),
        "total_loss": learner_stats.get("total_loss"),
        "policy_loss": learner_stats.get("policy_loss"),
        "vf_loss": learner_stats.get("vf_loss"),
        "entropy": learner_stats.get("entropy"),
    }

    metrics = {k: sanitize(v) for k, v in raw_metrics.items()}

    if metrics["episode_return_mean"] is None:
        logger.warning(
            "Iteration %d: no completed episodes yet, skipping metrics write", i
        )
        continue

    # Logging summary
    logger.info(
        "Summary - Episode Return Mean: %s, Episode Len Mean: %s, "
        "Num Episodes: %s, Total Steps: %s, Time: %s s",
        metrics["episode_return_mean"],
        metrics["episode_len_mean"],
        metrics["num_episodes"],
        metrics["num_env_steps_sampled_lifetime"],
        metrics["time_this_iter_s"]
    )

    logger.info(
        "Losses - Total: %s, VF: %s, Policy: %s, Entropy: %s",
        metrics["total_loss"],
        metrics["vf_loss"],
        metrics["policy_loss"],
        metrics["entropy"]
    )

    # Write JSONL
    with open(metrics_jsonl_path, "a") as f:
        f.write(json.dumps(metrics) + "\n")

    # Uncomment the line below to log the full result dict if needed
    # logger.info(result)

# Compute total training time and save information
total_time_s = time.perf_counter() - training_start
totals = {
    "timestamp": datetime.utcnow().isoformat(),
    "total_training_time_s": float(total_time_s),
    "total_training_time_min": float(total_time_s / 60),
    "num_env_runners": NUM_ENV_RUNNERS,
    "num_envs_per_env_runner": NUM_ENVS_PER_ENV_RUNNER,
    "total_env_steps_sampled_lifetime": metrics.get("num_env_steps_sampled_lifetime") if 'metrics' in locals() else None
}
totals_path = os.path.join(save_dir, "information.json")
with open(totals_path, "w") as f:
    json.dump({k: sanitize(v) for k, v in totals.items()}, f, indent=2)
logger.info("Information saved to %s", totals_path)

# ==============================
#  Evaluation
# ==============================
eval_result = algo.evaluate()

# Save checkpoint
checkpoint_path = algo.save(save_dir)
logger.info("Checkpoint saved at: %s", checkpoint_path)

# ==============================
#  Cleanup
# ==============================
algo.stop()
logger.info("Training completed.")
