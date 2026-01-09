import torch
import os
import gymnasium as gym
import numpy as np
import logging
import json
import time
from datetime import datetime


NUM_ITERATIONS = 60
NUM_ENV_RUNNERS = int(os.environ.get("NUM_ENV_RUNNERS", 8))
NUM_ENVS_PER_ENV_RUNNER = 1
TRAIN_BATCH_SIZE = 8000
NUM_EPOCHS = 10

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
    .env_runners(
        num_env_runners=NUM_ENV_RUNNERS,
        num_envs_per_env_runner=NUM_ENVS_PER_ENV_RUNNER,
        rollout_fragment_length=100,
        sample_timeout_s=120,
    )
    .rl_module(
        model_config=DefaultModelConfig(
            # DeepMind Atari CNN
            conv_activation="relu",
            conv_filters=[
                [32, [8, 8], 4],
                [64, [4, 4], 2],
                [64, [3, 3], 1],
            ],
            head_fcnet_hiddens=[512],
            vf_share_layers=True,
        )
    )
    .training(
        lr=0.0002,
        train_batch_size=TRAIN_BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
    )
    .evaluation(
        evaluation_interval=5,
        evaluation_num_env_runners=4,
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

# Ensure save_dir is inside project folder
try:
    common = os.path.commonpath([project_dir, os.path.abspath(save_dir)])
except Exception:
    common = None

if common != project_dir:
    save_dir = default_out
    print(f"WARNING: SAVE_DIR outside project tree. Using {save_dir}")

os.makedirs(save_dir, exist_ok=True)

logger = logging.getLogger("training")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")

fh = logging.FileHandler(os.path.join(save_dir, "train.log"))
fh.setFormatter(formatter)
sh = logging.StreamHandler()
sh.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(fh)
    logger.addHandler(sh)

metrics_jsonl_path = os.path.join(save_dir, f"metrics_{NUM_ENV_RUNNERS}envrunners.jsonl")


# ==============================
# Training Loop
# ==============================
training_start = time.perf_counter()
prev_env_steps = 0

for i in range(NUM_ITERATIONS):
    result = algo.train()
    logger.info("=== Training iteration %d ===", i)

    env_stats = result.get("env_runners", {})
    learner_stats = result.get("learners", {}).get("default_policy", {})

    total_steps = env_stats.get("num_env_steps_sampled_lifetime", 0)
    steps_this_iter = total_steps - prev_env_steps
    prev_env_steps = total_steps

    time_iter = result.get("time_this_iter_s", 0.0)
    steps_per_sec = (
        steps_this_iter / time_iter if time_iter > 0 and steps_this_iter > 0 else None
    )

    raw_metrics = {
        "timestamp": datetime.utcnow().isoformat(),
        "iteration": i,

        # parallelism config
        "num_env_runners": NUM_ENV_RUNNERS,
        "num_envs_per_env_runner": NUM_ENVS_PER_ENV_RUNNER,

        # sampling / throughput
        "env_steps_lifetime": total_steps,
        "env_steps_this_iter": steps_this_iter,
        "time_this_iter_s": time_iter,
        "env_steps_per_second": steps_per_sec,

        # env stats
        "episode_return_mean": env_stats.get("episode_return_mean"),
        "episode_len_mean": env_stats.get("episode_len_mean"),
        "num_episodes": env_stats.get("num_episodes"),

        # learner stats
        "total_loss": learner_stats.get("total_loss"),
        "policy_loss": learner_stats.get("policy_loss"),
        "vf_loss": learner_stats.get("vf_loss"),
        "entropy": learner_stats.get("entropy"),
    }

    metrics = {k: sanitize(v) for k, v in raw_metrics.items()}

    with open(metrics_jsonl_path, "a") as f:
        f.write(json.dumps(metrics) + "\n")

    logger.info(
        "Steps: %s | Time: %.2f s | Throughput: %s steps/s | Return: %s",
        metrics["env_steps_this_iter"],
        metrics["time_this_iter_s"],
        metrics["env_steps_per_second"],
        metrics["episode_return_mean"],
    )


# ==============================
# Save global information
# ==============================
total_time_s = time.perf_counter() - training_start

info = {
    "timestamp": datetime.utcnow().isoformat(),
    "num_env_runners": NUM_ENV_RUNNERS,
    "num_envs_per_env_runner": NUM_ENVS_PER_ENV_RUNNER,
    "num_iterations": NUM_ITERATIONS,
    "train_batch_size": TRAIN_BATCH_SIZE,
    "total_training_time_s": total_time_s,
    "total_training_time_min": total_time_s / 60,
    "total_env_steps": prev_env_steps,
}

with open(os.path.join(save_dir, f"information_{NUM_ENV_RUNNERS}envrunners.json"), "w") as f:
    json.dump({k: sanitize(v) for k, v in info.items()}, f, indent=2)

logger.info("Information saved")


# ==============================
# Evaluation + checkpoint
# ==============================
algo.evaluate()
checkpoint_path = algo.save(os.path.abspath(save_dir))
logger.info("Checkpoint saved at: %s", checkpoint_path)

algo.stop()
logger.info("Training completed.")
