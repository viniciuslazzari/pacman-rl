import torch
import os
import gymnasium as gym
import numpy as np
import logging
import json
import shutil
from datetime import datetime
from pprint import pprint

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
    .env_runners(num_env_runners=60, num_envs_per_env_runner=2)
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
#  Training Loop
# ==============================
# Prepare output directory (default: `out` inside project) and logging
project_dir = os.path.dirname(os.path.abspath(__file__))
# Force default save inside the project source folder. Prefer SAVE_DIR if provided
# (deploy.sh sets SAVE_DIR=/app/output which will be mounted from the host project's out dir).
default_out = os.path.join(project_dir, "out")
env_save = os.environ.get("SAVE_DIR")
if env_save:
    save_dir = env_save
else:
    # If no SAVE_DIR provided, fall back to the project's `out` dir inside source tree
    save_dir = default_out

# Ensure save_dir is inside the project source folder. If not, override and warn.
try:
    common = os.path.commonpath([project_dir, os.path.abspath(save_dir)])
except Exception:
    common = None
if common != project_dir:
    # save_dir is outside the project tree — force to project/out to satisfy requirement
    save_dir = default_out
    print(f"WARNING: SAVE_DIR was outside project tree. Forcing save_dir to {save_dir}")

os.makedirs(save_dir, exist_ok=True)

# Configure logging to both stdout and a file under the output dir
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

for i in range(100):
    result = algo.train()
    logger.info("=== Training iteration %d ===", i)
    # Summary of important metrics
    episode_return_mean = result.get('env_runners', {}).get('episode_return_mean', 'N/A')
    episode_len_mean = result.get('env_runners', {}).get('episode_len_mean', 'N/A')
    num_episodes = result.get('env_runners', {}).get('num_episodes', 'N/A')
    num_env_steps_sampled_lifetime = result.get('env_runners', {}).get('num_env_steps_sampled_lifetime', 'N/A')
    time_this_iter_s = result.get('time_this_iter_s', 'N/A')
    total_loss = result.get('learners', {}).get('default_policy', {}).get('total_loss', 'N/A')
    vf_loss = result.get('learners', {}).get('default_policy', {}).get('vf_loss', 'N/A')
    policy_loss = result.get('learners', {}).get('default_policy', {}).get('policy_loss', 'N/A')
    entropy = result.get('learners', {}).get('default_policy', {}).get('entropy', 'N/A')
    
    logger.info("Summary - Episode Return Mean: %s, Episode Len Mean: %s, Num Episodes: %s, Total Steps: %s, Time: %.2f s",
                episode_return_mean, episode_len_mean, num_episodes, num_env_steps_sampled_lifetime, time_this_iter_s)
    logger.info("Losses - Total: %s, VF: %s, Policy: %s, Entropy: %s",
                total_loss, vf_loss, policy_loss, entropy)
    
    # Uncomment the line below to log the full result dict if needed
    logger.info(result)

# ==============================
#  Evaluation
# ==============================
eval_result = algo.evaluate()

# Save a checkpoint into the output directory
checkpoint_path = algo.save(save_dir)
logger.info("Checkpoint saved at: %s", checkpoint_path)

# ==============================
#  Cleanup
# ==============================
algo.stop()
