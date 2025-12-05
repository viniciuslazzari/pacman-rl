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
    .env_runners(num_env_runners=2)
    .rl_module(
        model_config=DefaultModelConfig(
            conv_activation="relu",
            conv_filters=[
                [16, [8, 8], 4],
                [32, [4, 4], 2],
                [64, [3, 3], 1],
            ],
            head_fcnet_hiddens=[256]
        )
    )
    .training(
        lr=0.0002,
        train_batch_size_per_learner=200,
        num_epochs=10,
    )
    .evaluation(
        evaluation_interval=1,
        evaluation_num_env_runners=1
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

for i in range(1):
    result = algo.train()
    logger.info("=== Training iteration %d ===", i)
    try:
        logger.info("Mean Episode Return: %s", result['evaluation']['env_runners']['episode_return_mean'])
        logger.info("Mean Episode Length: %s", result['evaluation']['env_runners']['episode_len_mean'])
        logger.info("Total Loss: %s", result['learners']['default_policy']['total_loss'])
        logger.info("VF Loss: %s", result['learners']['default_policy']['vf_loss'])
        logger.info("Policy Loss: %s", result['learners']['default_policy']['policy_loss'])
    except Exception:
        logger.exception("Failed to log some training metrics")

# ==============================
#  Evaluation
# ==============================
eval_result = algo.evaluate()

# Save a checkpoint into the output directory
checkpoint_path = algo.save(save_dir)
logger.info("Checkpoint saved at: %s", checkpoint_path)

# Save metadata about the run
metadata = {
    "checkpoint_path": checkpoint_path,
    "eval_result_summary": {
        "episode_return_mean": eval_result.get('evaluation', {}).get('env_runners', {}).get('episode_return_mean'),
        "episode_len_mean": eval_result.get('evaluation', {}).get('env_runners', {}).get('episode_len_mean')
    },
    "timestamp": datetime.utcnow().isoformat() + "Z"
}
with open(os.path.join(save_dir, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

# Try to copy Ray session logs (if present) into the output folder for easier access
ray_session_src = os.path.join("/tmp", "ray", "session_latest")
ray_session_dst = os.path.join(save_dir, "ray_session_latest")
try:
    if os.path.exists(ray_session_src):
        shutil.copytree(ray_session_src, ray_session_dst, dirs_exist_ok=True)
        logger.info("Copied Ray session logs to: %s", ray_session_dst)
    else:
        logger.info("No Ray session directory (%s) found to copy", ray_session_src)
except Exception:
    logger.exception("Failed to copy Ray session logs")

# ==============================
#  Cleanup
# ==============================
algo.stop()
