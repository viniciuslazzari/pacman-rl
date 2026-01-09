import subprocess
import os
from datetime import datetime

# Path to your training script
TRAIN_SCRIPT = "main.py"

# Env runner configurations to test
ENV_RUNNERS_LIST = [1, 2, 4, 8, 16]

# Base output directory
BASE_OUT_DIR = "scaling_runs"

os.makedirs(BASE_OUT_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

for num_env_runners in ENV_RUNNERS_LIST:
    run_name = f"env_runners_{num_env_runners}"
    save_dir = os.path.join(BASE_OUT_DIR, f"{timestamp}_{run_name}")

    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print(f"Running experiment with NUM_ENV_RUNNERS={num_env_runners}")
    print(f"Saving to: {save_dir}")
    print("=" * 60)

    env = os.environ.copy()
    env["NUM_ENV_RUNNERS"] = str(num_env_runners)
    env["SAVE_DIR"] = save_dir

    subprocess.run(
        ["python", TRAIN_SCRIPT],
        env=env,
        check=True
    )

print("\nAll scaling experiments completed.")
