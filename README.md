# Pacman RL Training with Ray

This project implements a reinforcement learning (RL) agent using Ray's PPO algorithm to play Pacman. The training is distributed across a cluster using Ray, allowing scalability to multiple nodes.

## Overview

The agent uses a convolutional neural network (CNN) to process Pacman game images and learn to maximize the score through reinforcement learning. Training is executed in parallel using Ray, with support for multiple nodes in an OAR cluster.

## Requirements

- Python 3.9+
- Access to a cluster with OAR (or similar)
- NFS shared between nodes (for file synchronization)
- Dependencies listed in `requirements.txt`

## Setup

### 1. Cloning the Repository

```bash
git clone https://github.com/viniciuslazzari/pacman-rl.git
cd pacman-rl
```

### 2. Creating the Virtual Environment

```bash
python3 -m venv .env
source .env/bin/activate
```

### 3. Installing Dependencies

```bash
pip install -r requirements.txt
```

Make sure the virtual environment is activated throughout the execution.

## Running the Training

### Distributed Mode (OAR Cluster)

To run on a cluster with multiple nodes:

```bash
oarsub -S ./deploy.sh
```

This command:
- Requests 2 nodes from the cluster (configured in `#OAR -l host=2/core=1`)
- Activates the virtual environment on each node
- Starts the Ray head on the master node
- Connects worker nodes to the head
- Executes distributed training
- Collects logs and checkpoints in the `out/` directory

### Local Mode (Single Node)

To test locally on a single node:

```bash
bash deploy.sh
```

This will run training in single-node mode, useful for development and quick tests.

## Outputs and Results

All artifacts are saved in the `out/` directory (shared via NFS):

- `train.log`: Detailed training logs
- `console.log`: Standard output from the training process
- `metadata.json`: Information about the training (checkpoint, final metrics)
- `ray_session_latest/`: Detailed Ray logs (automatically copied)

### Monitoring

During training, you can access the Ray dashboard at `http://<master-ip>:8265` to view real-time metrics.

## Advanced Configuration

### Environment Variables

You can customize behavior through environment variables:

- `VENV_ACTIVATE`: Path to the virtualenv activation script (default: `./.env/bin/activate`)
- `SAVE_DIR`: Directory to save outputs (default: `./out`)
- `RAY_PORT`: Ray head port (default: 6379)
- `DASH_PORT`: Dashboard port (default: 8265)
- `RAY_TMP_DIR`: Ray temporary directory (default: `/tmp/ray`)

Example:

```bash
VENV_ACTIVATE=/path/to/venv/bin/activate SAVE_DIR=/mnt/nfs/results oarsub -S ./deploy.sh
```

### Model Architecture

The model uses a CNN with the following layers:

- Conv1: 32 filters, 8x8 kernel, stride 4
- Conv2: 64 filters, 4x4 kernel, stride 2
- Conv3: 128 filters, 3x3 kernel, stride 1
- Dense: 512 neurons (2 layers)

To modify the architecture, edit the `main.py` file.

## Troubleshooting

### Error: "source: filename argument required"

Ensure the path to `VENV_ACTIVATE` is correct and that the `activate` file exists.

### Error: "ray: command not found"

Check if the virtual environment is activated correctly and if Ray is installed.

### Training Does Not Start

Check logs in `out/console.log` and `OAR_*.err` for error messages.

### NFS Not Shared

If nodes do not share NFS, manually copy project files to each node before running.

## Contributing

To contribute:
1. Fork the repository
2. Create a branch for your feature
3. Commit your changes
4. Open a pull request

## License

This project is licensed under the MIT License.