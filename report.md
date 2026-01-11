# Report

## Pacman 

We decided to work with the [Pacman environment](https://ale.farama.org/environments/pacman/), which seemed to be an interesting and challenging game.

## Experiments


We chose the Rennes site and we ran the tests on the Paradoxe cluster, composed of 64 nodes, 128 cpus, 3328 cores, split into:

- paradoxe-[1-32] (32 nodes, 64 cpus, 1664 cores)
- paradoxe-[33-64] (32 nodes, 64 cpus, 1664 cores)

due to a difference in the SSD model.

Common specifications:

- Total Cores per Node: 52 Physical Cores (2 CPUs×26 cores).
- Memory: 384 GiB
- Networking: 25 Gbps
- Total Threads per Node: up to 104 (52 x 2 threads with Intel Hyper-Threading).


We tested different scenarios, changing the number of hosts (nodes) and the number of environment runners.

We fixed the following parameters:
- Iterations: 60
- Training batch size: 8000
- Epochs: 10
- Environment per environment runner: 1

We analyzed the perfomance by looking at the following metrics:
- Total training time
- Mean episode return (reward)
- Episode length
- Policy loss
- Value loss
- Entropy
- Environment steps per second

We generated 4 graphs.

Note: The throughput graph excludes the first iteration. The initial measurement included Ray worker initialization and environment setup overhead, resulting in an outlier (X steps/sec) that obscured the steady-state performance metrics.


### Experiment 1: 1 host, 1 environment runner

Master Node: paradoxe-31.rennes.grid5000.fr
Worker Nodes: (none)

- 1 env runner
- 176 min
![Training dashboard](dashboards/training_dashboard_1envrunners.png)


Early iterations (0–10) have smaller rewards (~11–50) and more fluctuations, but from iterations 30–59, rewards are higher and more consistently around 80–130, though some spikes/decreases exist. Mean episode reward increases from approximately 11 in the first iteration to over 90 by the last iteration, indicating effective learning. Some fluctuations remain due to exploration, but the overall trend shows consistent improvement in policy performance.

### Experiment 2: 1 host, 2 environment runners

Master Node: paradoxe-37.rennes.grid5000.fr
Worker Nodes: (none)

- 141 min

The training logs show that the agent initially learns steadily, with average episode rewards rising from around 14 to over 300, indicating effective early learning and exploitation of the environment. Episode lengths also increase, reflecting more complex or sustained behaviors. However, over time the policy entropy drops sharply, especially by the end, signaling that the policy has become almost deterministic. This is accompanied by highly negative policy loss and near-zero value loss, suggesting instability or collapse in training. Overall, while the agent achieves high rewards, the sharp decline in entropy and erratic loss values indicate overfitting and reduced exploration.


![Training dashboard](dashboards/training_dashboard_2envrunners.png)

### Experiment 3: 1 host, 4 environment runners
Master Node: paradoxe-38.rennes.grid5000.fr
Worker Nodes: (none)

- 120 min

![Training dashboard](dashboards/training_dashboard_4envrunners.png)

### Experiment 4: 1 host, 8 environment runners


![Training dashboard](dashboards/training_dashboard_8envrunners.png)

###  Experiment 5: 1 host, 16 environment runners

![Training dashboard](dashboards/training_dashboard_16envrunners.png)

### Experiment 6: 1 host, 24 environment runners

![Training dashboard](dashboards/training_dashboard_24envrunners.png)

### Experiment 7: 1 host, 32 environment runners

![Training dashboard](dashboards/training_dashboard_32envrunners.png)


### Experiment 8: 1 host, 48 environment runners
Master Node: paradoxe-34.rennes.grid5000.fr
Worker Nodes: (none)

"total_training_time_min": 106.57946914085,

![Training dashboard](dashboards/training_dashboard_48envrunners.png)

### Experiment 9: 1 host, 64 environment runners

Master Node: paradoxe-34.rennes.grid5000.fr
Worker Nodes: (none)

"total_training_time_min": 107.19128965038335,

![Training dashboard](dashboards/training_dashboard_64envrunners.png)

### Experiment 10: 1 host, 96 environment runners

Master Node: paradoxe-38.rennes.grid5000.fr
Worker Nodes: (none)

"total_training_time_min": 109.33933907701666,

![Training dashboard](dashboards/training_dashboard_96envrunners.png)

### Experiment 11: 1 host, 104 environment runners

Master Node: paradoxe-38.rennes.grid5000.fr
Worker Nodes: (none)

It was not possible to scale up to 104 environment runners with a single host.

- The following resource request cannot be scheduled right now: {'CPU': 1.0}. 

### Experiment 12: 2 hosts, 2 environment runners

### Experiment 13: 2 hosts, 4 environment runners

### Experiment 14: 2 hosts, 8 environment runners



### Experiment 15: 2 host, 16 environment runners

### Experiment 16: 2 host, 24 environment runners

### Experiment 17: 2 host, 32 environment runners

### Experiment 18: 2 host, 48 environment runners


### Experiment 19: 2 host, 64 environment runners



### Experiment 20: 2 host, 96 environment runners



### Experiment 21: 2 host, 104 environment runners


Master Node: paradoxe-34.rennes.grid5000.fr
Worker Nodes: paradoxe-38.rennes.grid5000.fr

## Contributions

- Laura Keidann:
- Vinicius Lazzari: 