#!/bin/bash
#OAR -l host=2,walltime=1:00:00
#OAR -p paradoxe

# Deploy script to build a Docker image on each allocated node,
# start a Ray head on the first node and start worker containers on the others.

set -euo pipefail

IMAGE_NAME="ray-pacman:latest"
REPO_DIR="$(pwd)"

if [ -z "${OAR_NODEFILE:-}" ]; then
    echo "ERROR: OAR_NODEFILE not set. Run inside an OAR job script."
    exit 1
fi

nodes=( $(uniq "$OAR_NODEFILE") )
if [ ${#nodes[@]} -eq 0 ]; then
    echo "No nodes found in $OAR_NODEFILE"
    exit 1
fi

head_node=${nodes[0]}
worker_nodes=( "${nodes[@]:1}" )

WORKERS_PER_NODE=${WORKERS_PER_NODE:-1}

echo "Building image on each node: $IMAGE_NAME"
for node in "${nodes[@]}"; do
    echo "*** Preparing $node ***"
    oarsh $node sudo-g5k || true
    oarsh $node g5k-setup-docker -t || true
    echo "*** Building image on $node (this may take a while) ***"
    oarsh $node bash -lc "docker build -t $IMAGE_NAME '$REPO_DIR'"
done



HEAD_OUTPUT_DIR=${HEAD_OUTPUT_DIR:-$REPO_DIR/out}
HEAD_LOG_DIR=${HEAD_LOG_DIR:-$REPO_DIR/out/ray_logs}

echo "Starting Ray head on $head_node (host output: $HEAD_OUTPUT_DIR, logs: $HEAD_LOG_DIR)"
oarsh $head_node bash -lc "docker rm -f ray_head || true"
# Defensive checks: ensure variables are non-empty to avoid `mkdir: missing operand`.
if [ -z "${HEAD_OUTPUT_DIR:-}" ] || [ -z "${HEAD_LOG_DIR:-}" ]; then
    echo "ERROR: HEAD_OUTPUT_DIR or HEAD_LOG_DIR is empty (HEAD_OUTPUT_DIR='$HEAD_OUTPUT_DIR', HEAD_LOG_DIR='$HEAD_LOG_DIR')"
    exit 1
fi

echo "Creating host dirs on head: '$HEAD_OUTPUT_DIR' and '$HEAD_LOG_DIR'"
# Use explicit quoting when invoking the remote shell so paths with spaces still work.
oarsh $head_node bash -lc 'mkdir -p "'"$HEAD_OUTPUT_DIR"'" "'"$HEAD_LOG_DIR"'" && chmod 777 "'"$HEAD_OUTPUT_DIR"'" "'"$HEAD_LOG_DIR"'"'
# Mount the project `out` directory from the host into the container at /app/output
# and set SAVE_DIR so the application writes directly into the project's source folder.
oarsh $head_node bash -lc 'docker run -d --name ray_head -v "'"$HEAD_OUTPUT_DIR"'":/app/output -v "'"$HEAD_LOG_DIR"'":/tmp/ray --env NODE_ROLE=head --env SAVE_DIR=/app/output --network host $IMAGE_NAME'

# Get head IP (first non-loopback address)
HEAD_IP=$(oarsh $head_node bash -lc "hostname -I | awk '{print \$1}'")
if [ -z "$HEAD_IP" ]; then
    echo "Failed to determine head IP"
    exit 1
fi
RAY_ADDRESS="$HEAD_IP:6379"
echo "Ray head address: $RAY_ADDRESS"

echo "Starting workers on ${#worker_nodes[@]} node(s)"
for node in "${worker_nodes[@]}"; do
    for i in $(seq 1 $WORKERS_PER_NODE); do
        cname="ray_worker_${node//./_}_$i"
        echo "Starting worker container $cname on $node"
        oarsh $node bash -lc "docker rm -f $cname || true"
        oarsh $node bash -lc "docker run -d --name $cname --env NODE_ROLE=worker --env RAY_HEAD_ADDRESS=$RAY_ADDRESS --network host $IMAGE_NAME"
    done
done

echo "Deployment complete. Ray head: $head_node ($RAY_ADDRESS)"
echo "Dashboard should be available at http://$head_node:8265 (if accessible from your network)"

# --- Try to retrieve artifacts automatically to the machine that ran this script ---
LOCAL_OUTPUT_DIR=${LOCAL_OUTPUT_DIR:-$(pwd)/ray_results_$(date +%s)}
mkdir -p "$LOCAL_OUTPUT_DIR"

echo "Attempting to stream and save outputs/logs from head ($head_node) to local: $LOCAL_OUTPUT_DIR"

echo "Streaming saved checkpoints (HEAD_OUTPUT_DIR -> $LOCAL_OUTPUT_DIR/outputs.tar.gz)"
oarsh $head_node bash -lc "test -d '$HEAD_OUTPUT_DIR' && tar -C '$HEAD_OUTPUT_DIR' -czf - ." > "$LOCAL_OUTPUT_DIR/outputs.tar.gz" 2>/dev/null || echo "Warning: failed to stream outputs from $head_node (check $HEAD_OUTPUT_DIR on the head)"

echo "Streaming Ray logs (HEAD_LOG_DIR -> $LOCAL_OUTPUT_DIR/logs.tar.gz)"
oarsh $head_node bash -lc "test -d '$HEAD_LOG_DIR' && tar -C '$HEAD_LOG_DIR' -czf - ." > "$LOCAL_OUTPUT_DIR/logs.tar.gz" 2>/dev/null || echo "Warning: failed to stream logs from $head_node (check $HEAD_LOG_DIR on the head)"

echo "Artifacts saved (if available) under: $LOCAL_OUTPUT_DIR"
echo "If streaming failed, you can retrieve files manually via scp or by inspecting the head node:
    oarsh $head_node ls -lah $HEAD_OUTPUT_DIR
    oarsh $head_node ls -lah $HEAD_LOG_DIR"

