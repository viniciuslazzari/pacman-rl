#!/bin/bash
#OAR -l host=2,walltime=1:00:00
#OAR -p paradoxe

# Deploy script to build a Docker image, start a Ray head on the first node,
# and start worker containers on the others across OAR-allocated nodes.

set -euo pipefail

# --- Configuration ---
IMAGE_NAME="ray-pacmann:latest"
REPO_DIR="$(pwd)"
WORKERS_PER_NODE=${WORKERS_PER_NODE:-1}
LOCAL_OUTPUT_DIR=${LOCAL_OUTPUT_DIR:-$(pwd)/ray_results_$(date +%s)}
# SSH options used for non-interactive connections to nodes
SSH_OPTS='-o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null'
# ---------------------

# 1. Input Validation and Node Identification
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

HEAD_CONTAINER_NAME="ray_head"
RAY_PORT="6375"

HEAD_TMP_OUT=""
HEAD_TMP_LOGS=""
# Keep PIDs of local background log streams to clean up on exit
STREAM_PIDS=()

# 2. Cleanup Function and Trap
# Ensure containers and remote temp dirs are cleaned up on script exit
cleanup() {
    echo -e "\n--- Cleaning up containers and remote directories ---"
    # stop any local background streaming
    if [ ${#STREAM_PIDS[@]} -gt 0 ]; then
        echo "Stopping local log streams..."
        kill "${STREAM_PIDS[@]}" >/dev/null 2>&1 || true
    fi
    # Remove head container
    ssh $SSH_OPTS "$head_node" docker rm -f "$HEAD_CONTAINER_NAME" >/dev/null 2>&1 || true
    # Remove worker containers (by name prefix is harder, so we rely on -f being idempotent)
    for node in "${worker_nodes[@]}"; do
        ssh $SSH_OPTS "$node" docker ps -a --filter name="ray_worker" --format '{{.Names}}' | while read -r cname; do
            ssh $SSH_OPTS "$node" docker rm -f "$cname" >/dev/null 2>&1 || true
        done
    done
    
    # Remove temp dirs
    if [ ! -z "$HEAD_TMP_OUT" ]; then
        ssh $SSH_OPTS "$head_node" rm -rf "$HEAD_TMP_OUT"
    fi
    if [ ! -z "$HEAD_TMP_LOGS" ]; then
        ssh $SSH_OPTS "$head_node" rm -rf "$HEAD_TMP_LOGS"
    fi
    echo "Cleanup complete."
}
trap cleanup EXIT # Execute cleanup function on exit (success or failure)

# 3. Setup and Image Build (parallel per-node builds)
echo "--- Preparing nodes and building Docker image in parallel: $IMAGE_NAME ---"
mkdir -p "$LOCAL_OUTPUT_DIR/logs"

pids=()
for node in "${nodes[@]}"; do
    echo "*** Scheduling build on $node ***"
    (
        echo "*** Preparing G5K environment on $node ***"
        ssh $SSH_OPTS "$node" sudo-g5k || true
        ssh $SSH_OPTS "$node" g5k-setup-docker -t || true
        echo "*** Building image on $node (this may take a while) ***"
        # Run diagnostics and docker build remotely; try sudo on failure
        ssh $SSH_OPTS "$node" bash -lc "set -x
which docker || true
docker --version || true
id || true
umask || true
echo '--- listing REPO_DIR contents ---'
ls -la '$REPO_DIR' || true
echo '--- running docker build ---'
docker build -t '$IMAGE_NAME' '$REPO_DIR' || { echo 'docker build failed; trying sudo'; sudo docker build -t '$IMAGE_NAME' '$REPO_DIR' || echo 'sudo docker build failed'; }
"  
    ) > "$LOCAL_OUTPUT_DIR/logs/build_${node//./_}.log" 2>&1 &
    pids+=("$!")
done

# Wait for all parallel builds to complete
for pid in "${pids[@]}"; do
    wait "$pid" || echo "Warning: one parallel build job failed (pid=$pid)"
done

# 4. Start Ray Head
echo -e "\n--- Starting Ray Head on $head_node ---"

# Create node-local temp dirs on head to collect outputs/logs (no NFS)
HEAD_TMP_OUT=$(ssh $SSH_OPTS "$head_node" bash -lc 'mktemp -d /tmp/ray_out_XXXX')
HEAD_TMP_LOGS=$(ssh $SSH_OPTS "$head_node" bash -lc 'mktemp -d /tmp/ray_logs_XXXX')
echo "Remote temp dirs: out=$HEAD_TMP_OUT logs=$HEAD_TMP_LOGS"

# Run head container binding those tmp dirs
ssh $SSH_OPTS "$head_node" docker run -d \
    --name "$HEAD_CONTAINER_NAME" \
    -v "$HEAD_TMP_OUT":/app/output \
    -v "$HEAD_TMP_LOGS":/tmp/ray \
    --env NODE_ROLE=head \
    --env SAVE_DIR=/app/output \
    --network host \
    "$IMAGE_NAME" || { echo "ERROR: Failed to start Ray head container."; exit 1; }

# Get head IP and define Ray address
HEAD_IP=$(ssh $SSH_OPTS "$head_node" hostname -I | awk '{print $1}')
RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"
echo "Ray head address: $RAY_ADDRESS"

# Stream head container logs back to the submit host so we can follow training
mkdir -p "$LOCAL_OUTPUT_DIR/logs"
if [ "${STREAM_HEAD_LOGS:-1}" != "0" ]; then
    echo "Streaming head logs from $head_node to $LOCAL_OUTPUT_DIR/logs/head_${head_node//./_}.log"
    ssh $SSH_OPTS "$head_node" bash -lc "docker logs -f $HEAD_CONTAINER_NAME" | sed "s|^|[$head_node] |" | tee "$LOCAL_OUTPUT_DIR/logs/head_${head_node//./_}.log" &
    STREAM_PIDS+=("$!")
fi

# 5. Start Ray Workers
echo -e "\n--- Starting Ray Workers ---"
for node in "${worker_nodes[@]}"; do
    for i in $(seq 1 "$WORKERS_PER_NODE"); do
        cname="ray_worker_${node//./_}_$i"
        echo "Starting worker $cname on $node"
        
        ssh $SSH_OPTS "$node" docker run -d \
            --name "$cname" \
            --env NODE_ROLE=worker \
            --env RAY_HEAD_ADDRESS="$RAY_ADDRESS" \
            --network host \
            "$IMAGE_NAME" || echo "Warning: worker failed on $node"
        # Optionally stream worker logs
        if [ "${STREAM_WORKER_LOGS:-0}" != "0" ]; then
            echo "Streaming logs for worker $cname on $node -> $LOCAL_OUTPUT_DIR/logs/worker_${cname}.log"
            ssh $SSH_OPTS "$node" bash -lc "docker logs -f $cname" | sed "s|^|[$node][$cname] |" | tee "$LOCAL_OUTPUT_DIR/logs/worker_${cname}.log" &
            STREAM_PIDS+=("$!")
        fi
    done
done

echo -e "\n--- Deployment complete. Ray head: $head_node ---"

# 6. Fetch Artifacts
echo -e "\n--- Fetching artifacts to local machine ($LOCAL_OUTPUT_DIR) ---"
mkdir -p "$LOCAL_OUTPUT_DIR/outputs" "$LOCAL_OUTPUT_DIR/logs"

# Function to pull directory contents
pull_dir() {
    local remote_dir=$1
    local local_dir=$2
    local dir_type=$3

    echo "Pulling $dir_type from head: $remote_dir -> $local_dir"

    # Save a remote listing for diagnostics
    mkdir -p "$LOCAL_OUTPUT_DIR/logs"
        echo "--- remote listing for $remote_dir ---" >> "$LOCAL_OUTPUT_DIR/logs/pull_${dir_type}.log"
        # Use a heredoc sent to remote bash to avoid nested-quoting issues
        ssh $SSH_OPTS "$head_node" bash -s -- "$remote_dir" >> "$LOCAL_OUTPUT_DIR/logs/pull_${dir_type}.log" 2>&1 <<'REMOTE'
remote_dir="$1"
if [ -d "$remote_dir" ]; then
    ls -la "$remote_dir"
else
    echo 'NO_DIR'
fi
REMOTE

        # Count files before attempting tar; if zero, skip tar and warn
        file_count=$(ssh $SSH_OPTS "$head_node" bash -s -- "$remote_dir" 2>/dev/null <<'REMOTE'
remote_dir="$1"
if [ -d "$remote_dir" ]; then
    find "$remote_dir" -type f | wc -l
else
    echo 0
fi
REMOTE
)
    if [ "${file_count:-0}" -eq 0 ]; then
        echo "Warning: no files to pull in $remote_dir (count=0)" | tee -a "$LOCAL_OUTPUT_DIR/logs/pull_${dir_type}.log"
        return 0
    fi

    # Stream the directory contents via tar
        ssh $SSH_OPTS "$head_node" bash -s -- "$remote_dir" <<'REMOTE' | tar -C "$local_dir" -xf - || { echo "Warning: Failed to pull $dir_type" | tee -a "$LOCAL_OUTPUT_DIR/logs/pull_${dir_type}.log"; }
remote_dir="$1"
tar -C "$remote_dir" -cf - .
REMOTE
}

pull_dir "$HEAD_TMP_OUT" "$LOCAL_OUTPUT_DIR/outputs" "outputs"
pull_dir "$HEAD_TMP_LOGS" "$LOCAL_OUTPUT_DIR/logs" "logs"

echo -e "\n✅ Artifacts available under: $LOCAL_OUTPUT_DIR"
# The cleanup trap will run automatically now