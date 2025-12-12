#!/bin/bash
#OAR -l host=30,walltime=3:00:00
#OAR -p paradoxe
#OAR -O OAR_%jobid%.out
#OAR -E OAR_%jobid%.err

set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VENV_ACTIVATE="${VENV_ACTIVATE:-$PROJECT_DIR/.env/bin/activate}"
SAVE_DIR="${SAVE_DIR:-$PROJECT_DIR/out}"
RAY_PORT=${RAY_PORT:-6379}
DASH_PORT=${DASH_PORT:-8265}
RAY_TMP_DIR=${RAY_TMP_DIR:-/tmp/ray}

mkdir -p "$SAVE_DIR"

remote_exec() {
    local host="$1"
    shift
    local cmd="$*"

    local wrapped="
        bash -lc '
            source \"$VENV_ACTIVATE\" >/dev/null 2>&1
            $cmd
        '
    "

    if command -v oarsh >/dev/null 2>&1; then
        oarsh "$host" "$wrapped"
    else
        ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$host" "$wrapped"
    fi
}

if [[ -n "${OAR_NODEFILE-}" && -f "$OAR_NODEFILE" ]]; then
    mapfile -t HOSTS < <(sort -u "$OAR_NODEFILE")
else
    echo "AVISO: OAR_NODEFILE não encontrado. Rodando em modo de nó único."
    HOSTS=("$(hostname -f)")
fi

MASTER="${HOSTS[0]}"
WORKERS=("${HOSTS[@]:1}")

echo "========================================"
echo " Início do Deploy"
echo "========================================"
echo "Diretório do Projeto: $PROJECT_DIR"
echo "Nó Master: $MASTER"
echo "Nós Workers: ${WORKERS[*]:-(nenhum)}"
echo "----------------------------------------"

cleanup() {
    echo "Limpando Ray..."
    for h in "${HOSTS[@]}"; do
        remote_exec "$h" "ray stop || true"
    done
}
trap cleanup EXIT INT TERM

MASTER_IP_RAW=$(remote_exec "$MASTER" "hostname -I")
set -- $MASTER_IP_RAW
MASTER_IP="$1"
echo "IP do Master: $MASTER_IP"

echo "Iniciando Ray head no master..."
remote_exec "$MASTER" "
    ray stop || true;
    ray start --head \
        --port=$RAY_PORT \
        --dashboard-port=$DASH_PORT \
        --temp-dir=$RAY_TMP_DIR \
        --disable-usage-stats
"

sleep 5

if [ ${#WORKERS[@]} -gt 0 ]; then
    echo "Iniciando workers..."
    for worker in "${WORKERS[@]}"; do
        echo " - Worker: $worker"
        remote_exec "$worker" "
            ray stop || true;
            ray start --address=$MASTER_IP:$RAY_PORT \
                --disable-usage-stats
        " &
    done
    wait
else
    echo "Nenhum worker para iniciar."
fi

sleep 3

echo "----------------------------------------"
echo "Iniciando o treinamento no master..."

remote_exec "$MASTER" "
    cd \"$PROJECT_DIR\"
    export SAVE_DIR=\"$SAVE_DIR\"
    python3 main.py
" > "$SAVE_DIR/console.log" 2>&1   # <-- FIXED REDIRECT

echo "Treinamento finalizado."
echo "----------------------------------------"

echo "Resultados e logs estão em: $SAVE_DIR"
ls -lah "$SAVE_DIR"

echo "========================================"
echo " Deploy Finalizado"
echo "========================================"
