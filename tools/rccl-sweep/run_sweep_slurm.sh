#!/bin/bash
# GPU-107 Option-2 sweep launcher (Slurm-aware).
# Reserves N XAI nodes via Slurm, points the sweep at exactly those nodes (writes servers.txt),
# runs rccl_autotune, records wall-time to a RUNLOG, then releases. Run ON a cluster node.
#
# Usage:  bash run_sweep_slurm.sh <N_nodes_to_reserve> [extra rccl_autotune args...]
#   e.g.  bash run_sweep_slurm.sh 1 -n 1 -c 32:64:32 --collective all_reduce --algo RING --proto SIMPLE --min-size 1M --max-size 512M
#   e.g.  bash run_sweep_slurm.sh 3 --config autotune_config.yaml     # full sweep, nodes 1-3
# NOTE: pass EITHER --config <file> OR explicit CLI flags (with --config, CLI sweep args are ignored).
# Env:    MY_PATH (binaries dir; default our current-RCCL bin), RUNLOG (csv path)
set -u
SWEEP_DIR="$(cd "$(dirname "$0")" && pwd)"
export MY_PATH="${MY_PATH:-/opt/shared/ylerman/GPU-107/bin}"
RUNLOG="${RUNLOG:-/opt/shared/ylerman/GPU-107/opt2-sweep/RUNLOG.csv}"
N="${1:?usage: run_sweep_slurm.sh N [extra args]}"; shift || true
EXTRA="$*"

mkdir -p "$(dirname "$RUNLOG")"
[ -f "$RUNLOG" ] || echo "utc_start,label,reserve_nodes,extra_args,duration_s,status,jobid,nodelist" > "$RUNLOG"

echo "[wrap] reserving $N XAI node(s) via salloc..."
O=$(salloc --no-shell -p XAI -N "$N" --exclude=amd-mi355x-2,amd-mi355x-9 --gres=gpu:8 -t 480 -J ylerman 2>&1)
JID=$(echo "$O" | grep -oE "job allocation [0-9]+" | grep -oE "[0-9]+" | tail -1)
if [ -z "$JID" ]; then echo "[wrap] NOALLOC: $O"; exit 1; fi
NODES=$(scontrol show hostnames "$(squeue -j "$JID" -h -o %N)")
NODES_1LINE=$(echo $NODES | tr '\n' ' ')
echo "[wrap] JID=$JID nodes: $NODES_1LINE"

# Point the sweep at exactly the allocated nodes (IP per node, in order).
: > "$SWEEP_DIR/servers.txt"
for n in $NODES; do
  # pick the cluster mgmt IP (172.30.160.x), never loopback (127.x) which /etc/hosts also maps
  ip=$(getent hosts "$n" | awk '{print $1}' | grep -E '^172\.30\.160\.' | head -1)
  [ -z "$ip" ] && ip=$(getent hosts "$n" | awk '{print $1}' | grep -vE '^127\.' | head -1)
  echo "$ip # $n" >> "$SWEEP_DIR/servers.txt"
done
echo "[wrap] servers.txt ->"; sed 's/^/[wrap]   /' "$SWEEP_DIR/servers.txt"

# --- Keep-alive step (cheap insurance; NOT the real failure mode) -----------
# CORRECTION: earlier sweeps died from NODE_FAIL during transient cluster infra
# instability (node/NFS/slurmctld wobbles that also killed OTHER users' jobs at
# the same instant) -- NOT from Slurm inactivity-reaping. On this cluster
# InactiveLimit=0 (disabled) and PreemptMode=OFF, so an idle --no-shell alloc is
# NOT auto-cancelled. This trivial background srun step (no GPUs, --overlap) just
# keeps one visibly-active step; it is harmless but does NOTHING against
# NODE_FAIL. The real safeguard against NODE_FAIL is external: watch for the job
# dying and relaunch (sweep-watcher cron) + archive results as we go.
echo "[wrap] launching keep-alive srun step (visible active step; insurance only)..."
srun --jobid="$JID" --overlap --gres=none -N"$N" --ntasks-per-node=1 sleep 100000 >/dev/null 2>&1 &
KA_PID=$!
sleep 3
echo "[wrap] keep-alive launched (local pid $KA_PID); steps now:"
squeue -s -j "$JID" 2>/dev/null | sed 's/^/[wrap]   /' || true

cd "$SWEEP_DIR"
LABEL="autotune_${N}n"
OUT="$SWEEP_DIR/${LABEL}_$(date +%Y%m%d_%H%M%S).out"
echo "[wrap] MY_PATH=$MY_PATH"
echo "[wrap] running: python3 rccl_autotune.py $EXTRA"
t0=$(date +%s); ts=$(date -u +%FT%TZ)
python3 rccl_autotune.py $EXTRA > "$OUT" 2>&1
rc=$?
t1=$(date +%s); dur=$((t1-t0))
status=$([ "$rc" -eq 0 ] && echo ok || echo "FAIL_rc$rc")
echo "${ts},${LABEL},${N},\"${EXTRA}\",${dur},${status},${JID},\"${NODES_1LINE}\"" >> "$RUNLOG"
echo "[wrap] DONE rc=$rc dur=${dur}s  out=$OUT"
echo "[wrap] RUNLOG row appended -> $RUNLOG"

kill "$KA_PID" 2>/dev/null   # stop keep-alive step before releasing
scancel "$JID" && echo "[wrap] released $JID"
echo "WRAP_DONE rc=$rc"
