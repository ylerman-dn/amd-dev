#!/bin/bash
# fetch one finished node's tree (bench logs, hit-maps, receipts; no rccl logs, no server.log) and score every pass.
# Usage: fetch_node.sh <model tag> <node> <jobid>      (run on the dev VM)
set -eu
TAG=$1; NODE=$2; JOB=$3
D=$(dirname "$(readlink -f "$0")")
REPO=$(cd "$D/../.." && pwd)
ssh -o BatchMode=yes amd-mi355x-1 "srun --overlap --jobid=$JOB -N1 -w $NODE tar -C /data/ylerman/inmodel-rocm10-2026-09-10 -cf - --exclude='*/logs' --exclude='server.log' $TAG" < /dev/null | tar -C "$D" -xf -
for P in "$D/$TAG"/*/pass1 "$D/$TAG"/*/pass2; do
  [ -d "$P" ] || continue
  python3 "$REPO/tools/rccl-sweep/infer_score.py" --base "$P" --prefix "" --ref stock_def > "$P/score.txt" 2>&1 || echo "score failed: $P"
  echo "## $P"; cut -d, -f1-7 "$P/score.csv" 2>/dev/null
done
echo "fetched $TAG from $NODE"
