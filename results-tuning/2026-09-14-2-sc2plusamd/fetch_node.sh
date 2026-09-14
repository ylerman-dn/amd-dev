#!/bin/bash
# fetch one finished chain's tree of campaign C (bench logs, hit-maps, receipts; no rccl logs, no server.log) and score every pass vs none_def.
# Usage: fetch_node.sh <model tag> <node> <jobid>      (run on the dev VM)
set -eu
TAG=$1; NODE=$2; JOB=$3
D=$(dirname "$(readlink -f "$0")")
REPO=$(cd "$D/../.." && pwd)
ssh -o BatchMode=yes amd-mi355x-1 "srun --overlap --jobid=$JOB -N1 -w $NODE tar -C /data/ylerman/inmodel-sc2plus-2026-09-14 -cf - --exclude='*/logs' --exclude='server.log' --exclude='CORRUPT_*' $TAG" < /dev/null | tar -C "$D" -xf -
for P in "$D/$TAG"/*/pass1 "$D/$TAG"/*/pass2; do
  [ -d "$P" ] || continue
  python3 "$REPO/tools/rccl-sweep/infer_score.py" --base "$P" --prefix "" --ref none_def > "$P/score.txt" 2>&1 || echo "score failed: $P"
  echo "## $P"; cut -d, -f1-7 "$P/score.csv" 2>/dev/null
done
echo "fetched $TAG from $NODE"
