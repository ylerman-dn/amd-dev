#!/usr/bin/env bash
# sweep_single.sh <runset_dir> — base single-node sweep: 3 collectives, full size range, 8 GPUs.
# Kept intentionally simple; the real sweep matrix belongs in the Python harness.
set -uo pipefail
RUNSET="${1:?need runset dir}"
HELPER="$(cd "$(dirname "$0")" && pwd)/rccl_run.sh"
mkdir -p "$RUNSET"
for coll in all_reduce_perf all_gather_perf alltoall_perf; do
  echo ">>> $coll -b 4 -e 8G -f 2 -g 8"
  bash "$HELPER" "$coll" "$RUNSET/${coll}__b4-e8G-f2-g8.log" -b 4 -e 8G -f 2 -g 8
done
echo ">>> single-node sweep complete: $RUNSET"
