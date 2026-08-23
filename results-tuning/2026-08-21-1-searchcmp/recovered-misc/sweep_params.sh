#!/usr/bin/env bash
# sweep_params.sh <runset_dir> — parameter play: all_reduce across NCCL_PROTO x NCCL_ALGO.
# Each run gets a per-run timeout so a known-bad combo (e.g. LL128) can't eat the budget.
set -uo pipefail
RUNSET="${1:?need runset dir}"
HELPER="$(cd "$(dirname "$0")" && pwd)/rccl_run.sh"
COLL=all_reduce_perf
ARGS=(-b 4 -e 256M -f 2 -g 8)
PER_RUN_TIMEOUT=240
mkdir -p "$RUNSET"
for proto in Simple LL LL128; do
  for algo in Ring Tree; do
    export NCCL_PROTO="$proto" NCCL_ALGO="$algo"
    name="${COLL}__b4-e256M-f2-g8__proto-${proto}__algo-${algo}"
    echo ">>> $name"
    timeout --signal=KILL "$PER_RUN_TIMEOUT" bash "$HELPER" "$COLL" "$RUNSET/${name}.log" "${ARGS[@]}" \
      || echo "    (run returned non-zero / timed out — see log)"
  done
done
unset NCCL_PROTO NCCL_ALGO
echo ">>> param sweep complete: $RUNSET"
