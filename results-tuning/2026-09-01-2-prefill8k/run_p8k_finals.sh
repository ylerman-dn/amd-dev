#!/bin/bash
# campaign7 phase 3: prefill-8k finals — top-3 screening combos + norule(def) + current conf,
# 3 rotated passes × 10 quiet reps per arm. Restart-safe (skips arm-pass with 10 bench logs).
# TOP3 placeholders are substituted at deploy time.
set -u
DIR=/opt/shared/ylerman/GPU-107/infer-2026-08-30
MODEL=/data/mlperf_llama31_8b/model
LOG=/data/ylerman/models-2026-08-30/campaign7.log
BASE=/data/ylerman/models-2026-08-30

# arm spec: <label>:<armtype>:<conf-or-->
ARMS=(
  "norule:def:-"
  "current:tun:all_reduce_1n.final.conf"
  "tree_ll_48:tun:sw_p8k_tree_ll_48.conf"
  "ring_simple_16:tun:sw_p8k_ring_simple_16.conf"
  "tree_ll_24:tun:sw_p8k_tree_ll_24.conf"
)

run_arm() {
  local spec=$1 pass=$2
  local label=${spec%%:*}
  local rest=${spec#*:}
  local armtype=${rest%%:*}
  local conf=${rest#*:}
  local name="p8kfin_${label}_p${pass}"
  local out="$BASE/${name}_${armtype}"
  local n
  n=$(ls "$out"/bench_rep*.log 2>/dev/null | wc -l)
  if [ "$n" -ge 10 ]; then
    echo "=== SKIP $name (already $n reps) $(date -u +%FT%TZ) ===" >> "$LOG"
    return 0
  fi
  echo "=== ARM $name arm=$armtype conf=$conf start $(date -u +%FT%TZ) ===" >> "$LOG"
  local envconf=()
  if [ "$armtype" = "tun" ]; then envconf=(RM_CONF="$conf"); fi
  if ! env RM_SUBSYS=INIT,ENV ${envconf[@]+"${envconf[@]}"} "$DIR/run_many_p8k.sh" "$MODEL" "$name" "$armtype" 10 >> "$LOG" 2>&1; then
    echo "=== RETRY $name after server failure $(date -u +%FT%TZ) ===" >> "$LOG"
    if ! env RM_SUBSYS=INIT,ENV ${envconf[@]+"${envconf[@]}"} "$DIR/run_many_p8k.sh" "$MODEL" "$name" "$armtype" 10 >> "$LOG" 2>&1; then
      echo "=== SKIPPED $name: server failed twice $(date -u +%FT%TZ) ===" >> "$LOG"
    fi
  fi
  echo "=== ARM $name end $(date -u +%FT%TZ) ===" >> "$LOG"
}

echo "=== campaign7 FINALS start $(date -u +%FT%TZ) ===" >> "$LOG"
NA=${#ARMS[@]}
for pass in 1 2 3; do
  echo "=== FINALS pass $pass $(date -u +%FT%TZ) ===" >> "$LOG"
  off=$((pass - 1))
  for i in $(seq 0 $((NA - 1))); do
    run_arm "${ARMS[$(((i + off) % NA))]}" "$pass"
  done
done
echo "=== campaign7 FINALS complete $(date -u +%FT%TZ) ===" >> "$LOG"
