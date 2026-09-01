#!/bin/bash
# campaign7 phase 2: prefill-8k screening — 15 valid combos × 5 quiet reps, serial.
# Restart-safe: skips any arm whose outdir already holds 5 bench_rep*.log files.
# Combos exclude ring_ll128, tree_ll128, tree_simple (plugin-IGNOREd, probe-proven 2026-08-31).
set -u
DIR=/opt/shared/ylerman/GPU-107/infer-2026-08-30
MODEL=/data/mlperf_llama31_8b/model
LOG=/data/ylerman/models-2026-08-30/campaign7.log
BASE=/data/ylerman/models-2026-08-30

echo "=== campaign7 SCREENING start $(date -u +%FT%TZ) ===" >> "$LOG"
for combo in ring_ll ring_simple tree_ll; do
  for ch in 8 16 24 32 48; do
    name="p8kscr_${combo}_${ch}"
    conf="sw_p8k_${combo}_${ch}.conf"
    out="$BASE/${name}_tun"
    n=$(ls "$out"/bench_rep*.log 2>/dev/null | wc -l)
    if [ "$n" -ge 5 ]; then
      echo "=== SKIP $name (already $n reps) $(date -u +%FT%TZ) ===" >> "$LOG"
      continue
    fi
    echo "=== ARM $name conf=$conf start $(date -u +%FT%TZ) ===" >> "$LOG"
    if ! RM_SUBSYS=INIT,ENV RM_CONF="$conf" "$DIR/run_many_p8k.sh" "$MODEL" "$name" tun 5 >> "$LOG" 2>&1; then
      echo "=== RETRY $name after server failure $(date -u +%FT%TZ) ===" >> "$LOG"
      if ! RM_SUBSYS=INIT,ENV RM_CONF="$conf" "$DIR/run_many_p8k.sh" "$MODEL" "$name" tun 5 >> "$LOG" 2>&1; then
        echo "=== SKIPPED $name: server failed twice $(date -u +%FT%TZ) ===" >> "$LOG"
      fi
    fi
    echo "=== ARM $name end $(date -u +%FT%TZ) ===" >> "$LOG"
  done
done
echo "=== campaign7 SCREENING complete $(date -u +%FT%TZ) ===" >> "$LOG"
