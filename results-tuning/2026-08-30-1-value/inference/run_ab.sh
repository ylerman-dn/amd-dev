#!/bin/bash
# Inference A/B with MSCCL disabled, so the model's all_reduces take the normal RCCL path
# where the tuner is consulted. Alternates arms so drift hits both equally.
#   def = stock RCCL, no tuner plugin
#   tun = stock RCCL + DN tuner + shipped all_reduce_1n.final.conf
# Usage: run_ab.sh <repeats>
set -u
REPS=${1:-3}
DRV=/opt/shared/ylerman/GPU-107/infer-2026-08-30/run_infer_nomsccl.sh

for r in $(seq 1 "$REPS"); do
  for arm in def tun; do
    tag="ab_${arm}_rep${r}"
    "$DRV" "$arm" "$tag" rccl-ar stock > /dev/null 2>&1
    tps=$(grep -oP 'Output token throughput \(tok/s\):\s+\K[0-9.]+' \
          "/data/ylerman/infer-2026-08-30/$tag/bench.log" 2>/dev/null)
    tpot=$(grep -oP 'Mean TPOT \(ms\):\s+\K[0-9.]+' \
          "/data/ylerman/infer-2026-08-30/$tag/bench.log" 2>/dev/null)
    hits=$(grep -h -c "Applied config" \
          "/data/ylerman/infer-2026-08-30/$tag"/logs/*.log 2>/dev/null | paste -sd+ | bc)
    printf "%-16s tok/s=%-9s TPOT=%-7s tuner_hits=%s\n" \
      "$tag" "${tps:-FAIL}" "${tpot:-—}" "${hits:-0}"
  done
done
echo "AB DONE"
