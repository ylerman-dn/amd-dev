#!/bin/bash
# Full campaign: 30-repeat tuned-vs-default A/B across several models, node 5.
# Each model runs the default arm first, then the tuned arm, both with the server started once.
# gpt-oss is included deliberately: hidden_size 2880 gives a 5760-byte all_reduce, which no
# shipped rule matches (they are exact powers of two), so it is the control showing what
# happens when the rules cannot fire.
set -u
DRV=/opt/shared/ylerman/GPU-107/infer-2026-08-30/run_many.sh
REPS=${1:-30}
LOG=/data/ylerman/models-2026-08-30/campaign.log

run_model() {  # $1=path $2=name $3=extra
  for arm in def tun; do
    echo "=== $(date -u +%H:%M) $2/$arm ===" | tee -a "$LOG"
    timeout 9000 "$DRV" "$1" "$2" "$arm" "$REPS" ${3:-} 2>&1 | tee -a "$LOG"
  done
}

mkdir -p /data/ylerman/models-2026-08-30
: > "$LOG"

run_model /data/mlperf_llama31_8b/model llama8b ""

QWEN=$(ls -d /huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/*/ 2>/dev/null | head -1)
[ -n "$QWEN" ] && run_model "$QWEN" qwen30b ""

GPTOSS=$(ls -d /huggingface/hub/models--openai--gpt-oss-120b/snapshots/*/ 2>/dev/null | head -1)
[ -n "$GPTOSS" ] && run_model "$GPTOSS" gptoss120b ""

echo "CAMPAIGN DONE $(date -u +%H:%M)" | tee -a "$LOG"
