#!/bin/bash
# Overnight campaign 2 (2026-08-30, node 5): 30-repeat tuned-vs-default A/B for
#   gptoss120b  — retry with --page-size 64 (page_size=1 crashed batch_prefill: KV pool
#                 21.6M pages > INT32_MAX byte offsets, no matching kernel; see
#                 gptoss120b_def_ps1fail/server.log)
#   qwen235b    — Qwen3-235B-A22B-Instruct-2507-FP8, hidden 4096 -> 8192B all_reduce, rule exists
#   qwen397b    — Qwen3.5-397B-A17B, hidden 4096 -> 8192B all_reduce, rule exists
# Bench params unchanged from campaign 1 (ISL 512 / OSL 512 / conc 32 / 128 prompts, 30 reps).
# NCCL debug logs are gzipped after each arm: campaign-1 tun arms left 47-62G of logs each
# and the disk is at 98%.
# Restart-safe: each attempted arm leaves an ATTEMPTED marker and is skipped on rerun.
set -u
DRV=/opt/shared/ylerman/GPU-107/infer-2026-08-30/run_many.sh
REPS=${1:-30}
BASE=/data/ylerman/models-2026-08-30
LOG=$BASE/campaign2.log

run_model() {  # $1=path $2=name $3=extra
  for arm in def tun; do
    if [ -e "$BASE/${2}_${arm}/ATTEMPTED" ]; then
      echo "=== skip $2/$arm (already attempted) ===" | tee -a "$LOG"
      continue
    fi
    echo "=== $(date -u +%F' '%H:%M) $2/$arm start (df: $(df -h --output=avail /data | tail -1 | tr -d ' ')) ===" | tee -a "$LOG"
    timeout 9000 "$DRV" "$1" "$2" "$arm" "$REPS" ${3:-} 2>&1 | tee -a "$LOG"
    mkdir -p "$BASE/${2}_${arm}" && touch "$BASE/${2}_${arm}/ATTEMPTED"
    # hits already counted by run_many.sh; compress the raw NCCL logs to save the disk
    if ls "$BASE/${2}_${arm}/logs/"*.log >/dev/null 2>&1; then
      gzip "$BASE/${2}_${arm}/logs/"*.log
    fi
    echo "=== $(date -u +%F' '%H:%M) $2/$arm end ===" | tee -a "$LOG"
  done
}

echo "=== CAMPAIGN2 $(date -u +%F' '%H:%M) start/restart ===" | tee -a "$LOG"

# step 0a: preserve campaign-1 gptoss failure evidence before run_many.sh wipes the dirs
for arm in def tun; do
  if [ -d "$BASE/gptoss120b_$arm" ] && [ ! -e "$BASE/gptoss120b_${arm}_ps1fail" ]; then
    mv "$BASE/gptoss120b_$arm" "$BASE/gptoss120b_${arm}_ps1fail"
  fi
done

# step 0b: reclaim disk — compress campaign-1 NCCL logs (kept, just gzipped)
for d in "$BASE"/llama8b_* "$BASE"/qwen30b_*; do
  ls "$d/logs/"*.log >/dev/null 2>&1 && gzip "$d/logs/"*.log
done
echo "disk after gzip: $(df -h --output=avail /data | tail -1 | tr -d ' ') free" | tee -a "$LOG"

# qwen models first: rules fire for both (8192B all_reduce) — the valuable data.
# gptoss last: control only — its server enables "Aiter AllReduce Fusion" (gptossps_def/server.log),
# absent for llama/qwen, so its all_reduces likely never reach RCCL's tuner at all.
Q235=$(ls -d /huggingface/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507-FP8/snapshots/*/ 2>/dev/null | head -1)
[ -n "$Q235" ] && run_model "$Q235" qwen235b ""

Q397=$(ls -d /huggingface/hub/models--Qwen--Qwen3.5-397B-A17B/snapshots/*/ 2>/dev/null | head -1)
[ -n "$Q397" ] && run_model "$Q397" qwen397b ""

GPTOSS_EXTRA_FILE=/data/ylerman/models-2026-08-30/gptoss_extra_args
if [ -s "$GPTOSS_EXTRA_FILE" ]; then
  GPTOSS=$(ls -d /huggingface/hub/models--openai--gpt-oss-120b/snapshots/*/ 2>/dev/null | head -1)
  [ -n "$GPTOSS" ] && run_model "$GPTOSS" gptoss120b "$(cat "$GPTOSS_EXTRA_FILE")"
else
  echo "gptoss skipped: no verified fix in $GPTOSS_EXTRA_FILE" | tee -a "$LOG"
fi

echo "CAMPAIGN2 DONE $(date -u +%F' '%H:%M)" | tee -a "$LOG"
