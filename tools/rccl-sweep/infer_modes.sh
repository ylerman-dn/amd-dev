#!/bin/bash
# Multi-mode in-model A/B campaign (2026-09-06): run ~9 traffic modes on one model, each
# mode A/B'ing a tuner conf vs no rules with the paired driver (one server per mode,
# hot-reload conf swaps, round-robin rounds). Different modes produce different all_reduce
# message sizes: decode size = concurrency x hidden x 2 (exact, repeated), prefill sizes =
# packed-chunk bands (radix cache disabled so prefill actually recomputes).
#
# Usage: infer_modes.sh <model-path> <model-tag> <conf-name> [rounds-per-arm]
# Env:   MODES        override the mode list (space-separated names from the table below)
#        IM_XTRA_SRV  extra server args applied to ALL modes (e.g. --attention-backend triton)
#        + the infer_paired.sh knobs (IM_BASE, IM_CONF_DIR, IM_IMAGE, ...)
#
# Modes (name / ISL / OSL / conc / radix):
#   d4 d8 d16 d32 d64 d128 d256 : 512/512/{4..256}/on — decode-heavy ladder
#   p2k                      : 2048/128/32/OFF        — prefill band ~small
#   p8k                      : 8192/128/32/OFF        — prefill band ~large
#   mix                      : 2048/512/64/OFF        — mixed prefill+decode
set -u
MODEL=$1
TAG=$2
CONF=$3
ROUNDS=${4:-6}
S=$(cd "$(dirname "$0")" && pwd)
PAIRED=${IM_PAIRED:-/opt/shared/ylerman/GPU-107/infer-2026-08-30/infer_paired.sh}
B=${IM_BASE:-/data/ylerman/models-2026-08-30}
L=$B/modes_${TAG}.log
MODES=${MODES:-"d4 d8 d16 d32 d64 d128 d256 p2k p8k mix"}
mkdir -p "$B"

mode_params() { # sets ISL OSL CONC NP XTRA per mode
  XTRA="${IM_XTRA_SRV:-}"
  case $1 in
    d4)   ISL=512;  OSL=512; CONC=4;   NP=32;;
    d8)   ISL=512;  OSL=512; CONC=8;   NP=64;;
    d16)  ISL=512;  OSL=512; CONC=16;  NP=64;;
    d32)  ISL=512;  OSL=512; CONC=32;  NP=128;;
    d64)  ISL=512;  OSL=512; CONC=64;  NP=256;;
    d128) ISL=512;  OSL=512; CONC=128; NP=384;;
    d256) ISL=512;  OSL=512; CONC=256; NP=512;;
    p2k)  ISL=2048; OSL=128; CONC=32;  NP=128; XTRA="$XTRA --disable-radix-cache";;
    p8k)  ISL=8192; OSL=128; CONC=32;  NP=96;  XTRA="$XTRA --disable-radix-cache";;
    mix)  ISL=2048; OSL=512; CONC=64;  NP=192; XTRA="$XTRA --disable-radix-cache";;
    *) echo "unknown mode $1" >&2; return 1;;
  esac
}

echo "==== MODES CAMPAIGN $TAG start $(date -u +%F' '%H:%M) conf=$CONF rounds=$ROUNDS modes=[$MODES] ====" | tee -a "$L"
for m in $MODES; do
  mode_params "$m" || continue
  if [ -e "$B/${TAG}_${m}_rule/CAMPAIGN_DONE" ]; then echo "skip $m (done)" | tee -a "$L"; continue; fi
  echo "=== mode $m: ISL=$ISL OSL=$OSL conc=$CONC np=$NP xtra='$XTRA' ===" | tee -a "$L"
  IM_ISL=$ISL IM_OSL=$OSL IM_CONC=$CONC IM_NPROMPTS=$NP IM_EXTRA="$XTRA" \
    timeout 7200 "$PAIRED" "$MODEL" "${TAG}_${m}" "$ROUNDS" "rule:$CONF" "none:NONE" >> "$L" 2>&1
  mkdir -p "$B/${TAG}_${m}_rule"; touch "$B/${TAG}_${m}_rule/CAMPAIGN_DONE"
  # 1-round detect canary per mode: which sizes, did the rule fire (TUNING logs)
  if [ ! -e "$B/${TAG}_${m}can_rule/CAMPAIGN_DONE" ]; then
    IM_ISL=$ISL IM_OSL=$OSL IM_CONC=$CONC IM_NPROMPTS=$NP IM_EXTRA="$XTRA" RM_SUBSYS=INIT,TUNING,ENV \
      timeout 3600 "$PAIRED" "$MODEL" "${TAG}_${m}can" 1 "rule:$CONF" >> "$L" 2>&1
    mkdir -p "$B/${TAG}_${m}can_rule"; touch "$B/${TAG}_${m}can_rule/CAMPAIGN_DONE"
    gzip -f "$B/${TAG}_${m}can_srv/logs/"*.log 2>/dev/null
  fi
done
echo "==== MODES CAMPAIGN $TAG DONE $(date -u +%F' '%H:%M) ====" | tee -a "$L"
