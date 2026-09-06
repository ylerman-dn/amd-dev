#!/bin/bash
# Verify campaign (2026-09-06 night): high-power re-test of the modes-campaign null on the
# 3 window modes, plus a true-deployment reference arm. Per mode:
#   A) paired server (flag REMOVED, MSCCL off, plugin hot-reload): rule(gpu107_1n_v3.conf)
#      vs none, 20 rounds/arm interleaved  -> drift-controlled core A/B, n=19 warm.
#   B) deployment-env server (image env intact: NCCL_MIN_NCHANNELS=112 present, MSCCL ON,
#      no plugin; custom-AR still disabled like all campaign arms): 20 reps -> reference.
# Metrics scored at analysis: tok/s, TTFT (mean/median/P99), TPOT.
#
# Usage: verify_modes.sh <model-path> <tag> [extra-server-args]
set -u
MODEL=$1
TAG=$2
XTRA="${3:-}"
S=/opt/shared/ylerman/GPU-107/infer-2026-08-30
B=${IM_BASE:-/data/ylerman/models-2026-08-30}
L=$B/verify_${TAG}.log
mkdir -p "$B"

mode_params() {
  MXTRA="$XTRA"
  case $1 in
    d128) ISL=512;  OSL=512; CONC=128; NP=384;;
    d256) ISL=512;  OSL=512; CONC=256; NP=512;;
    p8k)  ISL=8192; OSL=128; CONC=32;  NP=96; MXTRA="$MXTRA --disable-radix-cache";;
  esac
}

echo "==== VERIFY $TAG start $(date -u +%F' '%H:%M) ====" | tee -a "$L"
for m in d128 d256 p8k; do
  mode_params "$m"
  if [ ! -e "$B/${TAG}v_${m}_rule/VDONE" ]; then
    echo "=== $m paired (rule vs none, 20 rounds) ===" | tee -a "$L"
    IM_DROP_FLOOR=1 IM_ISL=$ISL IM_OSL=$OSL IM_CONC=$CONC IM_NPROMPTS=$NP IM_EXTRA="$MXTRA" \
      timeout 14400 "$S/infer_paired.sh" "$MODEL" "${TAG}v_${m}" 20 "rule:gpu107_1n_v3.conf" "none:NONE" >> "$L" 2>&1
    mkdir -p "$B/${TAG}v_${m}_rule"; touch "$B/${TAG}v_${m}_rule/VDONE"
  fi
  if [ ! -e "$B/${TAG}v_${m}dep_def/VDONE" ]; then
    echo "=== $m deployment arm (flag present, MSCCL on, 20 reps) ===" | tee -a "$L"
    RM_MSCCL=1 RM_SUBSYS=INIT,ENV IM_ISL=$ISL IM_OSL=$OSL IM_CONC=$CONC IM_NPROMPTS=$NP \
      timeout 14400 "$S/infer_many2.sh" "$MODEL" "${TAG}v_${m}dep" def 20 $MXTRA >> "$L" 2>&1
    mkdir -p "$B/${TAG}v_${m}dep_def"; touch "$B/${TAG}v_${m}dep_def/VDONE"
  fi
done
echo "==== VERIFY $TAG DONE $(date -u +%F' '%H:%M) ====" | tee -a "$L"
