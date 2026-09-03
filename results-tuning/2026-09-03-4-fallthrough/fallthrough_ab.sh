#!/bin/bash
# In-model plugin-fallthrough A/B — gpt-oss-120b on amd-mi350x-ses2-1 (2026-09-03 evening).
# (Planned on qwen30b, but the 350X hub has only stub dirs for it; gptoss weights are full.)
# Q: does merely LOADING the tuner plugin (zero matching rules) cost tok/s vs no plugin?
# Motivated by bracket-sweep finding: rule-free sizes under the plugin selected RING/LL vs
# default TREE/LL and lost 19-27% busbw (rccl-tests). Arms:
#   def : no plugin
#   ftr : plugin + sw_empty.conf (header only, zero rules) -> pure fallthrough path
# 3 rotated passes x 6 reps per arm per pass = 18 reps/arm. MSCCL off both (campaign standard).
set -u
S=/opt/shared/ylerman/GPU-107/infer-2026-08-30
DRV=$S/infer_many2.sh
B=/data/ylerman/models-2026-08-30
L=$B/fallthrough_ab.log
MODEL=$(ls -d /huggingface/hub/models--openai--gpt-oss-120b/snapshots/*/ | head -1)
XTRA="--attention-backend triton"
mkdir -p "$B"

run_one() {  # $1=arm-name $2=pass
  d=ftr_${1}_p$2
  if [ -e "$B/${d}_def/DONE" ] || [ -e "$B/${d}_tun/DONE" ]; then echo "skip $d" | tee -a "$L"; return; fi
  echo "=== $(date -u +%F' '%H:%M) $d start ===" | tee -a "$L"
  case $1 in
    def) RM_SUBSYS=INIT,ENV timeout 4500 "$DRV" "$MODEL" "$d" def 6 $XTRA >> "$L" 2>&1; arm=def;;
    ftr) RM_SUBSYS=INIT,ENV RM_CONF=sw_empty.conf timeout 4500 "$DRV" "$MODEL" "$d" tun 6 $XTRA >> "$L" 2>&1; arm=tun;;
  esac
  mkdir -p "$B/${d}_${arm}"; touch "$B/${d}_${arm}/DONE"
  gzip -f "$B/${d}_${arm}/logs/"*.log 2>/dev/null
}

echo "==== FALLTHROUGH-AB start $(date -u +%F' '%H:%M) ====" | tee -a "$L"
for pass in 1 2 3; do
  case $pass in 1) order="def ftr";; 2) order="ftr def";; 3) order="def ftr";; esac
  for a in $order; do run_one $a $pass; done
done
# canary: detect mode 1 rep, ftr arm — verify plugin loaded, 0 configs, consultations happen
d=ftr_canary
if [ ! -e "$B/${d}_tun/DONE" ]; then
  RM_SUBSYS=INIT,TUNING,ENV RM_CONF=sw_empty.conf timeout 4500 "$DRV" "$MODEL" "$d" tun 1 $XTRA >> "$L" 2>&1
  mkdir -p "$B/${d}_tun"; touch "$B/${d}_tun/DONE"; gzip -f "$B/${d}_tun/logs/"*.log 2>/dev/null
fi
echo "==== FALLTHROUGH-AB DONE $(date -u +%F' '%H:%M) ====" | tee -a "$L"
