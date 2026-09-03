#!/bin/bash
# In-model MSCCL A/B — gpt-oss-120b on node 8 (2026-09-03 evening).
# Q: is disabling MSCCL (required for tuner rules to fire) a net loss vs deployment default?
# Arms (all --disable-custom-all-reduce, decode-heavy ISL512/OSL512/conc32):
#   mon_def  : MSCCL ON  (deployment default), no plugin
#   moff_def : MSCCL OFF, no plugin            (our campaigns' baseline)
#   moff_rule: MSCCL OFF, plugin + gptoss decode rule ring/simple/8 @184320
# 3 rotated passes x 6 reps per arm per pass = 18 reps/arm. Quiet SUBSYS (INIT,ENV).
set -u
S=/opt/shared/ylerman/GPU-107/infer-2026-08-30
DRV=$S/infer_many2.sh
B=/data/ylerman/models-2026-08-30
L=$B/msccl_ab.log
MODEL=$(ls -d /huggingface/hub/models--openai--gpt-oss-120b/snapshots/*/ | head -1)
XTRA="--attention-backend triton"
mkdir -p "$B"

run_one() {  # $1=arm-name $2=pass
  d=mab_${1}_p$2
  if [ -e "$B/${d}_def/DONE" ] || [ -e "$B/${d}_tun/DONE" ]; then echo "skip $d" | tee -a "$L"; return; fi
  echo "=== $(date -u +%F' '%H:%M) $d start ===" | tee -a "$L"
  case $1 in
    mon_def)   RM_MSCCL=1 RM_SUBSYS=INIT,ENV timeout 4500 "$DRV" "$MODEL" "$d" def 6 $XTRA >> "$L" 2>&1; arm=def;;
    moff_def)  RM_MSCCL=0 RM_SUBSYS=INIT,ENV timeout 4500 "$DRV" "$MODEL" "$d" def 6 $XTRA >> "$L" 2>&1; arm=def;;
    moff_rule) RM_MSCCL=0 RM_SUBSYS=INIT,ENV RM_CONF=sw_gdec_ring_simple_8.conf timeout 4500 "$DRV" "$MODEL" "$d" tun 6 $XTRA >> "$L" 2>&1; arm=tun;;
  esac
  mkdir -p "$B/${d}_${arm}"; touch "$B/${d}_${arm}/DONE"
  gzip -f "$B/${d}_${arm}/logs/"*.log 2>/dev/null
}

echo "==== MSCCL-AB start $(date -u +%F' '%H:%M) ====" | tee -a "$L"
for pass in 1 2 3; do
  case $pass in 1) order="mon_def moff_def moff_rule";;
                2) order="moff_def moff_rule mon_def";;
                3) order="moff_rule mon_def moff_def";; esac
  for a in $order; do run_one $a $pass; done
done
# canary: detect-mode 1 rep with MSCCL ON + rule conf — does the tuner get consulted at all?
d=mab_canary_mon_rule
if [ ! -e "$B/${d}_tun/DONE" ]; then
  echo "=== canary mon+rule ===" | tee -a "$L"
  RM_MSCCL=1 RM_SUBSYS=INIT,TUNING,ENV RM_CONF=sw_gdec_ring_simple_8.conf timeout 4500 "$DRV" "$MODEL" "$d" tun 1 $XTRA >> "$L" 2>&1
  mkdir -p "$B/${d}_tun"; touch "$B/${d}_tun/DONE"; gzip -f "$B/${d}_tun/logs/"*.log 2>/dev/null
fi
echo "==== MSCCL-AB DONE $(date -u +%F' '%H:%M) ====" | tee -a "$L"
