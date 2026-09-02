#!/bin/bash
# Attribution experiment (2026-09-02, node 5): decompose Regev's -30% TTFT claim.
# Three arms, run at TWO of his input lengths (4k = his peak window, 8k = his probe window),
# gpt-oss-120b, conc 32, OSL 128, radix cache DISABLED everywhere (campaign8 flaw), triton attn.
#   A customAR      : SGLang's built-in all-reduce kernel (his baseline; IM_KEEP_CUSTOM_AR=1)
#   B rccl_default  : --disable-custom-all-reduce, no rules (our null baseline)
#   C rccl_tree     : disabled + band rule tree/-1/-1 over 32-96MiB (his NCCL_ALGO=Tree, per-size)
# 3 rotated passes x 10 reps per arm per ISL; arms differ by server flag so restart-per-arm-pass
# (the drift-controlled finals-v2 pattern). Read-out: (A vs B) = engine swap, (B vs C) = Tree.
set -u
S=/opt/shared/ylerman/GPU-107/infer-2026-08-30
DRV=$S/infer_many.sh
B=/data/ylerman/models-2026-08-30
L=$B/attribution.log
MODEL=$(ls -d /huggingface/hub/models--openai--gpt-oss-120b/snapshots/*/ | head -1)
XTRA="--attention-backend triton --disable-radix-cache"

CONF=sw_att_tree_wild.conf
if [ ! -f "$S/$CONF" ]; then
  { echo "collective_type,min_bytes,max_bytes,algorithm,protocol,channels,nNodes,nRanks,numPipeOps,regBuff";
    echo "allreduce,33554432,100663296,tree,-1,-1,1,8,-1,-1"; } > "$S/$CONF"
fi

run_one() {  # $1=isl $2=arm-name $3=pass
  d=att${1}_${2}_p$3
  if [ -e "$B/${d}_def/ATTEMPTED" ] || [ -e "$B/${d}_tun/ATTEMPTED" ]; then echo "skip $d" | tee -a "$L"; return; fi
  echo "=== $(date -u +%F' '%H:%M) $d start ===" | tee -a "$L"
  case $2 in
    customAR)     IM_ISL=$((1024*$1)) IM_OSL=128 RM_SUBSYS=INIT,ENV IM_KEEP_CUSTOM_AR=1 timeout 3600 "$DRV" "$MODEL" "$d" def 10 $XTRA >> "$L" 2>&1; arm=def;;
    rccl_default) IM_ISL=$((1024*$1)) IM_OSL=128 RM_SUBSYS=INIT,ENV timeout 3600 "$DRV" "$MODEL" "$d" def 10 $XTRA >> "$L" 2>&1; arm=def;;
    rccl_tree)    IM_ISL=$((1024*$1)) IM_OSL=128 RM_SUBSYS=INIT,ENV RM_CONF=$CONF timeout 3600 "$DRV" "$MODEL" "$d" tun 10 $XTRA >> "$L" 2>&1; arm=tun;;
  esac
  mkdir -p "$B/${d}_${arm}"; touch "$B/${d}_${arm}/ATTEMPTED"
}

echo "==== ATTRIBUTION start $(date -u +%F' '%H:%M) ====" | tee -a "$L"
for isl in 4 8; do
  for pass in 1 2 3; do
    case $pass in 1) order="customAR rccl_default rccl_tree";;
                  2) order="rccl_default rccl_tree customAR";;
                  3) order="rccl_tree customAR rccl_default";; esac
    for a in $order; do run_one $isl $a $pass; done
  done
done

# selection-evidence canaries (detect mode, 1 rep each, ISL 8k)
for a in customAR rccl_tree; do
  d=attcan_${a}
  if [ ! -e "$B/${d}_def/ATTEMPTED" ] && [ ! -e "$B/${d}_tun/ATTEMPTED" ]; then
    echo "=== canary $a ===" | tee -a "$L"
    case $a in
      customAR)  IM_ISL=8192 IM_OSL=128 IM_KEEP_CUSTOM_AR=1 timeout 3600 "$DRV" "$MODEL" "$d" def 1 $XTRA >> "$L" 2>&1; arm=def;;
      rccl_tree) IM_ISL=8192 IM_OSL=128 RM_CONF=$CONF timeout 3600 "$DRV" "$MODEL" "$d" tun 1 $XTRA >> "$L" 2>&1; arm=tun;;
    esac
    mkdir -p "$B/${d}_${arm}"; touch "$B/${d}_${arm}/ATTEMPTED"
    gzip "$B/${d}_${arm}/logs/"*.log 2>/dev/null
  fi
done
echo "==== ATTRIBUTION DONE $(date -u +%F' '%H:%M) ====" | tee -a "$L"
