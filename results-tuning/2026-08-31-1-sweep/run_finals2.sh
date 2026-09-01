#!/bin/bash
# Verified finals v2 (llama8b @262144B, node 3) — lessons from the retracted first finals:
#  * only algo/proto combos PROVEN to apply (logged canary probes; ll128 and tree+simple are
#    plugin-IGNOREd on this arch and silently run as no-rule)
#  * interleaved passes to average time drift (the fake winner beat the real no-rule arm by
#    +2% purely on drift): each arm runs 3 passes x 10 reps, pass order rotated.
# Arms are given as "name:def::" or "name:tun:<conf>:".
set -u
DRV=/opt/shared/ylerman/GPU-107/infer-2026-08-30/run_many.sh
MODEL=${F2_MODEL:-/data/mlperf_llama31_8b/model}
B=/data/ylerman/models-2026-08-30
L=$B/campaign6.log
ARMS="$@"

run_one() {  # $1=arm-spec $2=pass
  name=${1%%:*}; rest=${1#*:}; arm=${rest%%:*}; conf=${rest#*:}; conf=${conf%%:*}
  d=${F2_PFX:-lf2}_${name}_p$2
  if [ -e "$B/${d}/ATTEMPTED" ] || [ -e "$B/${d}_${arm}/ATTEMPTED" ]; then echo "skip $d" | tee -a "$L"; return; fi
  echo "=== $(date -u +%F' '%H:%M) $d start ===" | tee -a "$L"
  if [ "$arm" = tun ]; then
    RM_SUBSYS=INIT,ENV RM_CONF=$conf timeout 3600 "$DRV" "$MODEL" "$d" tun 10 >> "$L" 2>&1
  else
    RM_SUBSYS=INIT,ENV timeout 3600 "$DRV" "$MODEL" "$d" def 10 >> "$L" 2>&1
  fi
  mkdir -p "$B/${d}_${arm}"; touch "$B/${d}_${arm}/ATTEMPTED"
}

echo "==== FINALS2 start $(date -u +%F' '%H:%M) arms: $ARMS ====" | tee -a "$L"
n=0
for pass in 1 2 3; do
  # rotate order each pass
  set -- $ARMS
  case $pass in
    2) shift_count=2;;
    3) shift_count=4;;
    *) shift_count=0;;
  esac
  rotated=$(printf '%s\n' "$@" | awk -v s=$shift_count 'BEGIN{i=0}{a[i++]=$0}END{for(j=0;j<i;j++)print a[(j+s)%i]}')
  for spec in $rotated; do run_one "$spec" "$pass"; done
done

echo "FINAL2 MEDIANS (30 reps = 3 passes x 10):" | tee -a "$L"
for spec in $ARMS; do
  name=${spec%%:*}; rest=${spec#*:}; arm=${rest%%:*}
  m=$(cat "$B"/${F2_PFX:-lf2}_${name}_p*_${arm}/bench_rep*.log 2>/dev/null | grep "Output token throughput" | grep -oE "[0-9.]+" | sort -n | awk '{v[NR]=$1}END{if(NR)print (NR%2?v[(NR+1)/2]:(v[NR/2]+v[NR/2+1])/2)" n="NR}')
  echo "$name $m" | tee -a "$L"
done
echo "==== FINALS2 DONE $(date -u +%F' '%H:%M) ====" | tee -a "$L"
