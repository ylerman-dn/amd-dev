#!/bin/bash
# In-model per-size rule sweep (2026-08-31 overnight), node 5.
# For one model: screen a grid of single-rule confs at the model's dominant all_reduce
# size (5 quiet reps each), then run 30-rep finals for the top-3 screened combos plus
# two references: "no rule" (default arm, no plugin) and the currently shipped conf.
# All arms quiet (RM_SUBSYS=INIT,ENV); bench params frozen (ISL 512/OSL 512/conc 32/128 prompts).
# Restart-safe: per-arm ATTEMPTED markers; driver skips completed arms.
#
# Usage: run_sweep.sh <model-path> <prefix> <size_bytes> <current_conf_filename>
set -u
MODEL=$1
PFX=$2
SIZE=$3
CURCONF=$4
DRV=/opt/shared/ylerman/GPU-107/infer-2026-08-30/run_many.sh
CONFDIR=/opt/shared/ylerman/GPU-107/infer-2026-08-30
B=/data/ylerman/models-2026-08-30
L=$B/campaign5.log

med() { sort -n | awk '{v[NR]=$1} END{if(NR)print (NR%2? v[(NR+1)/2] : (v[NR/2]+v[NR/2+1])/2); else print 0}'; }
arm_median() { grep -h "Output token throughput" "$B/$1/bench_rep"*.log 2>/dev/null | grep -oE "[0-9.]+" | med; }

run_arm() {  # $1=arm-dir-name $2=def|tun $3=reps $4=conf-filename(for tun)
  if [ -e "$B/$1/ATTEMPTED" ]; then echo "skip $1" | tee -a "$L"; return; fi
  echo "=== $(date -u +%F' '%H:%M) $PFX arm $1 ($2, $3 reps, conf=${4:-none}) ===" | tee -a "$L"
  if [ "$2" = tun ]; then
    RM_SUBSYS=INIT,ENV RM_CONF=$4 timeout 7200 "$DRV" "$MODEL" "$1" tun "$3" >> "$L" 2>&1
  else
    RM_SUBSYS=INIT,ENV timeout 7200 "$DRV" "$MODEL" "$1" def "$3" >> "$L" 2>&1
  fi
  # run_many.sh names the dir <name>_<arm>; normalize to plain arm dir via symlink-free rename
  if [ -d "$B/${1}_$2" ] && [ ! -d "$B/$1" ]; then mv "$B/${1}_$2" "$B/$1"; fi
  mkdir -p "$B/$1"; touch "$B/$1/ATTEMPTED"
}

echo "==== SWEEP $PFX start $(date -u +%F' '%H:%M) size=$SIZE ====" | tee -a "$L"

# ---- screening: 30 combos, 5 reps each ----
COMBOS=""
for algo in ring tree; do for proto in ll ll128 simple; do for ch in 8 16 24 32 48; do
  COMBOS="$COMBOS ${algo}_${proto}_${ch}"
done; done; done

for c in $COMBOS; do
  algo=${c%%_*}; rest=${c#*_}; proto=${rest%%_*}; ch=${rest##*_}
  cf=sw_${PFX}_${c}.conf
  if [ ! -f "$CONFDIR/$cf" ]; then
    { echo "collective_type,min_bytes,max_bytes,algorithm,protocol,channels,nNodes,nRanks,numPipeOps,regBuff";
      echo "allreduce,$SIZE,$SIZE,$algo,$proto,$ch,1,8,-1,-1"; } > "$CONFDIR/$cf"
  fi
  run_arm "${PFX}scr_${c}" tun 5 "$cf"
done

# ---- pick top-3 by median ----
RANK=$(for c in $COMBOS; do echo "$(arm_median ${PFX}scr_${c}) $c"; done | sort -rn)
echo "SCREEN RANKING $PFX:" | tee -a "$L"; echo "$RANK" | tee -a "$L"
TOP3=$(echo "$RANK" | head -3 | awk '{print $2}')

# ---- finals: top-3 + no-rule + current conf, 30 reps each ----
for c in $TOP3; do run_arm "${PFX}fin_${c}" tun 30 "sw_${PFX}_${c}.conf"; done
run_arm "${PFX}fin_norule" def 30 ""
run_arm "${PFX}fin_current" tun 30 "$CURCONF"

echo "FINAL MEDIANS $PFX:" | tee -a "$L"
for d in "$B/${PFX}fin_"*/; do n=$(basename "$d"); echo "$n $(arm_median "$n")" | tee -a "$L"; done
echo "==== SWEEP $PFX DONE $(date -u +%F' '%H:%M) ====" | tee -a "$L"
